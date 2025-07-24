from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as torchd
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D, ConditionalResidualBlock1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
import diffusion_policy.policy.sailor_tools as tools 

import pdb


def gradient_penalty(
    learner_sa: torch.Tensor,
    expert_sa: torch.Tensor,
    f: nn.Module,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Calculates the gradient penalty for the given learner and expert state-action tensors.

    Args:
        learner_sa (torch.Tensor): The state-action tensor from the learner.
        expert_sa (torch.Tensor): The state-action tensor from the expert.
        f (nn.Module): The discriminator network.
        device (str, optional): The device to use. Defaults to "cuda".

    Returns:
        torch.Tensor: The gradient penalty.
    """
    batch_size = expert_sa.size()[0]

    alpha = torch.rand(batch_size, 1).to(device)
    alpha = alpha.expand_as(expert_sa)

    interpolated = alpha * expert_sa.data + (1 - alpha) * learner_sa.data

    interpolated = Variable(interpolated, requires_grad=True).to(device)

    f_interpolated = f(interpolated.float()).mode().to(device)

    gradients = torch_grad(
        outputs=f_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(f_interpolated.size()).to(device),
        create_graph=True,
        retain_graph=True,
    )[0].to(device)

    gradients = gradients.view(batch_size, -1)

    gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)
    # 2 * |f'(x_0)|
    return ((gradients_norm - 0.4) ** 2).mean()

# class MLP taken from SAILOR Paper
class MLP(nn.Module):
    def __init__(
        self,
        inp_dim,
        shape,
        layers,
        units,
        act="SiLU",
        norm=True,
        dist="normal",
        std=1.0,
        min_std=0.1,
        max_std=1.0,
        absmax=None,
        temp=0.1,
        unimix_ratio=0.01,
        outscale=1.0,
        symlog_inputs=False,
        device="cuda",
        name="NoName",
    ):
        super(MLP, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)
        act = getattr(torch.nn, act)
        self._dist = dist
        self._std = std if isinstance(std, str) else torch.tensor((std,), device=device)
        self._min_std = min_std
        self._max_std = max_std
        self._absmax = absmax
        self._temp = temp
        self._unimix_ratio = unimix_ratio
        self._symlog_inputs = symlog_inputs
        self._device = device

        self.layers = nn.Sequential()
        for i in range(layers):
            self.layers.add_module(
                f"{name}_linear{i}", nn.Linear(inp_dim, units, bias=False)
            )
            if norm:
                self.layers.add_module(
                    f"{name}_norm{i}", nn.LayerNorm(units, eps=1e-03)
                )
            self.layers.add_module(f"{name}_act{i}", act())
            if i == 0:
                inp_dim = units
        self.layers.apply(tools.weight_init)

        if isinstance(self._shape, dict):
            self.mean_layer = nn.ModuleDict()
            for name, shape in self._shape.items():
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.ModuleDict()
                for name, shape in self._shape.items():
                    self.std_layer[name] = nn.Linear(inp_dim, np.prod(shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))
        elif self._shape is not None:
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.Linear(units, np.prod(self._shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        if self._symlog_inputs:
            x = tools.symlog(x)
        out = self.layers(x)
        # Used for encoder output
        if self._shape is None:
            return out
        if isinstance(self._shape, dict):
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)
                if self._std == "learned":
                    std = self.std_layer[name](out)
                else:
                    std = self._std
                dists.update({name: self.dist(self._dist, mean, std, shape)})
            return dists
        else:
            mean = self.mean_layer(out)
            if self._std == "learned":
                std = self.std_layer(out)
            else:
                std = self._std
            return self.dist(self._dist, mean, std, self._shape)

    def dist(self, dist, mean, std, shape):
        if self._dist == "tanh_normal":
            mean = torch.tanh(mean)
            std = F.softplus(std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == "normal":
            std = (self._max_std - self._min_std) * torch.sigmoid(
                std + 2.0
            ) + self._min_std
            dist = torchd.normal.Normal(torch.tanh(mean), std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "normal_std_fixed":
            print("loc has NaN:", torch.isnan(mean).any().item())
            print("loc has Inf:", torch.isinf(mean).any().item())
            print("scale min:", self._std.min().item())
            print("scale max:", self._std.max().item())
            print("scale has NaN:", torch.isnan(self._std).any().item())
            print("scale has Inf:", torch.isinf(self._std).any().item())
            dist = torchd.normal.Normal(mean.float(), self._std.float())
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "trunc_normal":
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "onehot":
            dist = tools.OneHotDist(mean, unimix_ratio=self._unimix_ratio)
        elif self._dist == "onehot_gumble":
            dist = tools.ContDist(
                torchd.gumbel.Gumbel(mean, 1 / self._temp), absmax=self._absmax
            )
        elif dist == "huber":
            dist = tools.ContDist(
                torchd.independent.Independent(
                    tools.UnnormalizedHuber(mean, std, 1.0),
                    len(shape),
                    absmax=self._absmax,
                )
            )
        elif dist == "binary":
            dist = tools.Bernoulli(
                torchd.independent.Independent(
                    torchd.bernoulli.Bernoulli(logits=mean), len(shape)
                )
            )
        elif dist == "symlog_disc":
            dist = tools.DiscDist(logits=mean, device=self._device)
        elif dist == "symlog_mse":
            dist = tools.SymlogDist(mean)
        else:
            raise NotImplementedError(dist)
        return dist


class SmallQNetworkTanh(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.body(x)


class DiffusionUnetTimmPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: TimmObsEncoder,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            input_pertub=0.1,
            inpaint_fixed_action_prefix=False,
            train_diffusion_n_samples=1,
            gradient_penalty=False,
            gradient_penalty_weight=10.0,
            # parameters passed to step
            **kwargs
        ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']
        # get feature dim
        obs_feature_dim = np.prod(obs_encoder.output_shape())


        # create diffusion model
        assert obs_as_global_cond
        input_dim = action_dim
        global_cond_dim = obs_feature_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon # used for training
        self.obs_as_global_cond = obs_as_global_cond
        self.input_pertub = input_pertub
        self.inpaint_fixed_action_prefix = inpaint_fixed_action_prefix
        self.train_diffusion_n_samples = int(train_diffusion_n_samples)
        self.kwargs = kwargs

        # final_encoder = SmallQNetworkTanh(self.action_dim*self.action_horizon, 1)
        # self.final_encoder = final_encoder

        # now make use of the sailor stuff:

        # with outscale 0.0 nothing moves essentially,...
        # reward_discriminator = MLP(self.action_dim*self.action_horizon, (255,), 2, 512, 'SiLU', True, dist='normal_std_fixed', outscale=0.0, device=self.device, name='reward_discriminator')
        reward_discriminator = MLP(self.action_dim*self.action_horizon, (255,), 2, 512, 'SiLU', False, dist='normal_std_fixed', outscale=0.1, device=self.device, name='reward_discriminator')
        self.reward_discriminator = reward_discriminator

        self.gradient_penalty = gradient_penalty
        self.gradient_penalty_weight = gradient_penalty_weight

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data,
            condition_mask,
            local_cond=None,
            global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
        ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    # no grad needed here
    @torch.no_grad()
    def evaluate_action(self, obs_dict: Dict[str, torch.Tensor], fixed_action_prefix: torch.Tensor=None, actions=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        fixed_action_prefix: unnormalized action prefix
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = obs_dict
        B = next(iter(nobs.values())).shape[0]

        # condition through global feature
        global_cond = self.obs_encoder(nobs)

        # torch tensor from numpy
        actions = torch.from_numpy(actions).to(self.device, dtype=self.dtype) if isinstance(actions, np.ndarray) else actions

        batch_dim = actions.shape[0]
        

        trajectory = actions

        # adding of noise not needed here!
        # # Sample noise that we'll add to the images
        # noise = torch.randn(trajectory.shape, device=trajectory.device)
        # # input perturbation by adding additonal noise to alleviate exposure bias
        # # reference: https://github.com/forever208/DDPM-IP
        # noise_new = noise + self.input_pertub * torch.randn(trajectory.shape, device=trajectory.device)

        # Timesteps - here also keep them fixed! - to 0!
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (actions.shape[0],), device=trajectory.device
        ).long()*0

        # no noise scheduling needed!
        # # Add noise to the clean images according to the noise magnitude at each timestep
        # # (this is the forward diffusion process)
        # noisy_trajectory = self.noise_scheduler.add_noise(
        #     trajectory, noise_new, timesteps)
        
        # Predict the noise residual
        pred = self.model(
            trajectory,
            timesteps, 
            local_cond=None,
            global_cond=global_cond
        )

        # now basically reshape and pass thhrough the final layer:
        pred_prepared = torch.tanh(pred.reshape(batch_dim, -1))

        # now predict the quality
        qual_pred = torch.mean(self.reward_discriminator(pred_prepared).mode(), dim=1)
        
        return qual_pred




    def predict_action(self, obs_dict: Dict[str, torch.Tensor], fixed_action_prefix: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        fixed_action_prefix: unnormalized action prefix
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = obs_dict
        B = next(iter(nobs.values())).shape[0]

        # condition through global feature
        global_cond = self.obs_encoder(nobs)

        # empty data for action
        cond_data = torch.zeros(size=(B, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        if fixed_action_prefix is not None and self.inpaint_fixed_action_prefix:
            n_fixed_steps = fixed_action_prefix.shape[1]
            cond_data[:, :n_fixed_steps] = fixed_action_prefix
            cond_mask[:, :n_fixed_steps] = True
            cond_data = self.normalizer['action'].normalize(cond_data)


        # run sampling
        nsample = self.conditional_sample(
            condition_data=cond_data, 
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs)
        
        assert nsample.shape == (B, self.action_horizon, self.action_dim)
        
        return nsample
    
    def predict_action_with_infos(self, obs_dict: Dict[str, torch.Tensor], fixed_action_prefix: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        fixed_action_prefix: unnormalized action prefix
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = obs_dict
        B = next(iter(nobs.values())).shape[0]

        # condition through global feature
        global_cond = self.obs_encoder(nobs)

        # empty data for action
        cond_data = torch.zeros(size=(B, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        # pdb.set_trace()

        if fixed_action_prefix is not None and self.inpaint_fixed_action_prefix:
            n_fixed_steps = fixed_action_prefix.shape[1]
            cond_data[:, :n_fixed_steps] = fixed_action_prefix
            cond_mask[:, :n_fixed_steps] = True
            cond_data = self.normalizer['action'].normalize(cond_data)


        # run sampling
        nsample = self.conditional_sample(
            condition_data=cond_data, 
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs)
        
        assert nsample.shape == (B, self.action_horizon, self.action_dim)

        result = {
            'action': nsample,
            'action_pred': nsample,
            'global_cond': global_cond,
        }
        
        return nsample, result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        # pdb.set_trace()
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch0, batch1):
        # normalize input
        assert 'valid_mask' not in batch0
        # nobs = self.normalizer.normalize(batch['obs'])
        # nactions = self.normalizer['action'].normalize(batch['action'])
        nobs0 = batch0['obs']
        nactions0 = batch0['action']
        # pdb.set_trace()
        assert self.obs_as_global_cond
        global_cond0 = self.obs_encoder(nobs0)

        nobs1 = batch1['obs']
        nactions1 = batch1['action']
        global_cond1 = self.obs_encoder(nobs1)

        # do not do the multiple diffusion samples per observation here
        # # train on multiple diffusion samples per obs
        # if self.train_diffusion_n_samples != 1:
        #     # repeat obs features and actions multiple times along the batch dimension
        #     # each sample will later have a different noise sample, effecty training 
        #     # more diffusion steps per each obs encoder forward pass
        #     global_cond = torch.repeat_interleave(global_cond, 
        #         repeats=self.train_diffusion_n_samples, dim=0)
        #     nactions = torch.repeat_interleave(nactions, 
        #         repeats=self.train_diffusion_n_samples, dim=0)

        # now combine them rather:
        # COmbine them:
        global_cond_combined = torch.cat((global_cond0, global_cond1), dim=0)
        nactions_combined = torch.cat((nactions0, nactions1), dim=0)
        batch_dim = nactions_combined.shape[0]

        trajectory = nactions_combined

        # adding of noise not needed here!
        # # Sample noise that we'll add to the images
        # noise = torch.randn(trajectory.shape, device=trajectory.device)
        # # input perturbation by adding additonal noise to alleviate exposure bias
        # # reference: https://github.com/forever208/DDPM-IP
        # noise_new = noise + self.input_pertub * torch.randn(trajectory.shape, device=trajectory.device)

        # Timesteps - here also keep them fixed! - to 0!
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (nactions_combined.shape[0],), device=trajectory.device
        ).long()*0

        # no noise scheduling needed!
        # # Add noise to the clean images according to the noise magnitude at each timestep
        # # (this is the forward diffusion process)
        # noisy_trajectory = self.noise_scheduler.add_noise(
        #     trajectory, noise_new, timesteps)
        
        # Predict the noise residual
        pred = self.model(
            trajectory,
            timesteps, 
            local_cond=None,
            global_cond=global_cond_combined
        )

        # now basically reshape and pass thhrough the final layer:
        pred_prepared = torch.tanh(pred.reshape(batch_dim, -1))


        f_expert = pred_prepared[:int(batch_dim/2),...]
        f_suboptimal = pred_prepared[int(batch_dim/2):,...]

        score_expert = self.reward_discriminator(f_expert)
        score_suboptimal = self.reward_discriminator(f_suboptimal)

        pure_loss = torch.mean(score_suboptimal.mode()- score_expert.mode()).to(global_cond_combined.dtype) # this should be a scalar,...
        
        loss = pure_loss

        if (self.gradient_penalty):
            # eventually add gradient penalty:
            gp = gradient_penalty(
                    f_suboptimal, f_expert, self.reward_discriminator, device=self.device
                )  # Scalar

            loss += gp.to(global_cond_combined.dtype) * self.gradient_penalty_weight  # add gradient penalty with weighting!

        

        return loss

    def forward(self, batch0, batch1):
        return self.compute_loss(batch0, batch1)
    

    def is_first_conv1d_block_in_down_modules(self, module_name):
        return module_name.startswith('down_modules.0')

    def get_max_rank_of_noise_pred_net(self):
        return self.get_max_rank_of_network(self.noise_pred_net, layers_to_exclude=self.layers_to_exclude_in_noise_pred_net)

    def get_max_rank_of_network(self, network, layers_to_exclude=[]):
        for name, module in network.named_children():
            # check if it is excluded
            module_full_name = f"{network.__class__.__name__}.{name}"
            #print("layers_to_exclude:", layers_to_exclude)
            if module_full_name in layers_to_exclude:
                #print("module_full_name:", module_full_name)
                continue
            if isinstance(module, nn.Conv1d):
                curr_max_rank = min(module.in_channels, module.out_channels)
                # print("curr_max_rank:", curr_max_rank)
                if curr_max_rank > self.max_rank:
                    self.max_rank = curr_max_rank
            else:
                self.get_max_rank_of_network(module, layers_to_exclude=layers_to_exclude)

    def get_layers_to_exclude_in_noise_pred_net(self):
        layers_to_exclude = []
        network = self.noise_pred_net
        for name, module in network.named_modules():
            if isinstance(module, ConditionalResidualBlock1D):
                # Exclude residual_conv and cond_encoder in ConditionalResidualBlock1D
                layers_to_exclude.append(f"{name}.residual_conv")
                layers_to_exclude.append(f"{name}.cond_encoder")
            elif isinstance(module, (Downsample1d, Upsample1d)):
                # Exclude Downsample1d and Upsample1d layers
                layers_to_exclude.append(name)
            elif isinstance(module, Conv1dBlock):
                # Optionally exclude the first Conv1dBlock in down_modules
                if self.is_first_conv1d_block_in_down_modules(name):
                    layers_to_exclude.append(name)
            elif name == 'final_conv':
                # Exclude final_conv
                layers_to_exclude.append(name)
        return layers_to_exclude