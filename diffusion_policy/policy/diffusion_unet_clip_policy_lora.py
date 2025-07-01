from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.policy.diffusion_unet_clip_policy import DiffusionUnetTimmPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

import pdb

from diffusion_policy.model.common.lora import LoRAForConv1d, LoRAForConv2d


class DiffusionUnetTimmPolicyPolicyWithLoRA(DiffusionUnetTimmPolicy):
    def __init__(self, shape_meta: dict,
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
                lora_scale=0.1,
                # parameters passed to step
                **kwargs

                 ):
        super().__init__(shape_meta,
                noise_scheduler,
                obs_encoder,
                num_inference_steps,
                obs_as_global_cond,
                diffusion_step_embed_dim,
                down_dims,
                kernel_size,
                n_groups,
                cond_predict_scale,
                input_pertub,
                inpaint_fixed_action_prefix,
                train_diffusion_n_samples,
                # parameters passed to step
                **kwargs)
 
        self.lora_dropout_p = 0.1
        self.lora_scale = lora_scale
        print("LORA SCALE:", self.lora_scale)

        self.lora_injected = False

        self.main_params = None
        self.lora_params = None

        # self.noise_pred_net = self.model
        # self.vision_encoder = self.obs_encoder

        self.layers_to_exclude_in_noise_pred_net = []

        print("total number of parameters before injecting LoRA:", sum(p.numel() for p in self.parameters()))
        # pdb.set_trace()

    @classmethod
    def from_policy(cls, cfg, pretrained_policy):
        return cls(cfg.shape_meta,
                   noise_scheduler=pretrained_policy.noise_scheduler,
                   obs_encoder=pretrained_policy.obs_encoder,
                   num_inference_steps=pretrained_policy.num_inference_steps,
                   obs_as_global_cond=pretrained_policy.obs_as_global_cond,
                   diffusion_step_embed_dim=cfg.policy.diffusion_step_embed_dim,
                   down_dims=cfg.policy.down_dims,
                   kernel_size=cfg.policy.kernel_size,
                   n_groups=cfg.policy.n_groups,
                   cond_predict_scale=cfg.policy.cond_predict_scale,
                   input_pertub=cfg.policy.input_pertub,
                   train_diffusion_n_samples=cfg.policy.train_diffusion_n_samples,
                    lora_scale=cfg.finetuning.lora_scale,
                   # parameters passed to step 
                   **pretrained_policy.kwargs
                   )

    def freeze_main_network(self, apply_lora_to_visual_encoder):
        for param in self.model.parameters():
            param.requires_grad = False

        if apply_lora_to_visual_encoder:
            for param in self.obs_encoder.parameters():
                param.requires_grad = False

        if apply_lora_to_visual_encoder:
            for name, param in self.obs_encoder.named_parameters():
                if 'lora_down' in name or 'lora_up' in name:
                    # Keep LoRA parameters trainable
                    param.requires_grad = True
                elif isinstance(param, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for name, param in self.model.named_parameters():
            if 'lora_down' in name or 'lora_up' in name:
                param.requires_grad = True
            elif isinstance(param, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                param.requires_grad = True
            else:
                param.requires_grad = False

        lora_params_list = list(filter(lambda p: p.requires_grad, self.model.parameters())) + list(filter(lambda p: p.requires_grad, self.obs_encoder.parameters()))

        total_num_learnable_params = sum(p.numel() for p in lora_params_list)
        print("Freeze Non-LoRA layers. Now, the number of trainable parameters is:", total_num_learnable_params)

    def get_main_and_lora_params(self):
        return self.main_params, self.lora_params
    
    def set_model(self, model):
        """
        Set the model to be used in the policy.
        This is useful for loading a pretrained model.
        """
        self.model = model
        self.lora_injected = False

    def inject_lora(self, apply_lora_to_visual_encoder, lora_rank=256):
        self.model = apply_lora_to_network(self.model, lora_rank, self.lora_dropout_p,
                                                    self.lora_scale, self.layers_to_exclude_in_noise_pred_net)

        if apply_lora_to_visual_encoder:
            self.obs_encoder = apply_lora_to_network(self.obs_encoder, lora_rank, self.lora_dropout_p,
                                                        self.lora_scale, [])

        lora_params = []
        main_params = []

        for name, param in self.model.named_parameters():
            if 'lora_down' in name or 'lora_up' in name:
                lora_params.append(param)
            else:
                main_params.append(param)

        if apply_lora_to_visual_encoder:
            for name, param in self.obs_encoder.named_parameters():
                if 'lora_down' in name or 'lora_up' in name:
                    lora_params.append(param)
                else:
                    main_params.append(param)
        else:
            main_params.extend(self.obs_encoder.parameters())

        self.lora_injected = True

        total_num_lora_params = sum(p.numel() for p in lora_params)


        self.main_params = main_params
        self.lora_params = lora_params

    def merge_lora_weights(self, apply_lora_to_visual_encoder):
        self.model = merge_lora_weights(self.model)

        if apply_lora_to_visual_encoder:
            self.obs_encoder = merge_lora_weights(self.obs_encoder)

        self.lora_injected = False
        #print("LoRA weights have been merged into the main network weights.")



    def reduce_rank(self, new_rank, apply_lora_to_visual_encoder=False):
        self.reduce_lora_rank_in_network(self.model, new_rank)

        if apply_lora_to_visual_encoder:
            self.reduce_lora_rank_in_network(self.obs_encoder, new_rank)

        self.lora_params = []
        for name, param in self.model.named_parameters():
            if 'lora_down' in name or 'lora_up' in name:
                self.lora_params.append(param)

        if apply_lora_to_visual_encoder:
            for name, param in self.obs_encoder.named_parameters():
                if 'lora_down' in name or 'lora_up' in name:
                    self.lora_params.append(param)

    def reduce_lora_rank_in_network(self, network, new_rank):
        for module in network.modules():
            if isinstance(module, LoRAForConv1d) or isinstance(module, LoRAForConv2d):
                module.reduce_rank(new_rank)






def apply_lora_to_network(network, lora_rank=8, lora_dropout_p=0.1, lora_scale=1.0, layers_to_exclude=[]):
    for name, module in network.named_children():
        module_full_name = f"{network.__class__.__name__}.{name}"
        if module_full_name in layers_to_exclude:
            continue
        if isinstance(module, nn.Conv1d):
            device = module.weight.device
            dtype = module.weight.dtype

            lora_conv = LoRAForConv1d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=(module.bias is not None),
                r=lora_rank,
                dropout_p=lora_dropout_p,
                scale=lora_scale,
                device=device,
                dtype=dtype
            )
            # pdb.set_trace()
            lora_conv.conv.weight.data = module.weight.data.clone().to(device=device, dtype=dtype)
            if module.bias is not None:
                lora_conv.conv.bias.data = module.bias.data.clone().to(device=device, dtype=dtype)
            setattr(network, name, lora_conv)
        elif isinstance(module, nn.Conv2d):
            device = module.weight.device
            dtype = module.weight.dtype

            lora_conv = LoRAForConv2d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=(module.bias is not None),
                r=lora_rank,
                dropout_p=lora_dropout_p,
                scale=lora_scale,
                device=device,
                dtype=dtype
            )
            lora_conv.conv.weight.data = module.weight.data.clone().to(device=device, dtype=dtype)
            if module.bias is not None:
                lora_conv.conv.bias.data = module.bias.data.clone().to(device=device, dtype=dtype)
            setattr(network, name, lora_conv)
        else:
            apply_lora_to_network(module, lora_rank, lora_dropout_p, lora_scale)
    return network


def merge_lora_weights(network):
    for name, module in network.named_children():
        if isinstance(module, LoRAForConv1d) or isinstance(module, LoRAForConv2d):
            module.merge_lora_weights()
            setattr(network, name, module.conv)
        else:
            merge_lora_weights(module)
    return network

