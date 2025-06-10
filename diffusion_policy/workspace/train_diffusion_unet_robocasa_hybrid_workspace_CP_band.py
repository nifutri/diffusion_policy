if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_robocasa_hybrid_image_policy import DiffusionUnetRobocasaHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
import pdb

OmegaConf.register_new_resolver("eval", eval, replace=True)



from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

def get_unet(input_dim):
    return ConditionalUnet1D(
        input_dim=input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=128,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=False
    )


    

class CPDiffusionUnetRobocasaHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetRobocasaHybridImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetRobocasaHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume from checkpoint, but only eval
        # if cfg.training.resume:
        lastest_ckpt_path = cfg.dagger.checkpoint_path
        print(f"Resuming from checkpoint {lastest_ckpt_path}")
        self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        # 
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()
        # pdb.set_trace()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)


        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        # pdb.set_trace()
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)


        # load score network
        policy_type = 'diffusion'
        task_name = 'closedrawer'
        def get_steps(policy_type, task_name):
            if policy_type == 'diffusion':
                num_inference_steps = 60
                if task_name == 'transport':
                    num_inference_steps = 70
            else:
                num_inference_steps = 1
                if task_name == 'tool_hang':
                    num_inference_steps = 40
            return num_inference_steps
        num_inference_step = get_steps(policy_type, task_name)
        print(f"num_inference_step: {num_inference_step}")  
        
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()


        ## Get logpZO
        input_dim = 7
        net = get_unet(input_dim)
        # pdb.set_trace()
        ckpt = torch.load(cfg.score_network.score_path)
        net.load_state_dict(ckpt['model'])
        net.eval()

        # move net to device
        net.to(device)



        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        aggregated_data = {}
        total_successes = 0
        N_test_calib_rollouts = 30

        # run rollout
        aggregated_data = []
        for rollout_idx in tqdm.tqdm(range(N_test_calib_rollouts), desc="Running Rollouts"):

            runner_log = env_runner.compute_rollout_scores(policy, net, rollout_idx)
            print("runner_log", runner_log)
            aggregated_data.append(runner_log)

        #  dump to pickle
        # model_name = cfg.score_network.score_path.split(".ckpt")
        with open(f'closedrawer_img256_crop128_batch256_latest_FD_data_diffusion_runner_log_{policy_type}_{task_name}_numtrajs{N_test_calib_rollouts}.pkl', 'wb') as f:
            import pickle
            pickle.dump(aggregated_data, f)
        pdb.set_trace()
        # dump log to json
        # import json 
        # output_dir = os.path.join(f'../data/outputs/train_{policy_type}_unet_visual_{task_name}', 'final_eval', f'steps_{num_inference_step}')
        # os.makedirs(output_dir, exist_ok=True)
        # json_filename = f'eval_log_steps_{num_inference_step}.json'
        # json_log = dict()
        # for key, value in runner_log.items():
        #     if isinstance(value, wandb.sdk.data_types.video.Video):
        #         json_log[key] = value._path
        #     else:
        #         json_log[key] = value
        # out_path = os.path.join(output_dir, json_filename)
        # json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)



@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = EvalDiffusionUnetRobocasaHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
