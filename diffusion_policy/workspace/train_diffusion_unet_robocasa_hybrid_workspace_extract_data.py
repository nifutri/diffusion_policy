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

class DataExtractionDiffusionUnetRobocasaHybridWorkspace(BaseWorkspace):
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

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        # 
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        device = torch.device(cfg.training.device)
        self.model.set_normalizer(normalizer)
        self.ema_model.set_normalizer(normalizer)
        self.model.to(device)
        self.ema_model.to(device)

        # configure logging
        full_x = []; full_y = []
        with torch.no_grad():
            for i, batch in enumerate(train_dataloader):
                print("Processing batch:", i)
                # device transfer
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                
                mod = self.ema_model # See "policy" folder
                nobs = mod.normalizer.normalize(batch['obs'])
                nactions = mod.normalizer['action'].normalize(batch['action'])
                batch_size = nactions.shape[0]

                # handle different ways of passing observation
                trajectory = nactions.reshape(batch_size, -1)
                # Get latent representation of observations
                this_nobs = dict_apply(nobs, 
                    lambda x: x[:,:mod.n_obs_steps,...].reshape(-1,*x.shape[2:]))
                nobs_features = mod.obs_encoder(this_nobs)
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)

                print(f'At batch {i}/{len(train_dataloader)}')
                print(f'X: {global_cond.shape}, Y: {trajectory.shape}')
                full_x.append(global_cond.cpu()); full_y.append(trajectory.cpu())
        full_x = torch.cat(full_x, dim=0)
        full_y = torch.cat(full_y, dim=0)
        print(f'Full X: {full_x.shape}, Full Y: {full_y.shape}')
        torch.save({'X': full_x, 'Y': full_y}, self.logging)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetRobocasaHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
