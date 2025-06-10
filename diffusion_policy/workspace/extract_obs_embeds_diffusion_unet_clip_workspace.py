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
import pickle
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_clip_policy import DiffusionUnetTimmPolicy
from diffusion_policy.dataset.clip_dataset import InMemoryVideoDataset
from diffusion_policy.model.vision.clip_obs_encoder import FrozenOpenCLIPImageEmbedder
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from accelerate import Accelerator
import dill
OmegaConf.register_new_resolver("eval", eval, replace=True)

import pdb

class ExtractObsDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        lastest_ckpt_path = cfg.training.FD_base_policy_ckpt_path
        payload = torch.load(open(lastest_ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
        self.payload_cfg = payload['cfg']
        # self.payload_cfg.task_description = 'closedrawer'

        # self.payload_cfg.task.dataset.mode = cfg.dataset_mode

        cls = hydra.utils.get_class(self.payload_cfg._target_)
        workspace = cls(self.payload_cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        policy = workspace.ema_model
        policy.num_inference_steps = self.cfg.training.eval_num_inference_steps

        self.device = torch.device('cuda')
        policy.eval().to(self.device)
        self.policy = policy
        
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        self.dataset = dataset
        self.train_dataloader = DataLoader(dataset, **cfg.dataloader)

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        
        full_x = []; full_y = []
        with torch.no_grad():
            for i, batch in enumerate(self.train_dataloader):
                print("Processing batch:", i)
                # device transfer
                batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                
                mod = self.policy # See "policy" folder


                # Get observations and actions, these are already normalized
                nobs = batch['obs']
                nactions = batch['action']
                batch_size = next(iter(nobs.values())).shape[0]

                # condition through global feature
                nobs_features = mod.obs_encoder(nobs)
                global_cond = nobs_features.reshape(batch_size, -1)

                # Take actions
                trajectory = nactions

                # flatten actions
                trajectory = nactions.reshape(batch_size, -1)
                # pdb.set_trace()


                print(f'At batch {i}/{len(self.train_dataloader)}')
                print(f'X: {global_cond.shape}, Y: {trajectory.shape}')
                full_x.append(global_cond.cpu()); full_y.append(trajectory.cpu())
        full_x = torch.cat(full_x, dim=0)
        full_y = torch.cat(full_y, dim=0)
        print(f'Full X: {full_x.shape}, Full Y: {full_y.shape}')
        torch.save({'X': full_x, 'Y': full_y}, cfg.training.FD_save_obs_embeds_path)

        # print where X and Y are saved
        print(f'X and Y saved to {cfg.training.FD_save_obs_embeds_path}')



@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
