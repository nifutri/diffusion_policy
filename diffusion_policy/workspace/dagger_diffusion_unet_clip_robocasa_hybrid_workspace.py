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
import copy
import random
import pickle
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.clip_dataset import InMemoryVideoDataset
import dill
import json
from filelock import FileLock
import imageio
import open_clip
from collections import deque
from termcolor import colored
import robosuite
from robocasa.utils.dataset_registry import get_ds_path
# from robocasa.utils.eval_utils import create_eval_env_modified
# import robocasa.utils.eval_utils
from hydra.core.hydra_config import HydraConfig
from diffusion_policy.env_runner.robocasa_dagger_dp_clip_eval_image_runner import DAggerRobocasaImageRunner
from diffusion_policy.failure_detection.UQ_baselines.CFM.net_CFM import get_unet
import diffusion_policy.failure_detection.UQ_baselines.data_loader as data_loader
import pickle
## Get metrics
def adjust_xshape(x, in_dim):
    total_dim = x.shape[1]
    # Calculate the padding needed to make total_dim a multiple of in_dim
    remain_dim = total_dim % in_dim
    if remain_dim > 0:
        pad = in_dim - remain_dim
        total_dim += pad
        x = torch.cat([x, torch.zeros(x.shape[0], pad, device=x.device)], dim=1)
    # Calculate the padding needed to make (total_dim // in_dim) a multiple of 4
    reshaped_dim = total_dim // in_dim
    if reshaped_dim % 4 != 0:
        extra_pad = (4 - (reshaped_dim % 4)) * in_dim
        x = torch.cat([x, torch.zeros(x.shape[0], extra_pad, device=x.device)], dim=1)
    return x.reshape(x.shape[0], -1, in_dim)

def logpZO_UQ(baseline_model, observation, action_pred = None, task_name = 'square'):
    observation = observation
    in_dim = 7
    observation = adjust_xshape(observation, in_dim)
    if action_pred is not None:
        action_pred = action_pred
        observation = torch.cat([observation, action_pred], dim=1)
    with torch.no_grad():
        timesteps = torch.zeros(observation.shape[0], device=observation.device)
        pred_v = baseline_model(observation, timesteps)
        observation = observation + pred_v
        logpZO = observation.reshape(len(observation), -1).pow(2).sum(dim=-1)
    return logpZO


OmegaConf.register_new_resolver("eval", eval, replace=True)

import pdb

class DAggerFDDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        payload = torch.load(open(cfg.ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
        self.payload_cfg = payload['cfg']

        self.payload_cfg.task.dataset.mode = cfg.dataset_mode

        self.payload_cfg.task.dataset.tasks = {
            # 'CoffeePressButton': None,
            # 'CoffeeServeMug': None,
            # 'CoffeeSetupMug': None,
            # 'CloseDoubleDoor': None,
            # 'CloseSingleDoor': None,
            # 'OpenDoubleDoor': None,
            # 'OpenSingleDoor': None,
            'CloseDrawer': None,
            # 'OpenDrawer': None,
            # 'TurnOffMicrowave': None,
            # 'TurnOnMicrowave': None,
            # 'PnPCabToCounter': None,
            # 'PnPCounterToCab': None,
            # 'PnPCounterToMicrowave': None,
            # 'PnPCounterToSink': None,
            # 'PnPCounterToStove': None,
            # 'PnPMicrowaveToCounter': None,
            # 'PnPSinkToCounter': None,
            # 'PnPStoveToCounter': None,
            # 'TurnOffSinkFaucet': None,
            # 'TurnOnSinkFaucet': None,
            # 'TurnSinkSpout': None,
            # 'TurnOffStove': None,
            # 'TurnOnStove': None,
        }

        for key in self.payload_cfg.task.dataset.tasks:
            self.payload_cfg.task.dataset.tasks[key] = {
                "environment_file": cfg.environment_file,
                "num_experiments": cfg.num_experiments
            }

        cls = hydra.utils.get_class(self.payload_cfg._target_)
        workspace = cls(self.payload_cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        policy = workspace.ema_model
        policy.num_inference_steps = self.cfg.num_inference_steps

        self.device = torch.device('cuda')
        policy.eval().to(self.device)
        self.policy = policy
        
        dataset: InMemoryVideoDataset
        dataset = hydra.utils.instantiate(self.payload_cfg.task.dataset)
        self.dataset = dataset

        self.run_dir = HydraConfig.get().run.dir

        ## Get logpZO
        input_dim = 7
        net = get_unet(input_dim)
        # pdb.set_trace()
        ckpt = torch.load(cfg.fail_detect.save_score_network_path)
        net.load_state_dict(ckpt['model'])
        net.eval()

        # move net to device
        net.to(self.device)
        self.score_network = net

        env_runner = DAggerRobocasaImageRunner(self.run_dir,
            dataset_path = cfg.task.env_runner.dataset_path,
            shape_meta = cfg.task.env_runner.shape_meta,
            max_steps=400,
            n_obs_steps=1,
            n_action_steps=16,
            render_obs_key='robot0_agentview_right_image',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=1.0,
            device=self.device,
            n_dagger_rollouts=2,
            teleop_device= 'spacemouse',
            pos_sensitivity= 4,
            rot_sensitivity= 4,
            vendor_id= 9583,
            product_id= 50734)
        # pdb.set_trace()
        self.env_runner = env_runner

    def run(self):
        # self.env_runner.collect_dagger_rollout_data(None, None)

        # load cp band from pickle
        with open('CP_band.pkl', 'rb') as f:
            cp_band = pickle.load(f)

        N_dagger_eps = 1

        # run dagger rollout
        for dagger_ep in range(N_dagger_eps):

            dagger_hdf5_path = self.env_runner.run_interactive_dagger_rollout(self.dataset, self.policy, self.score_network, cp_band, dagger_ep, N_dagger_eps-1)
            print(f"DAgger rollout {dagger_ep} completed, data saved to {dagger_hdf5_path}")




@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = DAggerFDDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
