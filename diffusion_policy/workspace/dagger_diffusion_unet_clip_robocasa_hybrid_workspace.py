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

TASK_NAME_TO_HUMAN_PATH = {'PnPCabToCounter': "../robocasa/datasets_first/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24/demo_gentex_im128_randcams_im256.hdf5",
                           'PnPSinkToCounter': "../robocasa/datasets_first/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2/demo_gentex_im128_randcams_im256.hdf5",
                           'OpenSingleDoor': "../robocasa/datasets_first/v0.1/single_stage/kitchen_doors/OpenSingleDoor/2024-04-24/demo_gentex_im128_randcams_im256.hdf5",
                           'OpenDrawer': "../robocasa/datasets_first/v0.1/single_stage/kitchen_drawer/OpenDrawer/2024-05-03/demo_gentex_im128_randcams_im256.hdf5",
                           'CloseDrawer': "../robocasa/datasets_first/v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30/demo_gentex_im128_randcams_im256.hdf5",
                           'TurnOnStove': "../robocasa/datasets_first/v0.1/single_stage/kitchen_stove/TurnOnStove/2024-05-02/demo_gentex_im128_randcams_im256.hdf5",
                           'TurnOnSinkFaucet': "../robocasa/datasets_first/v0.1/single_stage/kitchen_sink/TurnOnSinkFaucet/2024-04-25/demo_gentex_im128_randcams_im256.hdf5",
                           'CoffeePressButton': "../robocasa/datasets_first/v0.1/single_stage/kitchen_coffee/CoffeePressButton/2024-04-25/demo_gentex_im128_randcams_im256.hdf5",
                            'CoffeeServeMug': "../robocasa/datasets_first/v0.1/single_stage/kitchen_coffee/CoffeeServeMug/2024-05-01/demo_gentex_im128_randcams_im256.hdf5",
                           }


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
        print("loading from checkpoint", cfg.ckpt_path)
        payload = torch.load(open(cfg.ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
        self.payload_cfg = payload['cfg']

        self.payload_cfg.task.dataset.mode = cfg.dataset_mode

        # Read task name and configure human_path and tasks
        task_name = cfg.task_name
        self.task_name = task_name
        self.payload_cfg.task.dataset.tasks = {
            task_name: None,
        }
        self.payload_cfg.task.dataset.tasks = {task_name: None}
        self.payload_cfg.task.dataset.human_path = TASK_NAME_TO_HUMAN_PATH[task_name]
        cfg.task.env_runner.dataset_path = TASK_NAME_TO_HUMAN_PATH[task_name]

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

        self.cp_band_path = cfg.fail_detect.cp_band_path
        self.n_dagger_episodes = cfg.dagger.num_interactive_rollouts

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

        with open("datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24/demo_gentex_im256_randcams_100_train_envs.pkl", "rb") as pickle_file:
            environment_data = pickle.load(pickle_file)
        max_traj_len = self.cfg['max_traj_len']
        camera_names = environment_data['env_kwargs']['camera_names']   # ['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand']
        camera_height = round(self.payload_cfg.task.dataset['frame_height']/self.payload_cfg.task.dataset['aug']['crop'])
        camera_width = round(self.payload_cfg.task.dataset['frame_width']/self.payload_cfg.task.dataset['aug']['crop'])
        pred_horizon = self.payload_cfg.task['action_horizon']
        action_horizon = self.cfg['execution_horizon']
        controller_configs=environment_data['env_kwargs']['controller_configs']

        environment_data['env_kwargs']['has_renderer'] = True
        environment_data['env_kwargs']["renderer"] = "mjviewer"
        # env = create_eval_env_modified(env_name=task_name, controller_configs=environment_data['env_kwargs']['controller_configs'], id_selection=demo_number//10)
        # pdb.set_trace()

        env_runner = DAggerRobocasaImageRunner(self.run_dir,
            dataset_path = cfg.task.env_runner.dataset_path,
            shape_meta = cfg.task.env_runner.shape_meta,
            task_name=task_name,
            controller_configs=controller_configs,
            max_steps=400,
            n_obs_steps=1,
            n_action_steps=16,
            render_obs_key='robot0_agentview_left_image',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=1.0,
            device=self.device,
            n_dagger_rollouts=2,
            teleop_device= 'spacemouse',
            pos_sensitivity= 2,
            rot_sensitivity= 2,
            vendor_id= 9583,
            product_id= 50734)
        # pdb.set_trace()
        self.env_runner = env_runner

    def run(self):
        # self.env_runner.collect_dagger_rollout_data(None, None)

        # load cp band from pickle
        with open(self.cp_band_path, 'rb') as f:
            cp_band = pickle.load(f)

        N_dagger_eps = self.n_dagger_episodes

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
