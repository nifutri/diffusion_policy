import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env.robocasa.robocasa_image_wrapper import RobocasaImageWrapper
import robomimic.utils.file_utils as FileUtils
import robocasa.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robocasa.scripts.playback_dataset import get_env_metadata_from_dataset, get_env_from_dataset
from diffusion_policy.common.robocasa_util import create_environment, create_eval_environment
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
# from robosuite.renderers import OpenCVRenderer
import pdb
import robosuite
from robosuite.devices import SpaceMouse
import copy
# import cv2
# import robocasa.utils.eval_utils as EvalUtils

def create_env(dataset_path, env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = create_environment(dataset_path=dataset_path)

    return env

def create_eval_env(dataset_path, env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = create_eval_environment(dataset_path=dataset_path)

    return env
    
# def create_robocasa_eval_env(dataset_path, env_meta, shape_meta, enable_render=True):

#     env_name = 'CloseDrawer'
#     env = create_eval_env(env_name)
#     return env


class EvalRobocasaImageRunner(BaseImageRunner):
    """
    Robocasa envs also enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            shape_meta:dict,
            n_eval_rollouts=1,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_key='agentview_image',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            teleop_device='spacemouse',
            pos_sensitivity=1.0,
            rot_sensitivity=1.0,
            vendor_id=None,
            product_id=None,
        ):
        super().__init__(output_dir)

        self.n_eval_rollouts = n_eval_rollouts


        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = get_env_metadata_from_dataset(dataset_path)
        self.dataset_path = dataset_path
        # disable object state observation
        # env_meta['env_kwargs']['use_object_obs'] = False

        rotation_transformer = None
        # pdb.set_trace()
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        # self.robomimic_env
        self.env_meta = env_meta
        self.shape_meta = shape_meta
        self.robomimic_env = create_env(dataset_path=self.dataset_path, env_meta=self.env_meta, shape_meta=self.shape_meta)
        
        # create folder for dagger data if it does not exist
        rollout_dir = pathlib.Path(output_dir) / 'rollouts'
        if not os.path.exists(rollout_dir):
            os.makedirs(rollout_dir)
        self.rollout_dir = rollout_dir


        
        self.render_obs_key = render_obs_key

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.steps_per_render = steps_per_render
        self.output_dir = output_dir

    def run(self, policy: BaseImagePolicy, rollout_idx):
        device = policy.device
        dtype = policy.dtype

        rollout_idx_to_successes = {}
        num_successes = 0

        # for rollout_idx in range(self.n_eval_rollouts):
        print("rollout_idx", rollout_idx)
        # robomimic_env = create_eval_env(dataset_path=self.dataset_path, env_meta=self.env_meta, shape_meta=self.shape_meta)
        filepath = self.rollout_dir / f"rollout_{rollout_idx}.mp4"
        # create filepath 
        filename = pathlib.Path(self.output_dir).joinpath(
            'rollouts', str(rollout_idx) + ".mp4")
        filename.parent.mkdir(parents=False, exist_ok=True)
        filename = str(filename)


    

        env = MultiStepWrapper(
                VideoRecordingWrapper(
                    RobocasaImageWrapper(
                        env=self.robomimic_env,
                        shape_meta=self.shape_meta,
                        init_state=None,
                        render_obs_key=self.render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=self.fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=self.crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=filename,
                    steps_per_render=self.steps_per_render
                ),
            n_obs_steps=self.n_obs_steps,
            n_action_steps=self.n_action_steps,
            max_episode_steps=self.max_steps
        )

        # env = self.env
        
        # start rollout
        obs = env.reset()
        past_action = None
        policy.reset()
        self.timestep = 0

        env_name = self.env_meta['env_name']
        pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Timestep {self.timestep}/{self.max_steps}", 
            leave=False, mininterval=self.tqdm_interval_sec)
        
        
        done = False
        while not done:
            # create obs dict
            np_obs_dict = dict(obs)
            if self.past_action and (past_action is not None):
                # TODO: not tested
                np_obs_dict['past_action'] = past_action[
                    :,-(self.n_obs_steps-1):].astype(np.float32)
            
            # device transfer
            obs_dict = dict_apply(np_obs_dict, 
                lambda x: torch.from_numpy(x).to(
                    device=device))
            
            
            # run policy
            with torch.no_grad():
                # pdb.set_trace() # shape 2,3
                # images need to be [1, 2, 3, 128, 128]
                for key in obs_dict.keys():
                    obs_dict[key] = obs_dict[key].unsqueeze(0)

                # image_keys = ['robot0_agentview_right_image','robot0_eye_in_hand_image']
                for img_key in obs_dict:
                    if 'image' in img_key:
                    # images are currently torch.Size([1, 2, 128, 128, 3]) but need to be [1, 2, 3, 128, 128]
                        obs_dict[img_key] = obs_dict[img_key].permute(0, 1, 4, 2, 3)

                # pdb.set_trace()
                # print("running prediction")
                action_dict = policy.predict_action(obs_dict)
                # print("done predicting")

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                action_horz = action.shape[1]
                batch_size = action.shape[0]
                added_dims = np.tile([0,0,0,0,-1], (batch_size, action_horz, 1))
                action = np.concatenate([action, added_dims], axis=2)
                
                # step env
                env_action = action[0] # since we run only env at a time, need to take the first
                # env_action = env_action[:,:,:12]
                # print("action", action.shape, action.dtype)
                # print("action val", action)
                # pdb.set_trace()
                if self.abs_action:
                    env_action = self.undo_transform_action(action)


                obs, reward, done, info = env.step(env_action, render=False)
        
            
            # print("done rendering")
            done = np.all(done)
            past_action = action
            self.timestep += 1

            # update pbar
            pbar.update(action.shape[1])
        pbar.close()

        env.render()
        rollout_idx_to_successes[rollout_idx] = info
        print("rollout_idx_to_successes", rollout_idx_to_successes)
        if info['is_success'][0] ==  True:
            num_successes += 1
        print("num_successes", num_successes)

        # clear out video buffer
        # pdb.set_trace()
        env.close()

        

        return rollout_idx_to_successes, num_successes

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = 6
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        remaining_actions = action[...,3+d_rot:]

        # pdb.set_trace()
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, remaining_actions
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
