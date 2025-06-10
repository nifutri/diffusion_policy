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
import open_clip
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

class EvalRobocasaImageRunner(BaseImageRunner):
    """
    Robocasa envs also enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            shape_meta:dict,
            n_eval_rollouts=1,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_key='agentview_image',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            device='cuda',
        ):
        super().__init__(output_dir)

        self.n_eval_rollouts = n_eval_rollouts
        self.device = device

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        # steps_per_render = max(robosuite_fps // fps, 1)
        steps_per_render = 1

        # read from dataset
        # pdb.set_trace()
        dataset_path = 'datasets/v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30/demo_gentex_im128_randcams_im256.hdf5'
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
        # pdb.set_trace()
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

    def convert_observations(self, dataset, obs, clip_embedding):
        # pdb.set_trace()
        # left_image = np.stack([dataset.convert_frame(frame=frame, swap_rgb=dataset.swap_rgb) for frame in left_image_queue])
        # right_image = np.stack([dataset.convert_frame(frame=frame, swap_rgb=dataset.swap_rgb) for frame in right_image_queue])
        # gripper_image = np.stack([dataset.convert_frame(frame=frame, swap_rgb=dataset.swap_rgb) for frame in gripper_image_queue])

        # print(left_image.shape)
        # dataset.undo_operations_and_save(left_image, 'left_image.png')
        # dataset.undo_operations_and_save(right_image, 'right_image.png')
        # dataset.undo_operations_and_save(gripper_image, 'gripper_image.png')
        # exit()
        # (249, 249, 3)

        # (1, 3, 249, 249) (1, 3, 249, 249) (1, 3, 249, 249)


        left_image = obs['robot0_agentview_left_image']
        right_image = obs['robot0_agentview_right_image']
        gripper_image = obs['robot0_eye_in_hand_image']
        # pdb.set_trace()

        # flip right and left image over, currently they are upside down
        left_image[0] = np.flipud(left_image[0])
        right_image[0] = np.flipud(right_image[0])
        gripper_image[0] = np.flipud(gripper_image[0])

        

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        # axs[0].imshow(left_image[-1])
        # axs[0].set_title('Left Image')
        # axs[1].imshow(right_image[-1])
        # axs[1].set_title('Right Image')
        # axs[2].imshow(gripper_image[-1])
        # axs[2].set_title('Gripper Image')
        # plt.show()

        left_image = dataset.convert_frame(frame=left_image[0], swap_rgb=dataset.swap_rgb)
        right_image = dataset.convert_frame(frame=right_image[0], swap_rgb=dataset.swap_rgb)
        gripper_image = dataset.convert_frame(frame=gripper_image[0], swap_rgb=dataset.swap_rgb)
        # pdb.set_trace()

        left_image = torch.tensor(left_image, dtype=torch.float32)
        right_image = torch.tensor(right_image, dtype=torch.float32)
        gripper_image = torch.tensor(gripper_image, dtype=torch.float32)

        left_image = (left_image + 1) / 2
        right_image = (right_image + 1) / 2
        gripper_image = (gripper_image + 1) / 2

        left_image = dataset.augmentation_transform(left_image, dataset.transform_rgb)
        right_image = dataset.augmentation_transform(right_image, dataset.transform_rgb)
        gripper_image = dataset.augmentation_transform(gripper_image, dataset.transform_rgb)
        
        left_image = left_image.unsqueeze(0)
        right_image = right_image.unsqueeze(0)
        gripper_image = gripper_image.unsqueeze(0)

        # unsqueeze again
        left_image = left_image.unsqueeze(0)
        right_image = right_image.unsqueeze(0)
        gripper_image = gripper_image.unsqueeze(0)

        return {
            "task_description": clip_embedding,
            "left_image": left_image,
            "right_image": right_image,
            "gripper_image": gripper_image,
            }


    def compute_dp_clip_rollout_scores(self, video_dataset, policy: BaseImagePolicy, score_network, rollout_idx):
        self.video_dataset = video_dataset
        device = policy.device
        dtype = policy.dtype

        self.score_network = score_network
        self.policy = policy

        rollout_idx_to_successes = {}
        num_successes = 0

        # for rollout_idx in range(self.n_eval_rollouts):
        print("rollout_idx", rollout_idx)
        self.robomimic_env = create_env(dataset_path=self.dataset_path, env_meta=self.env_meta, shape_meta=self.shape_meta)
        # robomimic_env = create_eval_env(dataset_path=self.dataset_path, env_meta=self.env_meta, shape_meta=self.shape_meta)
        # create filepath 
        # pdb.set_trace()
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
        logpZO_local_slices = []
        dones = []
        rewards = []
        observations = []
        actions = []
        infos = []
        X_encodings = []
        Y_encodings = []
        when_success = []

        # pdb.set_trace() 
        task_description = env.env.env.env.get_ep_meta()["lang"]
        task_description = open_clip.tokenize([task_description]) # returns torch.Size([1, 77])
        with torch.no_grad():
            clip_embedding = self.video_dataset.lang_model(task_description.to(self.device)).cpu().unsqueeze(0) # returns torch.Size([1, 1, 1024])

        
        

        # print batch shape
        # for key in batch:
        #     print(key, batch[key].shape)

        # task_description torch.Size([1, 1, 1024])
        # left_image torch.Size([1, 1, 256, 224, 224])
        # right_image torch.Size([1, 1, 256, 224, 224])
        # gripper_image torch.Size([1, 1, 256, 224, 224])

        # task_description torch.Size([1, 1, 1024])
        # left_image torch.Size([1, 1, 3, 224, 224])
        # right_image torch.Size([1, 1, 3, 224, 224])
        # gripper_image torch.Size([1, 1, 3, 224, 224])


        # action_pred, action_pred_infos_result = self.policy.predict_action_with_infos(batch)
        # action_pred = ((action_pred.detach().cpu().numpy() + 1) / 2) * (self.video_dataset.max - self.video_dataset.min) + self.video_dataset.min
        # action_pred = np.squeeze(action_pred)
        # action_pred = np.hstack((action_pred, [[0, 0, 0, 0, -1]] * action_pred.shape[0]))
        # action_horizon = 16
        # action_pred = action_pred[0:action_horizon]
        # pdb.set_trace()

        while not done:
            batch = self.convert_observations(self.video_dataset, obs, clip_embedding)
            batch = {key: value.to(self.device, dtype=torch.float32) for key, value in batch.items()}
            
            # import matplotlib.pyplot as plt
            # # plot all images in batch in one plot
            # fig, axs = plt.subplots(1, 3, figsize=(15, 5))


            # for img_name in ['left_image', 'right_image', 'gripper_image']:
            #     img = batch[img_name][0, -1].permute(1, 2, 0).cpu().numpy()
            #     if img_name == 'left_image':
            #         axs[0].imshow(img)
            #         axs[0].set_title('Left Image')
            #     elif img_name == 'right_image':
            #         axs[1].imshow(img)
            #         axs[1].set_title('Right Image')
            #     elif img_name == 'gripper_image':
            #         axs[2].imshow(img)
            #         axs[2].set_title('Gripper Image')
            # plt.show()
            
            action_pred, action_pred_infos_result = self.policy.predict_action_with_infos(batch)
            action_pred = ((action_pred.detach().cpu().numpy() + 1) / 2) * (self.video_dataset.max - self.video_dataset.min) + self.video_dataset.min
            action_pred = np.squeeze(action_pred)
            action_pred = np.hstack((action_pred, [[0, 0, 0, 0, -1]] * action_pred.shape[0]))
            action_horizon = 16
            action_pred = action_pred[0:action_horizon]

            baseline_metric = logpZO_UQ(self.score_network, action_pred_infos_result['global_cond'])


            obs, reward, done, info = env.step(action_pred, render=True)
            dones.append(done)
            rewards.append(reward)
            observations.append(obs)
            actions.append(action_pred)
            infos.append(info)
            logpZO_local_slices.append(baseline_metric)

            is_success = env.env.env.env._check_success()
            when_success.append(is_success)
            if is_success:
                break

            
            # update pbar
            pbar.update(self.n_action_steps)
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

        aggregated_data = {
            'rollout_idx': rollout_idx,
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'infos': infos,
            'logpZO_local_slices': logpZO_local_slices,
            'X_encodings': X_encodings,
            'Y_encodings': Y_encodings,
            'when_success': when_success,
        }

        

        return rollout_idx_to_successes, num_successes, logpZO_local_slices, aggregated_data
    

    