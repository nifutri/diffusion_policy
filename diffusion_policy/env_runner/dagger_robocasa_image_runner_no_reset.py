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
from robocasa.scripts.collect_demos import gather_dagger_demonstrations_as_hdf5
from robocasa.utils.robomimic.robomimic_dataset_utils import convert_to_robomimic_format
from diffusion_policy.common.robocasa_util import create_environment_interactive
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper, DAggerDataCollectionWrapper
# from robosuite.renderers import OpenCVRenderer
import pdb
import robosuite
from robosuite.devices import SpaceMouse
import copy
import json
# import cv2

def display_controller_cv2(controller_text):
    print(f"Current controller: {controller_text}")

def create_env(dataset_path, env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env, env_meta = create_environment_interactive(dataset_path=dataset_path)

    # env_type = 1
    # env_kwargs = env_meta["env_kwargs"]
    # env_kwargs["env_name"] = env_meta["env_name"]
    # env_kwargs["has_renderer"] = False
    # env_kwargs["renderer"] = "mjviewer"
    # env_kwargs["has_offscreen_renderer"] = False
    # env_kwargs["use_camera_obs"] = False
    # # pdb.set_trace()
    # env = robosuite.make(**env_kwargs)

    # import pdb; pdb.set_trace()

    # 


    # env = EnvUtils.create_env(env_name=env_meta['env_name'], **env_kwargs,)

    # pdb.set_trace()
    # env = EnvUtils.create_env_from_metadata(env_meta=env_meta,render=False, render_offscreen=enable_render,use_image_obs=enable_render, )
    return env, env_meta

def is_empty_input_spacemouse(action_dict):
    if not np.all(action_dict["right_delta"] == 0):
        return False
    if "base_mode" in action_dict and action_dict["base_mode"] != -1:
        return False
    if "base" in action_dict and not np.all(action_dict["base"] == 0):
        return False

    return True


class DAggerRobocasaImageRunner(BaseImageRunner):
    """
    Robocasa envs also enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            shape_meta:dict,
            n_dagger_rollouts=1,
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

        self.n_dagger_rollouts = n_dagger_rollouts
        # self.teleop_device = teleop_device
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.vendor_id = vendor_id
        self.product_id = product_id
        print(f"Using vendor_id: {self.vendor_id}, product_id: {self.product_id}")
        # pdb.set_trace()


        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = get_env_metadata_from_dataset(dataset_path)
        # disable object state observation
        # env_meta['env_kwargs']['use_object_obs'] = False

        rotation_transformer = None
        # pdb.set_trace()
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        
        robomimic_env, robomimic_env_meta = create_env(dataset_path=dataset_path, env_meta=env_meta, shape_meta=shape_meta)
        # Robosuite's hard reset causes excessive memory consumption.
        # Disabled to run more envs.
        # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
        
        # robomimic_env.hard_reset = False
        robomimic_env = VisualizationWrapper(robomimic_env)

        # Grab reference to controller config and convert it to json-encoded string
        # pdb.set_trace()
        self.env_info = json.dumps(robomimic_env_meta)

        # setup teleop
        # create folder for dagger data if it does not exist
        dagger_dir = pathlib.Path(output_dir) / 'dagger_data'
        if not os.path.exists(dagger_dir):
            os.makedirs(dagger_dir)
        self.dagger_dir = dagger_dir

        # create folder for dagger data if it does not exist
        processed_dagger_dir = pathlib.Path(output_dir) / 'processed_dagger_data'
        if not os.path.exists(processed_dagger_dir):
            os.makedirs(processed_dagger_dir)
        self.processed_dagger_dir = processed_dagger_dir
        # # get episode index by looking at the existing files in dagger_dir and incrementing it by 1
        # num_files = len(list(dagger_dir.glob('*.h5')))

        robomimic_env = DAggerDataCollectionWrapper(robomimic_env, dagger_dir)

        # setup teleop device
    
        self.teleop_device = SpaceMouse(
            env=robomimic_env,
            pos_sensitivity=self.pos_sensitivity,
            rot_sensitivity=self.rot_sensitivity,
            vendor_id=int(self.vendor_id),
            product_id=int(self.product_id),
        )
        



        env = MultiStepWrapper(
                RobocasaImageWrapper(
                    env=robomimic_env,
                    shape_meta=shape_meta,
                    init_state=None,
                    render_obs_key=render_obs_key
                ),
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps
        )
        

        


        self.env_meta = env_meta
        self.env = env

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # start rollout
        obs = env.reset()
        past_action = None
        policy.reset()
        self.timestep = 0

        env_name = self.env_meta['env_name']
        pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Timestep {self.timestep}/{self.max_steps}", 
            leave=False, mininterval=self.tqdm_interval_sec)
        
        # setup teleop
        self.teleop_device.start_control()

        nonzero_ac_seen = False
        # Keep track of prev gripper actions when using since they are position-based and must be maintained when arms switched
        # pdb.set_trace()
        all_prev_gripper_actions = [
            {
                f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                for robot_arm in robot.arms
                if robot.gripper[robot_arm].dof > 0
            }
            for robot in env.env.env.robots
        ]
        zero_action = np.zeros(env.env.env.action_dim)
        discard_dagger_traj = False
        acting_agent = 'robot'
        new_controller = "Teleop Policy (Human)" if acting_agent=='human' else "Auto Policy (Robot)"
        display_controller_cv2(new_controller)
        mirror_actions = True

        done = False
        timestep = 0
        num_human_segments = 0
        human_segment_idx = 0
        human_segment_to_length = {}
        while not done:
            timestep += 1
            # print(f"timestep {timestep}, acting agent {acting_agent}")
            if acting_agent == 'human':
                human_segment_to_length[human_segment_idx] += 1
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
            
            if acting_agent == 'human' and self.teleop_device.is_acting_agent is False:
                # pdb.set_trace()
                acting_agent = 'robot'
                env.env.env.set_dagger_acting_agent(acting_agent)
            
                
                new_controller = "Teleop Policy (Human)" if acting_agent=='human' else "Auto Policy (Robot)"
                display_controller_cv2(new_controller)
            elif acting_agent == 'robot' and self.teleop_device.is_acting_agent is True:
                # pdb.set_trace()
                num_human_segments += 1
                human_segment_idx += 1
                human_segment_to_length[human_segment_idx] = 0
                acting_agent = 'human'
                env.env.env.set_dagger_acting_agent(acting_agent)
                new_controller = "Teleop Policy (Human)" if acting_agent=='human' else "Auto Policy (Robot)"
                display_controller_cv2(new_controller)
            
            if acting_agent == 'human':
                # Set active robot
                active_robot = env.env.env.robots[self.teleop_device.active_robot]
                active_arm = self.teleop_device.active_arm

                # Get the newest action
                input_ac_dict = self.teleop_device.input2action(mirror_actions=mirror_actions)

                # If action is none, then this a reset so we should break
                if input_ac_dict is None:
                    discard_traj = True
                    break

                action_dict = copy.deepcopy(input_ac_dict)

                # set arm actions
                for arm in active_robot.arms:
                    controller_input_type = active_robot.part_controllers[arm].input_type
                    if controller_input_type == "delta":
                        action_dict[arm] = input_ac_dict[f"{arm}_delta"]
                    elif controller_input_type == "absolute":
                        action_dict[arm] = input_ac_dict[f"{arm}_abs"]
                    else:
                        raise ValueError
                    
                if is_empty_input_spacemouse(action_dict):
                    if not nonzero_ac_seen:
                        env.render()
                        continue
                else:
                    nonzero_ac_seen = True

                # Maintain gripper state for each robot but only update the active robot with action
                env_action = [
                    robot.create_action_vector(all_prev_gripper_actions[i])
                    for i, robot in enumerate(env.env.env.robots)
                ]
                env_action[self.teleop_device.active_robot] = active_robot.create_action_vector(action_dict)
                env_action = np.concatenate(env_action)
                env_action =  np.expand_dims(env_action, axis=0) # add time dimension
                # pdb.set_trace()
                obs, _, _, _ = env.step(env_action)
                env.render()
                if env.env.env._check_success():
                    print("finished task")
                    break

            else:
                # run policy
                with torch.no_grad():
                    # pdb.set_trace() # shape 2,3
                    # images need to be [1, 2, 3, 128, 128]
                    for key in obs_dict.keys():
                        obs_dict[key] = obs_dict[key].unsqueeze(0)

                    image_keys = ['robot0_agentview_right_image','robot0_eye_in_hand_image']
                    for img_key in image_keys:
                        # images are currently torch.Size([1, 2, 128, 128, 3]) but need to be [1, 2, 3, 128, 128]
                        obs_dict[img_key] = obs_dict[img_key].permute(0, 1, 4, 2, 3)

                    # pdb.set_trace()
                    print("running prediction")
                    action_dict = policy.predict_action(obs_dict)
                    print("done predicting")

                    # device_transfer
                    np_action_dict = dict_apply(action_dict,
                        lambda x: x.detach().to('cpu').numpy())

                    action = np_action_dict['action']
                    if not np.all(np.isfinite(action)):
                        print(action)
                        raise RuntimeError("Nan or Inf action")
                    
                    # step env
                    env_action = action[0] # since we run only env at a time, need to take the first
                    # env_action = env_action[:,:,:12]
                    print("action", action.shape, action.dtype)
                    print("action val", action)
                    # pdb.set_trace()
                    if self.abs_action:
                        env_action = self.undo_transform_action(action)

                    obs, reward, done, info = env.step(env_action, render=True)
            
            
            print("done rendering")
            done = np.all(done)
            past_action = action
            self.timestep += 1

            # update pbar
            pbar.update(action.shape[1])
        pbar.close()
        print("human_segment_to_length", human_segment_to_length)
        
        want_to_save = input("Done with this episode, do you want to save the demo? (y/n): ").lower()
        if want_to_save == 'y':
            discard_dagger_traj = False
        else:
            discard_dagger_traj = True

        # clear out video buffer
        # _ = env.reset()
        env.close()
        # Cleanup
        if nonzero_ac_seen and hasattr(env, "ep_directory"):
            ep_directory = env.env.env.ep_directory
        else:
            ep_directory = None

        excluded_eps = []
        if discard_dagger_traj and ep_directory is not None:
            excluded_eps.append(ep_directory.split("/")[-1])
        hdf5_path = gather_dagger_demonstrations_as_hdf5(
                self.dagger_dir, self.processed_dagger_dir, self.env_info, excluded_episodes=excluded_eps
            )
        print(f"Gathered DAgger demonstrations to {hdf5_path}")
        convert_to_robomimic_format(hdf5_path)
        print(f"Converted DAgger demonstrations to robomimic format at {hdf5_path}")
        
        

        return discard_dagger_traj

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
