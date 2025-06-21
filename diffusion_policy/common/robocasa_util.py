import numpy as np
import copy
import os
import h5py
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
import robocasa.utils.env_utils as EnvUtils
from scipy.spatial.transform import Rotation

from robocasa.scripts.playback_dataset import get_env_metadata_from_dataset, get_env_from_dataset
# from robomimic.config import config_factory
import pdb
import robosuite
# import robocasa
from robocasa.utils.env_utils import create_env, run_random_rollouts
# from robocasa.utils.eval_utils import create_eval_env
from robocasa.scripts.playback_dataset import reset_to

def create_environment(dataset_path):
    """
    Create a robosuite environment from the dataset metadata.
    """
    env_meta = get_env_metadata_from_dataset(dataset_path)
    
    # setup env arguments
    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["renderer"] = "mjviewer"
    # render_camera="robot0_agentview_left",
    # env_kwargs["render_camera"] = "robot0_agentview_left"
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = True
    env_kwargs["obj_instance_split"] = "B"
    env_kwargs["generative_textures"] = None
    env_kwargs["randomize_cameras"] = False
    env_kwargs["layout_and_style_ids"] = ((1, 1), (2, 2), (4, 4), (6, 9), (7, 10))
    env_kwargs["layout_ids"] = None
    env_kwargs["style_ids"] = None

    # create the environment
    env = robosuite.make(**env_kwargs)
    
    return env

def create_eval_environment(dataset_path):
    """
    Create a robosuite environment from the dataset metadata.
    """
    env_meta = get_env_metadata_from_dataset(dataset_path)
    
    # setup env arguments
    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = True
    
    # obj_instance_split="B",
    # generative_textures=None,
    # randomize_cameras=False,
    # layout_and_style_ids=((1, 1), (2, 2), (4, 4), (6, 9), (7, 10)),
    env_kwargs["obj_instance_split"] = "B"
    env_kwargs["generative_textures"] = None
    env_kwargs["randomize_cameras"] = False
    env_kwargs["layout_and_style_ids"] = ((1, 1), (2, 2), (4, 4), (6, 9), (7, 10))
    env_kwargs["layout_ids"] = None
    env_kwargs["style_ids"] = None

    # create the environment
    env = robosuite.make(**env_kwargs)
    
    return env

def create_environment_interactive(dataset_path):
    """
    Create a robosuite environment from the dataset metadata.
    """
    env_meta = get_env_metadata_from_dataset(dataset_path)
    
    # setup env arguments
    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = True
    env_kwargs['render_camera'] = 'robot0_agentview_right'
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = True
    env_kwargs["obj_instance_split"] = "B"
    env_kwargs["generative_textures"] = None
    env_kwargs["randomize_cameras"] = False
    env_kwargs["camera_names"] = ["robot0_agentview_left_image", "robot0_agentview_right_image", "robot0_eye_in_hand_image"]
    # env_kwargs["layout_and_style_ids"] = ((1, 1), (2, 2), (4, 4), (6, 9), (7, 10))
    # env_kwargs["layout_ids"] = None
    # env_kwargs["style_ids"] = None

    # create the environment
    pdb.set_trace()
    env = robosuite.make(**env_kwargs)
    
    return env, env_meta


def create_interactive_eval_env_modified(
    env_name,
    # robosuite-related configs
    robots="PandaMobile",
    controllers="OSC_POSE",
    camera_names=[
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ],
    camera_widths=256,
    camera_heights=256,
    seed=None,
    # robocasa-related configs
    obj_instance_split="B",
    generative_textures=None,
    randomize_cameras=False,
    layout_and_style_ids=((1, 1), (2, 2), (4, 4), (6, 9), (7, 10)),
    controller_configs=None,
    id_selection=None,
):
    # controller_configs = load_controller_config(default_controller=controllers)   # somehow this line doesn't work for me

    layout_and_style_ids = (layout_and_style_ids[id_selection],)

    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_configs,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=True,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        # seed=seed,
        renderer = 'mjviewer',
        # render_camera="robot0_agentview_left",
        obj_instance_split=obj_instance_split,
        generative_textures=generative_textures,
        randomize_cameras=randomize_cameras,
        # layout_and_style_ids=layout_and_style_ids,
        translucent_robot=False,
    )
    

    env = robosuite.make(**env_kwargs)
    return env, env_kwargs

class RobocasaAbsoluteActionConverter:
    def __init__(self, dataset_path, demo_idx_to_initial_state, algo_name='bc'):
        # default BC config
        # config = config_factory(algo_name=algo_name)

        # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
        # must ran before create dataset
        # ObsUtils.initialize_obs_utils_with_config(config)
        # self.f = h5py.File(dataset_path, "r")
        self.demo_idx_to_initial_state = demo_idx_to_initial_state
        
        env_meta = get_env_metadata_from_dataset(dataset_path)
        print("env_meta = {}".format(env_meta))
        abs_env_meta = copy.deepcopy(env_meta)
        # abs_env_meta['env_kwargs']['controller_configs']['control_delta'] = False
        

        # setup env arguments
        env_kwargs = env_meta["env_kwargs"]
        env_kwargs["env_name"] = env_meta["env_name"]
        env_kwargs["has_renderer"] = False
        env_kwargs["renderer"] = "mjviewer"
        env_kwargs["has_offscreen_renderer"] = False
        env_kwargs["use_camera_obs"] = False
        
        # env = get_env_from_dataset(dataset_path)
        env = robosuite.make(**env_kwargs)
        # pdb.set_trace()
        # display robosuite path and version
        # robosuite_path = robosuite.__file__
        # robosuite_version = robosuite.__version__
        # print(f"Robosuite path: {robosuite_path}")


        # env = EnvUtils.create_env_from_metadata(env_meta=env_meta,
        #     render=False, 
        #     render_offscreen=False,
        #     use_image_obs=False, 
        # )
        assert len(env.robots) in (1, 2)
        abs_env_kwargs = abs_env_meta["env_kwargs"]
        abs_env_kwargs["env_name"] = abs_env_meta["env_name"]
        abs_env_kwargs["has_renderer"] = False
        abs_env_kwargs["renderer"] = "mjviewer"
        abs_env_kwargs["has_offscreen_renderer"] = False
        abs_env_kwargs["use_camera_obs"] = False
        abs_env_kwargs['controller_configs']['control_delta'] = False
        # 
        abs_env_kwargs['controller_configs']['composite_controller_specific_configs'] = {}
        abs_env_kwargs['controller_configs']['composite_controller_specific_configs']["body_part_ordering"] = ["right", "right_gripper", "base", "torso"]
        for part in abs_env_kwargs['controller_configs']['composite_controller_specific_configs']["body_part_ordering"]:
            abs_env_kwargs['controller_configs']['composite_controller_specific_configs'][part] = {}
            abs_env_kwargs['controller_configs']['composite_controller_specific_configs'][part]['control_delta'] = False
            abs_env_kwargs['controller_configs']['composite_controller_specific_configs'][part]['input_type'] = 'absolute'

        # abs_env_kwargs['controller_configs']['composite_controller_specific_configs']['right']['control_delta'] = False
        # abs_env_kwargs['controller_configs']['composite_controller_specific_configs']['right_gripper']['control_delta'] = False
        # abs_env_kwargs['controller_configs']['composite_controller_specific_configs']['base']['control_delta'] = False
        # abs_env_kwargs['controller_configs']['composite_controller_specific_configs']['torso']['control_delta'] = False

        abs_env = robosuite.make(**abs_env_kwargs)
        abs_env.robots[0].composite_controller.get_controller('right').input_type = 'absolute'

        # abs_env = EnvUtils.create_env_from_metadata(
        #     env_meta=abs_env_meta,
        #     render=False, 
        #     render_offscreen=False,
        #     use_image_obs=False, 
        # )
        # pdb.set_trace()
        abs_env_using_delta = abs_env.robots[0].composite_controller.get_controller('right').input_type
        assert abs_env_using_delta != 'delta'

        self.env = env
        self.abs_env = abs_env
        self.file = h5py.File(dataset_path, 'r')
    
    def __len__(self):
        return len(self.file['data'])

    def convert_actions(self, 
                        ep_idx,
            states: np.ndarray, 
            actions: np.ndarray) -> np.ndarray:
        """
        Given state and delta action sequence
        generate equivalent goal position and orientation for each step
        keep the original gripper action intact.
        """
        # in case of multi robot
        # reshape (N,14) to (N,2,7)
        # or (N,7) to (N,1,7)
        # pdb.set_trace()
        # print("actions shape", actions.shape)
        # assert 0 == 1

        initial_state = self.demo_idx_to_initial_state[ep_idx]

        stacked_actions = actions.reshape(*actions.shape[:-1],-1,12)

        env = self.env
        reset_to(env, initial_state)
        # pdb.set_trace()
        # generate abs actions
        action_goal_pos = np.zeros(
            stacked_actions.shape[:-1]+(3,), 
            dtype=stacked_actions.dtype)
        action_goal_ori = np.zeros(
            stacked_actions.shape[:-1]+(3,), 
            dtype=stacked_actions.dtype)
        
        all_deltas_except_arm = stacked_actions[..., 6:]
        action_gripper = stacked_actions[...,[-1]]
        for i in range(len(states)):
            # pdb.set_trace()
            # _ = reset_to(env, states[i]) # state is 150-length vector
            reset_to(env, {"states": states[i]})

            # taken from robot_env.py L#454
            for idx, robot in enumerate(env.robots):
                # run controller goal generator
                # pdb.set_trace()
                robot.control(stacked_actions[i,idx], policy_step=True)
            
                # read pos and ori from robots
                controller = robot.composite_controller
                action_goal_pos[i,idx] = controller.part_controllers['right'].goal_pos
                action_goal_ori[i,idx] = Rotation.from_matrix(controller.part_controllers['right'].goal_ori).as_rotvec()

        stacked_abs_actions = np.concatenate([
            action_goal_pos,
            action_goal_ori,
            all_deltas_except_arm
        ], axis=-1)
        # pdb.set_trace()
        abs_actions = stacked_abs_actions.reshape(actions.shape)
        return abs_actions

    def convert_idx(self, idx):
        file = self.file
        demo = file[f'data/demo_{idx}']
        # input
        states = demo['states'][:]
        actions = demo['actions'][:]

        # generate abs actions
        abs_actions = self.convert_actions(idx, states, actions)
        return abs_actions

    def convert_and_eval_idx(self, idx):
        env = self.env
        abs_env = self.abs_env
        file = self.file
        # first step have high error for some reason, not representative
        eval_skip_steps = 1
        # pdb.set_trace()
        demo = file[f'data/demo_{idx}']
        # input
        states = demo['states'][:]
        actions = demo['actions'][:]

        # arm_only actions 
        # arm_actions = actions[..., :7]

        # generate abs actions
        
        abs_actions = self.convert_actions(idx, states, actions)
        

        # verify
        robot0_eef_pos = demo['obs']['robot0_eef_pos'][:]
        robot0_eef_quat = demo['obs']['robot0_eef_quat'][:]

        delta_error_info = self.evaluate_rollout_error(
            env, idx, self.demo_idx_to_initial_state, states, actions, robot0_eef_pos, robot0_eef_quat, 
            metric_skip_steps=eval_skip_steps)
        abs_error_info = self.evaluate_rollout_error(
            abs_env, idx, self.demo_idx_to_initial_state, states, abs_actions, robot0_eef_pos, robot0_eef_quat,
            metric_skip_steps=eval_skip_steps)
        # pdb.set_trace()
        info = {
            'delta_max_error': delta_error_info,
            'abs_max_error': abs_error_info
        }
        return abs_actions, info

    @staticmethod
    def evaluate_rollout_error(env, ep_idx, demo_idx_to_initial_state,
            states, actions, 
            robot0_eef_pos, 
            robot0_eef_quat, 
            metric_skip_steps=1):
        # first step have high error for some reason, not representative

        # evaluate abs actions
        initial_state = demo_idx_to_initial_state[ep_idx]
        reset_to(env, initial_state)

        rollout_next_states = list()
        rollout_next_eef_pos = list()
        rollout_next_eef_quat = list()
        obs = reset_to(env, states[0])
        for i in range(len(states)):
            obs = reset_to(env,states[i])
            obs, reward, done, info = env.step(actions[i])
            obs = env._get_observations()
            # env.render()
            # pdb.set_trace()
            rollout_next_states.append(np.array(env.sim.get_state().flatten()))
            rollout_next_eef_pos.append(obs['robot0_eef_pos'])
            rollout_next_eef_quat.append(obs['robot0_eef_quat'])
            # pdb.set_trace()
        rollout_next_states = np.array(rollout_next_states)
        rollout_next_eef_pos = np.array(rollout_next_eef_pos)
        rollout_next_eef_quat = np.array(rollout_next_eef_quat)
        # pdb.set_trace()
        next_state_diff = states[1:] - rollout_next_states[:-1]
        max_next_state_diff = np.max(np.abs(next_state_diff[metric_skip_steps:]))

        next_eef_pos_diff = robot0_eef_pos[1:] - rollout_next_eef_pos[:-1]
        next_eef_pos_dist = np.linalg.norm(next_eef_pos_diff, axis=-1)
        max_next_eef_pos_dist = next_eef_pos_dist[metric_skip_steps:].max()

        next_eef_rot_diff = Rotation.from_quat(robot0_eef_quat[1:]) \
            * Rotation.from_quat(rollout_next_eef_quat[:-1]).inv()
        next_eef_rot_dist = next_eef_rot_diff.magnitude()
        max_next_eef_rot_dist = next_eef_rot_dist[metric_skip_steps:].max()

        info = {
            'state': max_next_state_diff,
            'pos': max_next_eef_pos_dist,
            'rot': max_next_eef_rot_dist
        }
        return info
