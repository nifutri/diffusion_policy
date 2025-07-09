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
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
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
import matplotlib.pyplot as plt

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

def create_eval_env_modified(
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
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        # renderer="mjviewer",
        # render_camera="robot0_agentview_right",
        seed=seed,
        obj_instance_split=obj_instance_split,
        generative_textures=generative_textures,
        randomize_cameras=randomize_cameras,
        # layout_and_style_ids=layout_and_style_ids,
        translucent_robot=False,
    )

    env = robosuite.make(**env_kwargs)
    return env

class EvalDiffusionUnetImageWorkspace(BaseWorkspace):
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

        # Read task name and configure human_path and tasks
        task_name = cfg.task_name
        self.payload_cfg.task.dataset.tasks = {
            task_name: None,
        }
        self.payload_cfg.task.dataset.tasks = {task_name: None}
        self.payload_cfg.task.dataset.human_path = TASK_NAME_TO_HUMAN_PATH[task_name]

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

    def create_environment_data_from_yaml(self, config, output_json_file):

        # Initialize data structure
        data = {"tasks": {}}

        # Populate the data structure with experiments
        for env_name, env_details in config["tasks"].items():
            num_experiments = env_details.get("num_experiments", 0)
            experiments = {
                f"demo_{i}": {"status": "pending", "success": -1}
                for i in range(num_experiments)
            }
            data["tasks"][env_name] = {
                "environment_file": env_details["environment_file"],
                "experiments": experiments
            }

        # Write the JSON file
        with open(output_json_file, "w") as f:
            json.dump(data, f, indent=4)

        print(f"JSON file '{output_json_file}' created successfully.")

    def get_earliest_pending_experiments(self, json_file, max_experiments):

        # Load the JSON data
        with open(json_file, "r") as f:
            data = json.load(f)

        # Iterate through environments
        for env_name, env_details in data["tasks"].items():
            # Filter pending experiments
            pending_experiments = {
                key: experiment
                for key, experiment in env_details["experiments"].items()
                if experiment["status"] == "pending"
            }

            # If there are pending experiments, build and return the structure
            if pending_experiments:
                # Limit to max_experiments
                limited_pending_experiments = dict(
                    list(pending_experiments.items())[:max_experiments]
                )
                return {
                    "tasks": {
                        env_name: {
                            "environment_file": env_details["environment_file"],
                            "experiments": limited_pending_experiments
                        }
                    }
                }

        return None

    def set_all_status_to_in_progress(self, data):

        # Iterate through environments
        for env in data["tasks"].values():
            # Iterate through experiments within the environment
            for experiment in env["experiments"].values():
                # Update the status
                experiment["status"] = "in_progress"
        
        return data

    def update_json_file(self, json_file, updated_data):

        with open(json_file, "r") as f:
            existing_data = json.load(f)

        def merge_dicts(source, target):

            for key, value in source.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    merge_dicts(value, target[key])
                else:
                    target[key] = value

        merge_dicts(updated_data, existing_data)

        with open(json_file, "w") as f:
            json.dump(existing_data, f, indent=4)

        print(f"JSON file '{json_file}' updated successfully.")

    def reset_to(self, env, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state that contains one or more of:
                - states (np.ndarray): initial state of the mujoco environment
                - model (str): mujoco scene xml

        Returns:
            observation (dict): observation dictionary after setting the simulator state (only
                if "states" is in @state)
        """
        should_ret = False
        if "model" in state:
            if state.get("ep_meta", None) is not None:
                # set relevant episode information
                ep_meta = json.loads(state["ep_meta"])
            else:
                ep_meta = {}
            if hasattr(env, "set_attrs_from_ep_meta"):  # older versions had this function
                env.set_attrs_from_ep_meta(ep_meta)
            elif hasattr(env, "set_ep_meta"):  # newer versions
                env.set_ep_meta(ep_meta)
            # this reset is necessary.
            # while the call to env.reset_from_xml_string does call reset,
            # that is only a "soft" reset that doesn't actually reload the model.
            env.reset()
            robosuite_version_id = int(robosuite.__version__.split(".")[1])
            if robosuite_version_id <= 3:
                from robosuite.utils.mjcf_utils import postprocess_model_xml

                xml = postprocess_model_xml(state["model"])
            else:
                # v1.4 and above use the class-based edit_model_xml function
                xml = env.edit_model_xml(state["model"])

            env.reset_from_xml_string(xml)
            env.sim.reset()
            # hide teleop visualization after restoring from model
            # env.sim.model.site_rgba[env.eef_site_id] = np.array([0., 0., 0., 0.])
            # env.sim.model.site_rgba[env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
        if "states" in state:
            env.sim.set_state_from_flattened(state["states"])
            env.sim.forward()
            should_ret = True

        # update state as needed
        if hasattr(env, "update_sites"):
            # older versions of environment had update_sites function
            env.update_sites()
        if hasattr(env, "update_state"):
            # later versions renamed this to update_state
            env.update_state()

        # if should_ret:
        #     # only return obs if we've done a forward call - otherwise the observations will be garbage
        #     return get_observation()
        return None

    def convert_observations(self, dataset, left_image_queue, right_image_queue, gripper_image_queue, clip_embedding):
        # save png of the three camera views before conversion
        # left_image_queue = deque(maxlen=dataset.img_obs_horizon)
        # right_image_queue = deque(maxlen=dataset.img_obs_horizon)

        left_image = np.stack([dataset.convert_frame(frame=frame, swap_rgb=dataset.swap_rgb) for frame in left_image_queue])
        right_image = np.stack([dataset.convert_frame(frame=frame, swap_rgb=dataset.swap_rgb) for frame in right_image_queue])
        gripper_image = np.stack([dataset.convert_frame(frame=frame, swap_rgb=dataset.swap_rgb) for frame in gripper_image_queue])

        # print(left_image.shape)
        # dataset.undo_operations_and_save(left_image, 'left_image.png')
        # dataset.undo_operations_and_save(right_image, 'right_image.png')
        # dataset.undo_operations_and_save(gripper_image, 'gripper_image.png')
        # exit()

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

        return {
            "task_description": clip_embedding,
            "left_image": left_image,
            "right_image": right_image,
            "gripper_image": gripper_image,
            }

    def run(self):

        experiment_record = f"{self.run_dir}/multi_environment_experiment_record.json"
        experiment_record_lock = f"{experiment_record}.lock"
        lock = FileLock(experiment_record_lock)
        with lock:
            # if os.path.exists(experiment_record):
            #     print("Experiment record already exists.")
            # else:
            #     print("Experiment record does not exist.")
            self.create_environment_data_from_yaml(self.payload_cfg.task.dataset, experiment_record)

            experiments_data = self.get_earliest_pending_experiments(experiment_record, self.cfg.number_of_tasks)
            # if experiments_data is None:
            #     print('All experiments are in progress or completed.')
            #     exit()
            experiments_data = self.set_all_status_to_in_progress(experiments_data)
            self.update_json_file(experiment_record, experiments_data)

        task_name = list(experiments_data['tasks'].keys())[0]

        with open("datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24/demo_gentex_im256_randcams_100_train_envs.pkl", "rb") as pickle_file:
            environment_data = pickle.load(pickle_file)

        # env = robosuite.make(**environment_data['env_kwargs'])

        demos = list(experiments_data['tasks'][task_name]['experiments'].keys())

        max_traj_len = self.cfg['max_traj_len']
        camera_names = environment_data['env_kwargs']['camera_names']   # ['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand']
        camera_height = round(self.payload_cfg.task.dataset['frame_height']/self.payload_cfg.task.dataset['aug']['crop'])
        camera_width = round(self.payload_cfg.task.dataset['frame_width']/self.payload_cfg.task.dataset['aug']['crop'])
        pred_horizon = self.payload_cfg.task['action_horizon']
        action_horizon = self.cfg['execution_horizon']

        for demo in demos:

            demo_number = int(demo.replace("demo_", ""))
            
            environment_data['env_kwargs']['has_renderer'] = True
            environment_data['env_kwargs']["renderer"] = "mjviewer"
            env = create_eval_env_modified(env_name=task_name, controller_configs=environment_data['env_kwargs']['controller_configs'], id_selection=demo_number//10)
            # pdb.set_trace()
            # initial_state = environment_data['demos'][demo]['initial_state']
            # self.reset_to(env, initial_state)

            env.reset()
            # pdb.set_trace()

            task_description = env.get_ep_meta()["lang"]
            task_description = open_clip.tokenize([task_description]) # returns torch.Size([1, 77])
            with torch.no_grad():
                clip_embedding = self.dataset.lang_model(task_description.to(self.device)).cpu().unsqueeze(0) # returns torch.Size([1, 1, 1024])

            video_path = f'{self.run_dir}/{task_name}_{demo}.mp4'
            video_writer = imageio.get_writer(video_path, fps=30)

            left_image_queue = deque(maxlen=self.payload_cfg.task.img_obs_horizon)
            right_image_queue = deque(maxlen=self.payload_cfg.task.img_obs_horizon)
            gripper_image_queue = deque(maxlen=self.payload_cfg.task.img_obs_horizon)
            env = DataCollectionWrapper(env, directory=self.run_dir)
            env.reset()

            for i in range(int(max_traj_len/action_horizon)):

                video_img = []
                visual_obs = env._get_observations()
                for cam_name in camera_names:
                    # im_og = env.sim.render(
                    #     height=camera_height, width=camera_width, camera_name=cam_name
                    # )[::-1]
                    # pdb.set_trace()
                    # im = env.sim.render(
                    #     height=camera_height, width=camera_width, camera_name=cam_name
                    # )[::-1]
                    # 
                    im = visual_obs[cam_name+'_image']
                    # flip im, currently it is upside down
                    im = np.flip(im, axis=0)
                    # pdb.set_trace()
                    video_img.append(im)

                left_image_queue.append(video_img[0])
                right_image_queue.append(video_img[1])
                gripper_image_queue.append(video_img[2])

                while(len(left_image_queue)) < self.payload_cfg.task.img_obs_horizon:
                    left_image_queue.append(video_img[0])
                    right_image_queue.append(video_img[1])
                    gripper_image_queue.append(video_img[2])

                batch = self.convert_observations(self.dataset, left_image_queue, right_image_queue, gripper_image_queue, clip_embedding)
                batch = {key: value.to(self.device, dtype=torch.float32) for key, value in batch.items()}

                action_pred = self.policy.predict_action(batch)
                action_pred = ((action_pred.detach().cpu().numpy() + 1) / 2) * (self.dataset.max - self.dataset.min) + self.dataset.min
                action_pred = np.squeeze(action_pred)
                action_pred = np.hstack((action_pred, [[0, 0, 0, 0, -1]] * action_pred.shape[0]))
                action_pred = action_pred[0:action_horizon]

                print(i)

                for step in range(action_pred.shape[0]):

                    env.step(action_pred[step])

                    # video render
                    video_img = []
                    visual_obs = env._get_observations()
                    for cam_name in camera_names:
                        # im = env.sim.render(
                        #     height=camera_height, width=camera_width, camera_name=cam_name
                        # )[::-1]
                        im = visual_obs[cam_name+'_image']
                        im = np.flip(im, axis=0)
                        video_img.append(im)
                        
                    video_img = np.concatenate(
                        video_img, axis=1
                    )  # concatenate horizontally
                    video_writer.append_data(video_img)
                    # print("video_img", video_img)
                    # plt.imshow(video_img)
                    # plt.axis('off')
                    # plt.show(block=False)
                    # plt.pause(0.00001)

                    if env._check_success():
                        break
                
                if env._check_success():
                    break

            experiments_data['tasks'][task_name]['experiments'][demo]['status'] = 'done'
            if env._check_success():
                experiments_data['tasks'][task_name]['experiments'][demo]['success'] = 1
            else:
                experiments_data['tasks'][task_name]['experiments'][demo]['success'] = 0

            print(colored(f"Saved video to {video_path}", "green"))
            video_writer.close()

            with lock:
                self.update_json_file(experiment_record, experiments_data)

            env.close()

        pass


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = EvalDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()