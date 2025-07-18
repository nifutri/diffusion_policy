from typing import Dict
import torch
import torch.nn as nn
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.normalizer import LinearNormalizer

# more custom imports for the blending implementation:
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
import h5py

class BaseBlendingPolicy(ModuleAttrMixin):  
    # ========= inference  ============
    # also as self.device and self.dtype for inference device transfer
    def predict_action(self, obs):
        """
        obs_dict:
            obs: B,To,Do
        return: 
            action: B,Ta,Da
        To = 3
        Ta = 4
        T = 6
        |o|o|o|
        | | |a|a|a|a|
        |o|o|
        | |a|a|a|a|a|
        | | | | |a|a|
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass

    

class SampleBasedBlending(BaseBlendingPolicy):
    def __init__(self, 
        proposal_policy,
        bad_policy,
        blending_horizon=8,
        num_proposals=25,
        pos_cropping=0.02,
        rot_cropping=0.02,
        pos_weighting_good_bad=1.0,
        rot_weighting_good_bad=0.1,
        pos_weighting_good_good=1.0,
        rot_weighting_good_good=0.1,
        greedy_behavior=1,
        policy_inference_steps=50,
        # parameters passed to step
        **kwargs
    ):
        super().__init__()

        self.blending_horizon = blending_horizon
        self.num_proposals = num_proposals
        self.pos_cropping = pos_cropping
        self.rot_cropping = rot_cropping
        self.greedy_behavior = greedy_behavior

        self.pos_weighting_good_bad = pos_weighting_good_bad
        self.rot_weighting_good_bad = rot_weighting_good_bad
        self.pos_weighting_good_good = pos_weighting_good_good
        self.rot_weighting_good_good = rot_weighting_good_good

        device = torch.device('cuda')

        # first load the proposal policy:
        payload_proposal_policy = torch.load(open(proposal_policy, 'rb'), map_location='cpu', pickle_module=dill)
        payload_proposal_cfg = payload_proposal_policy['cfg']

        cls = hydra.utils.get_class(payload_proposal_cfg._target_)
        workspace = cls(payload_proposal_cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload_proposal_policy, exclude_keys=None, include_keys=None)
        policy = workspace.ema_model
        policy.num_inference_steps = policy_inference_steps

        policy.eval().to(device)
        self.proposal_policy = policy

        # then load the bad policy
        payload_bad_policy = torch.load(open(bad_policy, 'rb'), map_location='cpu', pickle_module=dill)
        payload_bad_cfg = payload_bad_policy['cfg']
        cls = hydra.utils.get_class(payload_bad_cfg._target_)
        workspace = cls(payload_proposal_cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload_bad_policy, exclude_keys=None, include_keys=None)
        bad_policy = workspace.ema_model
        bad_policy.num_inference_steps = policy_inference_steps
        bad_policy.eval().to(device)
        self.bad_policy = bad_policy

    def set_dataset_info(self, dataset_info):
        # this is done to make sure that we can normalize, etc,...
        self.dataset_info = dataset_info

    def predict_action(self, obs):
        if self.greedy_behavior != 1 and self.num_proposals > 1:
            # we need to increase along the batch dimension
            for key in obs:
                if obs[key].ndim == 3:
                    obs[key] = obs[key].repeat(self.num_proposals, 1, 1)
                elif obs[key].ndim == 4:
                    obs[key] = obs[key].repeat(self.num_proposals, 1, 1, 1)
                elif obs[key].ndim == 5:
                    obs[key] = obs[key].repeat(self.num_proposals, 1, 1, 1, 1)
                else:
                    raise ValueError(f"Unsupported observation dimension: {obs[key].ndim} for key {key}")
                
        if self.greedy_behavior == 1:
            # we use the proposal policy to get the action
            action_pred = self.proposal_policy.predict_action(obs)
            action_pred = ((action_pred.detach().cpu().numpy() + 1) / 2) * (self.dataset_info.max - self.dataset_info.min) + self.dataset_info.min
            return np.squeeze(action_pred)
        
        else:
            proposal_actions = self.proposal_policy.predict_action(obs)
            proposal_actions = ((proposal_actions.detach().cpu().numpy() + 1) / 2) * (self.dataset_info.max - self.dataset_info.min) + self.dataset_info.min
            bad_actions = self.bad_policy.predict_action(obs)
            bad_actions = ((bad_actions.detach().cpu().numpy() + 1) / 2) * (self.dataset_info.max - self.dataset_info.min) + self.dataset_info.min

            import scipy.spatial.distance as ssd
            pairwise_distances_position = ssd.cdist(
                proposal_actions[:,:self.blending_horizon,:3].reshape(self.num_proposals,-1),
                bad_actions[:,:self.blending_horizon,:3].reshape(self.num_proposals,-1),
                metric='euclidean'
            )

            pairwise_distances_rot = ssd.cdist(
                proposal_actions[:,:self.blending_horizon,3:6].reshape(self.num_proposals,-1),
                bad_actions[:,:self.blending_horizon,3:6].reshape(self.num_proposals,-1),
                metric='euclidean'
            )


            pairwise_distances_gg_position = ssd.cdist(
                proposal_actions[:,:self.blending_horizon,:3].reshape(self.num_proposals,-1),
                bad_actions[:,:self.blending_horizon,:3].reshape(self.num_proposals,-1),
                metric='euclidean'
            )

            pairwise_distances_gg_rot = ssd.cdist(
                proposal_actions[:,:self.blending_horizon,3:6].reshape(self.num_proposals,-1),
                bad_actions[:,:self.blending_horizon,3:6].reshape(self.num_proposals,-1),
                metric='euclidean'
            )

            acc_distances_position = np.sum(np.clip(pairwise_distances_position,0,self.pos_cropping), axis=1)
            acc_distances_rot = np.sum(np.clip(pairwise_distances_rot,0,self.rot_cropping), axis=1)

            acc_distances_gg_position = np.sum(np.clip(pairwise_distances_gg_position,0,self.pos_cropping), axis=1)
            acc_distances_gg_rot = np.sum(np.clip(pairwise_distances_gg_rot,0,self.rot_cropping), axis=1)

            accumulated_distances = (
                self.pos_weighting_good_bad * acc_distances_position +
                self.rot_weighting_good_bad * acc_distances_rot -
                self.pos_weighting_good_good * acc_distances_gg_position -
                self.rot_weighting_good_good * acc_distances_gg_rot
            )


            arg_max_dist = np.argmax(accumulated_distances)

            action_pred = proposal_actions[arg_max_dist,...]
            return np.squeeze(action_pred)
        
