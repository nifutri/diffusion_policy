import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import pandas as pd
import pickle
import argparse
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix
import sys
# move out of the diffusion_policy directory
import torch
import pdb 
import tqdm

def get_obs_demos():
    filename = 'data/outputs/2025.06.02/latest_FD_data.pt'

    # filename = f'{type}_data{suffix}.pt'
    data = torch.load(filename)
    X, Y = data['X'], data['Y']
    successes = [1] * len(X)
    return X, Y, successes


def load_score_network():
    path  = '../UQ_baselines/logpZO/closedrawer_600_diffusion.ckpt'
    # load score network

    ## Get logpZO
    input_dim = 7
    net = get_unet(input_dim)
    # pdb.set_trace()
    ckpt = torch.load(path)
    net.load_state_dict(ckpt['model'])
    net.eval()

    # move net to device
    net.to('cuda:0')
    return net

def get_scores_for_demos(score_network):
    """
    Merge rollouts with demonstrations.
    """
    # Load demonstrations
    X, Y, demo_successes = get_obs_demos()
    print(f'Shape of X: {X.shape}, Shape of Y: {Y.shape}, Length of demo successes: {len(demo_successes)}')
    
    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    demo_successes = torch.tensor(demo_successes, dtype=torch.float32)

    # Concatenate rollouts and demonstrations
    # pdb.set_trace()
    X = X.to('cuda:0')
    baseline_metric = logpZO_UQ(score_network, X).unsqueeze(1)
    return baseline_metric.cpu().numpy(), X.cpu().numpy(),  demo_successes.cpu().numpy()


from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

def get_unet(input_dim):
    return ConditionalUnet1D(
        input_dim=input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=128,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=False
    )


def merge_rollouts_w_demos(successes, v_obs, nv_obs):
    """
    Merge rollouts with demonstrations.
    """
    # Load demonstrations
    X, Y, demo_successes = get_obs_demos()
    print(f'Shape of X: {X.shape}, Shape of Y: {Y.shape}, Length of demo successes: {len(demo_successes)}')
    
    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    demo_successes = torch.tensor(demo_successes, dtype=torch.float32)

    # Concatenate rollouts and demonstrations
    # pdb.set_trace()
    v_obs_combined = torch.cat([v_obs, X], dim=0)
    nv_obs_combined = torch.cat([nv_obs, Y], dim=0)
    successes_combined = torch.cat([torch.tensor(successes, dtype=torch.float32), demo_successes], dim=0)

    return successes_combined, v_obs_combined, nv_obs_combined

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






def get_obs_rollouts():
    """
    Load evaluation log and extract successes, visual/non-visual observations, and actions.
    """
    successes = []
    v_obs = []
    nv_obs = []
    log_scores = []
    infos = []
    filename = 'runner_log_diffusion_closedrawer_numtrajs100.pkl'
    with open(filename, 'rb') as file:
        data = pickle.load(file)

    max_length = 0
    for rollout_idx in range(len(data)):
        rollout_idx_to_successes, num_successes, logpZO_local_slices, aggregated_data = data[rollout_idx]
        success = rollout_idx_to_successes[rollout_idx]['is_success'][0]
        success = 1 if success else 0
        visual_obs = []
        non_visual_obs = []
        # pdb.set_trace()
        log_scores.append([elem.cpu().numpy() for elem in logpZO_local_slices])
        # pdb.set_trace()
        inf = aggregated_data['infos']
        dones = [elem['is_success'][0] for elem in inf]
        infos.append(dones)
        obs_encoding_over_time = aggregated_data['X_encodings'] 
        proprio_encodings_over_time = aggregated_data['Y_encodings']
        for t in range(len(obs_encoding_over_time)):
            obs_enc = obs_encoding_over_time[t][0].cpu().numpy()
            visual_obs.append(obs_enc)
            proprio_enc = proprio_encodings_over_time[t][0].cpu().numpy()
            non_visual_obs.append(proprio_enc)
        # pdb.set_trace()
        v_obs.append(np.array(visual_obs))
        nv_obs.append(np.array(non_visual_obs))
        successes.append(success)
        if len(visual_obs) > max_length:
            max_length = len(visual_obs)
    print(f'Max length of observations: {max_length}')

    # max_length = max(len(obs) for obs in v_obs + nv_obs)
    # # Pad the lists to the same length with the last observation
    # for i in range(len(v_obs)):
    #     if len(v_obs[i]) < max_length:
    #         v_obs[i].extend([v_obs[i][-1]] * (max_length - len(v_obs[i])))
    #     if len(nv_obs[i]) < max_length:
    #         nv_obs[i].extend([nv_obs[i][-1]] * (max_length - len(nv_obs[i])))
    # # Ensure all lists are of the same length
    # if len(v_obs) != len(nv_obs) or len(v_obs) != len(successes):
    #     raise ValueError("Length mismatch: v_obs, nv_obs, and successes must have the same length.")


    # Convert lists to tensors
    v_obs = np.array(v_obs)
    nv_obs = np.array(nv_obs)
    log_scores = np.array(log_scores)
    print("shape of v_obs:", v_obs.shape)
    print("shape of nv_obs:", nv_obs.shape)
    # pdb.set_trace()
    # # first convert to numpy
    # for i in range(len(v_obs)):
    #     print(f'Length of v_obs[{i}]: {len(v_obs[i])}, nv_obs[{i}]: {len(nv_obs[i])}, successes[{i}]: {successes[i]}')
    v_obs = torch.from_numpy(v_obs)
    nv_obs = torch.from_numpy(nv_obs)
    log_scores = torch.from_numpy(log_scores)

    # successes = torch.tensor(successes, dtype=torch.float32)
    success = np.array(successes, dtype=np.float32)

    return successes, v_obs, nv_obs, log_scores, infos

def rbf_kernel_matrix(X, gamma):
    """
    Compute the RBF kernel matrix for tensor X.
    """
    sq_dists = torch.cdist(X, X, p=2) ** 2
    K = torch.exp(-gamma * sq_dists)
    return K

def compute_mmd_from_kernel_matrix(K, n1):
    """
    Compute the MMD statistic from a kernel matrix.
    """
    K_XX = K[:n1, :n1]
    K_YY = K[n1:, n1:]
    K_XY = K[:n1, n1:]
    mmd_stat = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return mmd_stat

def bootstrap_null_statistics(A_sub1, A_sub2, gamma, num_bootstrap=1000, device='cpu'):
    """
    Bootstrap the null distribution of the MMD statistic.
    """
    A_sub = torch.cat([A_sub1, A_sub2], dim=0)
    N = A_sub.shape[0]
    N2 = A_sub2.shape[0]
    K = rbf_kernel_matrix(A_sub, gamma)
    bootstrap_stats = []
    for _ in range(num_bootstrap):
        indices = torch.randperm(N, device=device)[:N2 * 2]
        K_sub = K[indices][:, indices]
        mmd_stat = compute_mmd_from_kernel_matrix(K_sub, N2)
        bootstrap_stats.append(mmd_stat.item())
    return torch.tensor(bootstrap_stats, device=device).quantile(0.95)

def MMD_together(A, A_success, B):
    """
    Compute the MMD statistic and bootstrap threshold for two groups in A, split by B.
    """
    t=10 
    bw=1
    multiply=False
    multiplier=8
    device='cuda:0'
    d=1
    if len(A.shape)==3:
        T, d = A.shape[1:]
        start_idx = 0
        end_idx = T
        if multiply:
            start_idx *= multiplier
            end_idx *= multiplier
        print(f'Start index: {start_idx}, End index: {end_idx}')
        A_sub = A[:, start_idx:end_idx, :].to(device)
    else:
        A_sub = A.to(device)

    # pdb.set_trace()
    A_success = torch.from_numpy(A_success)
    A_success = A_success.to(device)
    # A = torch.from_numpy(A)
    A = A.to(device)
    # pdb.set_trace()
    B = np.array(B, dtype=np.float32)
    # A_sub1 = A_sub[B == 1].reshape(-1, d)

    A_sub1 = A_success
    # A_sub1 = A_sub[B == 1].reshape(-1, d)
    # A_sub2 = A
    A_sub2 = A_sub[B == 0].reshape(-1, d)
    print(f'Shape of A_sub1: {A_sub1.shape}, Shape of A_sub2: {A_sub2.shape}')
    A_sub_combined = torch.cat([A_sub1, A_sub2], dim=0)
    median_distance = torch.median(torch.cdist(A_sub_combined, A_sub_combined, p=2))
    gamma = 1.0 / (2 * median_distance ** 2)
    K_combined = rbf_kernel_matrix(A_sub_combined, gamma)
    mmd_stat = compute_mmd_from_kernel_matrix(K_combined, len(A_sub1))
    threshold = bootstrap_null_statistics(A_sub1, A_sub2, gamma, device=device)
    # plot_pca(A_sub1, A_sub2, B)
    return mmd_stat.item(), threshold.item()

def main():
    network = load_score_network()
    # get scores for demonstrations
    print("Get scores for demonstrations.")
    demo_logpzo, v_obs_demo, successes = get_scores_for_demos(network)
    
    print("Examine separation of rollout failures and successes in visual and non-visual observations.")
    successes, v_obs, nv_obs, logpzo, infos = get_obs_rollouts()
    # plot scores of logpZO
    logpzo=logpzo[:,:,0]
    failed_scores_at_time = {}
    success_scores_at_time = {}
    for rollout_idx in range(logpzo.shape[0]):
        # for t in range(logpzo.shape[1]):
        # if successes[rollout_idx] == 1:
        #     plt.plot(logpzo[rollout_idx].cpu().numpy(), label=f'Success Rollout {rollout_idx}', color='green')
        # else:
        #     plt.plot(logpzo[rollout_idx].cpu().numpy(), label=f'Failure Rollout {rollout_idx}', linestyle='--', color='red')
        # pdb.set_trace()
        rollout_scores = logpzo[rollout_idx].cpu().numpy()
        dones = infos[rollout_idx]
        last_rollout_score = 0
        for t in range(logpzo.shape[1]):
            if t not in failed_scores_at_time:
                failed_scores_at_time[t] = []
            if t not in success_scores_at_time:
                success_scores_at_time[t] = []
            if dones[t] == False:
                if successes[rollout_idx] == 0:
                    failed_scores_at_time[t].append(rollout_scores[t])
                else:
                    success_scores_at_time[t].append(rollout_scores[t])
            else:
                if dones[t] == True and dones[t-1] == False:
                    # if the current time step is done, we add the last score to the success scores
                    if successes[rollout_idx] == 0:
                        failed_scores_at_time[t].append(rollout_scores[t])
                    else:
                        success_scores_at_time[t].append(rollout_scores[t])
                    last_rollout_score = rollout_scores[t]
                # else:
                #     # add last score to the success scores
                #     if successes[rollout_idx] == 0:
                #         failed_scores_at_time[t].append(last_rollout_score)
                #     else:
                #         success_scores_at_time[t].append(last_rollout_score)
    # pdb.set_trace()
    # Plot the scores at each time step with mean and shaded standard deviation
    # success should be green, failure should be red
    plt.figure(figsize=(12, 6))
    failure_means = [np.mean(elem) for elem in failed_scores_at_time.values()]
    failure_stds = [np.std(elem) for elem in failed_scores_at_time.values()]
    success_means = [np.mean(elem) for elem in success_scores_at_time.values()]
    success_stds = [np.std(elem) for elem in success_scores_at_time.values()]
    time_steps = list(failed_scores_at_time.keys())
    plt.plot(time_steps, failure_means, label='Failure Mean', color='red')
    plt.fill_between(time_steps,
                     np.array(failure_means) - np.array(failure_stds),
                     np.array(failure_means) + np.array(failure_stds),
                     color='red', alpha=0.2)
    plt.plot(time_steps, success_means, label='Success Mean', color='green')
    plt.fill_between(time_steps,
                     np.array(success_means) - np.array(success_stds),
                     np.array(success_means) + np.array(success_stds),
                     color='green', alpha=0.2)
    # plt.axhline(y=0, color='black', linestyle='--', label='Zero Line')
    # plt.legend()

    plt.xlabel('Time Step')
    plt.ylabel('LogpZO Score')
    plt.title('LogpZO Scores for Rollouts')
    # plt.legend()
    plt.show()

    # pdb.set_trace()
    # v_obs_demo, Y, _ = get_obs_demos()
    # print shapes
    # print(f'Shape of v_obs: {v_obs.shape}, Shape of nv_obs: {nv_obs.shape}, Length of successes: {len(successes)}')
    print("successes:", successes)
    result = {}
    print('########## MMD non-visual ##########')

    # result['non_visual_P_S_P_T'] = MMD_together(nv_obs, successes)
    print('########## MMD visual ##########')
    # Nfirst = int(v_obs_demo.shape[0] * 0.5)
    # rand_indices = torch.randperm(v_obs_demo.shape[0])
    # Nsecond = v_obs_demo.shape[0] - Nfirst
    # v_obs_demo_first = v_obs_demo[rand_indices[:Nfirst]]
    # v_obs_demo_second = v_obs_demo[rand_indices[Nfirst:]]
    # # v_obs_demo_first = v_obs_demo[:Nfirst]
    # # v_obs_demo_second = v_obs_demo[Nfirst:]
    # result['visual_P_S_P_T'] = MMD_together(v_obs_demo_first, v_obs_demo_second, successes)
    # result['visual_P_S_P_T'] = MMD_together(v_obs_demo_first, v_obs_demo_second, successes)
    # result['visual_P_S_P_T'] = MMD_together(v_obs, v_obs_demo, successes)
    result['log_visual_P_S_P_T'] = MMD_together(logpzo, demo_logpzo, successes)
    print('####### Sanity check of successes')
    print("results:", result)
    import logging
    for key, value in result.items():
        logging.info(f'########## (MMD, threshold),  ##########')
        logging.info(f'{key}: ({value[0]:.2e}, {value[1]:.2e})')
    # print(f'Number of failures: {num_failure}')

    # check for rollouts versus demonstrations
    # print('########## MMD non-visual with demonstrations ##########')
    # successes, v_obs, nv_obs = merge_rollouts_w_demos(successes, v_obs, nv_obs)
    # print(f'Shape of v_obs: {v_obs.shape}, Shape of nv_obs: {nv_obs.shape}, Length of successes: {len(successes)}')
    # result['non_visual_P_S_P_T_with_demos'] = MMD_together(nv_obs, successes)
    # print('########## MMD visual with demonstrations ##########')
    # result['visual_P_S_P_T_with_demos'] = MMD_together(v_obs, successes)
    # print('####### Sanity check of successes with demonstrations')
    # print("results with demonstrations:", result)
    # # Save results to a file
    # with open('mmd_results.json', 'w') as f:
    #     json.dump(result, f, indent=4)
    
    

def plot_pca(A_sub1, A_sub2, successes):
    # from sklearn.decomposition import PCA
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    pca = PCA(n_components=2)
    # Fit PCA on the combined data
    combined_data = torch.cat([A_sub1, A_sub2], dim=0)
    pca.fit(combined_data.cpu().numpy())
    # Transform the data
    A_sub1_pca = pca.transform(A_sub1.cpu().numpy())
    A_sub2_pca = pca.transform(A_sub2.cpu().numpy())
    # Plot the PCA results
    plt.figure(figsize=(8, 6))
    plt.scatter(A_sub1_pca[:, 0], A_sub1_pca[:, 1], label='Group 1', alpha=0.5, color='blue')
    plt.scatter(A_sub2_pca[:, 0], A_sub2_pca[:, 1], label='Group 2', alpha=0.5, color='red')
    plt.title('PCA of Observations')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == "__main__":
    main()


