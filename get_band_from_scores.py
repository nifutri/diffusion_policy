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
import pdb
import torch
from timeseries_cp.utils.data_utils import RegressionType
from timeseries_cp.methods.functional_predictor import FunctionalPredictor, ModulationType
fsize = 12
def adjust_label(ax, factor=8):
    """Adjust x-axis labels by multiplying with a factor."""
    current_ticks = ax.get_xticks()
    new_ticks = [int(i) for i in current_ticks * factor]
    ax.set_xticklabels(new_ticks)

def plot_on_subfig_traj(ax, log_probs, successes, factor=8):
    """Plot trajectories on a subplot, distinguishing between success and failure.
    Args:
        ax: Matplotlib axis object
        log_probs: List of log probability trajectories
        successes: List of binary success indicators
        factor: Scaling factor for x-axis
    """
    failure_plotted = False
    success_plotted = False
    multiplier = 8
    xaxis = np.arange(len(log_probs[0])) * multiplier
    for log_prob, success in zip(log_probs, successes):
        alpha = 0.1 if success == 1 else 0.5
        color = 'blue' if success else 'red'
        label = 'Success' if success else 'Failure'
        if success and not success_plotted:
            ax.plot(xaxis, log_prob, color=color, label=label, alpha=alpha)
            success_plotted = True
        elif not success and not failure_plotted:
            ax.plot(xaxis, log_prob, color=color, label=label, alpha=alpha)
            failure_plotted = True
        else:
            ax.plot(xaxis, log_prob, color=color, alpha=alpha)
    ax.set_xlabel('Time Step', fontsize=fsize)
    ax.legend(fontsize=fsize-4, loc='upper center', bbox_to_anchor=(0.42, -0.16), ncol=2)
    ax.tick_params(axis='both', which='major', labelsize=fsize-4)
    ax.grid()


def plot_on_subfig_traj_new(ax, log_probs, successes, predictions, plot_TP=False):
    """Plot mean and standard error of trajectories grouped by prediction outcomes.
    Args:
        ax: Matplotlib axis object
        log_probs: List of log probability trajectories
        successes: List of binary success indicators
        predictions: List of binary prediction outcomes
        plot_TP: If True, plot True Positives/Negatives; if False, plot False Positives/Negatives
    """
    fp_log_probs, fn_log_probs, tp_log_probs, tn_log_probs = [], [], [], []
    multiplier = 8
    xaxis = np.arange(len(log_probs[0])) * multiplier
    for log_prob, success, prediction in zip(log_probs, successes, predictions):
        if success == 1 and prediction == 1:
            fp_log_probs.append(log_prob)
        elif success == 0 and prediction == 1:
            tp_log_probs.append(log_prob)
        elif success == 0 and prediction == 0:
            fn_log_probs.append(log_prob)
        elif success == 1 and prediction == 0:
            tn_log_probs.append(log_prob)

    def plot_mean_with_error(ax, log_probs, color, label):
        mean_log_prob = np.mean(log_probs, axis=0)
        std_error_log_prob = np.std(log_probs, axis=0) / np.sqrt(len(log_probs))
        ax.plot(xaxis, mean_log_prob, color=color, label=label)
        ax.fill_between(xaxis, mean_log_prob - std_error_log_prob, mean_log_prob + std_error_log_prob, color=color, alpha=0.3)

    if plot_TP:
        if tp_log_probs:
            plot_mean_with_error(ax, tp_log_probs, 'red', 'TP')
        if tn_log_probs:
            plot_mean_with_error(ax, tn_log_probs, 'blue', 'TN')
    else:
        if fp_log_probs:
            plot_mean_with_error(ax, fp_log_probs, 'blue', 'FP')
        if fn_log_probs:
            plot_mean_with_error(ax, fn_log_probs, 'red', 'FN')

    ax.set_xlabel('Time Step', fontsize=fsize)
    ax.legend(fontsize=fsize-4, loc='upper center', bbox_to_anchor=(0.42, -0.16), ncol=2)
    ax.tick_params(axis='both', which='major', labelsize=fsize-4)
    ax.grid()

def plot_on_subfig_traj_failure(ax, scores, n_idx, small=True):
    """Plot specific failure trajectories with custom styling.
    Args:
        ax: Matplotlib axis object
        scores: List of score trajectories
        n_idx: List of indices to plot
        small: If True, plot smallest failures; if False, plot largest failures
    """
    colors = plt.cm.viridis(np.linspace(0, 1, len(n_idx)))
    fsize = 28
    for i, score in enumerate(scores):
        alpha = 1
        if i not in n_idx:
            continue
        kk = n_idx.index(i) + 1
        midfix = 'smallest' if small else 'largest'
        label_dict = {1: f'{midfix} failure', 2: 'median success'}
        label = label_dict[kk]
        ax.plot(range(len(score)), score, '-o', color=colors[kk-1], label=label, alpha=alpha)
    ax.set_xlabel('Time Step', fontsize=fsize)
    ax.set_ylabel('Score', fontsize=fsize)
    # Custom legend
    ax.legend(fontsize=fsize, loc='upper center', 
            bbox_to_anchor=(0.5, -0.2),  ncol=1,
            title_fontsize=fsize-2)
    ax.tick_params(axis='both', which='major', labelsize=fsize-4)
    adjust_label(ax)

def get_detection_with_plot(log_probs, successes, alpha=0.1, lb=False, CPband=True, suffix=''):
    

    """Detect anomalies using prediction bands and create visualization plots.
    Returns:
        Tuple containing (first_idx_ls, positive_ls, successes_test, amount_exceed_ratio)
    """
    num_train = 40
    num_test = 20
    num_cal = 40

    num_te = 20
    max_tr = 80



    log_probs_train = log_probs[:max_tr]; successes_train = successes[:max_tr]
    log_probs_test = log_probs[max_tr:max_tr+num_te]; successes_test = successes[max_tr:max_tr+num_te]
    log_probs_train = np.array([log_probs_train[i] for i, success in enumerate(successes_train) if success])
    ntr = int(len(log_probs_train) * num_train / (num_train + num_cal))
    ncal = len(log_probs_train) - ntr

    print(f'#### Use {ntr} trajectories for training and {ncal} for calibration')
    print(f'#### Number of training trajectories: {len(log_probs_train)}')
    print("#### Number of test trajectories: ", len(log_probs_test))
    print(f'#### Use {len(log_probs_train)} successful trajectories for calibration')
    predictor = FunctionalPredictor(modulation_type=ModulationType.Tfunc, regression_type=RegressionType.Mean)
    if CPband:
        print(f'Number of success for mean {ntr} and for band {ncal}')
        target_traj = predictor.get_one_sided_prediction_band(log_probs_train[:ntr], log_probs_train[-ncal:], alpha=alpha, lower_bound=lb).flatten()
    else:
        metric_tr = [np.cumsum(val)[-1] for val in log_probs_train]
        threshold = np.quantile(metric_tr, 1 - alpha)
        # Repeat for each trajectory
        target_traj = np.repeat(threshold, len(log_probs_train[0]))


    print(f'###  Mean of band width: {np.mean(target_traj)}')
    to_plot = min(150, len(log_probs_test))
    rand_idx = np.random.choice(len(log_probs_test), to_plot, replace=False)
    log_probs_test_plt = [log_probs_test[i] for i in rand_idx]
    successes_test_plt = [successes_test[i] for i in rand_idx]

    first_idx_ls = []; positive_ls = []; amount_exceed_success = []; amount_exceed_failure = []
    for log_prob_test, success in zip(log_probs_test, successes_test):
        positive = 0
        for i, log_prob in enumerate(log_prob_test):
            cond = log_prob <= target_traj[i] if lb else log_prob >= target_traj[i]
            if cond:
                if success < 1:
                    # Record the first index of failure when ground truth is failure
                    first_idx_ls.append(i)
                positive = 1
                if success:
                    amount_exceed_success.append(np.abs(log_prob - target_traj[i]))
                else:
                    amount_exceed_failure.append(np.abs(log_prob - target_traj[i]))
                break
        positive_ls.append(positive)

    if len(amount_exceed_failure) > 0:
        eps = 1e-5
        amount_exceed_ratio = np.mean(amount_exceed_failure) / (np.mean(amount_exceed_success) + eps)
    else:
        amount_exceed_ratio = 0

    fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    multiplier = 8; xaxis = np.arange(len(log_probs_test_plt[0])) * multiplier
    func = np.max if lb else np.min
    placeholder_traj = func(log_probs_train) * np.ones_like(target_traj)
    if lb:
        upper, lower = placeholder_traj, target_traj
    else:
        upper, lower = target_traj, placeholder_traj

    # pdb.set_trace()
    for a in ax:
        a.fill_between(xaxis, upper, lower, color='blue', alpha=0.25)     

    fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    multiplier = 8; xaxis = np.arange(len(log_probs_test_plt[0])) * multiplier
    func = np.max if lb else np.min
    placeholder_traj = func(log_probs_train) * np.ones_like(target_traj)
    if lb:
        upper, lower = placeholder_traj, target_traj
    else:
        upper, lower = target_traj, placeholder_traj
    for a in ax:
        a.fill_between(xaxis, upper, lower, color='blue', alpha=0.25)

    output_dir=''
    plot_on_subfig_traj(ax[0], log_probs_test_plt, successes_test_plt)
    positive_ls_plt = [positive_ls[i] for i in rand_idx]
    plot_on_subfig_traj_new(ax[1], log_probs_test_plt, successes_test_plt, positive_ls_plt, plot_TP=True)
    plot_on_subfig_traj_new(ax[2], log_probs_test_plt, successes_test_plt, positive_ls_plt, plot_TP=False)
    for a in ax:
        a.set_ylabel('Score', fontsize=fsize)
        a.set_xlabel('Time Step', fontsize=fsize)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'prediction_band{suffix}.png'), bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close('all')
    first_idx_ls = np.array(first_idx_ls); positive_ls = np.array(positive_ls); successes_test = np.array(successes_test)
    return first_idx_ls, positive_ls, 1-successes_test, amount_exceed_ratio          


def plot_all_scores(log_probs, successes):
    # plot all scores, color red for failure and blue for success
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), sharex=True, sharey=True)
    # multiplier = 8
    # xaxis = np.arange(len(log_probs[0])) 
    for i in range(len(log_probs)):
        log_prob = log_probs[i]
        color = 'blue' if successes[i]==1 else 'red'
        label = 'Success' if successes[i]==1 else 'Failure'
        # duplicate the log_prob for plotting
        # log_prob = np.array(log_prob)
        # log_prob_mult = np.repeat(log_prob, multiplier)
        ax.plot(np.arange(len(log_prob)), log_prob, color=color, label=label, alpha=0.4)
        # ax[1].plot(xaxis, log_prob, color=color, alpha=0.1)
    ax.set_title('All Trajectories', fontsize=fsize)

    ax.set_xlabel('Time Step', fontsize=fsize)

    ax.set_ylabel('Score', fontsize=fsize)
    # ax.legend(fontsize=fsize-4, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax.tick_params(axis='both', which='major', labelsize=fsize-4)

    ax.grid()

    fig.tight_layout()
    fig.savefig('all_scores.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()


if __name__ == "__main__":
    policy_type = 'diffusion'
    task_name = 'closedrawer'
    num_inference_step = 100  # Default value, can be adjusted based on task
    data_file = f'runner_log_{policy_type}_{task_name}_numtrajs{num_inference_step}.pkl'
    with open(data_file, 'rb') as f:
        runner_log = pickle.load(f)

    successes = []
    all_log_probs = []
    max_length = 0
    for rollout_idx in range(len(runner_log)):
        success = runner_log[rollout_idx][0][rollout_idx]['is_success'][0]
        success = 1 if success else 0
        successes.append(success)

        # log_probs
        log_probs = []
        score_data = runner_log[rollout_idx][2]
        for j in range(len(score_data)):
            elem = score_data[j]
            # convert to scalr from tensor
            log_prob = elem[0].item() if isinstance(elem[0], torch.Tensor) else elem[0]
            log_probs.append(log_prob)
            if success == 1 and abs(score_data[j][0].item()-score_data[j-1][0].item())<1:
                break

        all_log_probs.append(log_probs)
    max_length = max(len(log_probs) for log_probs in all_log_probs)
    print("max_length", max_length)
    # pad each log_probs with the last value to make them the same length
    padded_log_probs = []
    for log_probs in all_log_probs:
        if len(log_probs) < max_length:
            log_probs += [log_probs[-1]] * (max_length - len(log_probs))
        padded_log_probs.append(log_probs)
    # Convert to numpy array
    log_probs = np.array(padded_log_probs)
    successes = np.array(successes)



    # pdb.set_trace()
    plot_all_scores(all_log_probs, successes)

    get_detection_with_plot(log_probs, successes, alpha=0.1)