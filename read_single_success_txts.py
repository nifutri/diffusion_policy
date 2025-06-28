import json
import os
import pathlib
import numpy as np
import pdb

experiment_name = 'train_diffusion_unet_clip'
experiment_tag = 'ST_OOD_DAgger'
task_name = 'CoffeePressButton'
finetune_type = 'dagger_episode_0_finetune_w_human_onlyFalse_useonlyoriginalFalse_fromscratchTrue_after_dagger'
epoch_folder = 'eval_compute_rollout_scores_latest'


result_path = f'data/outputs/{experiment_tag}_{experiment_name}_{task_name}/{finetune_type}/{epoch_folder}/successes.txt'
# read the file line by line and look for failure or success, save 1 or 0
successes = []
with open(result_path, 'r') as f:
    for line in f:
        if 'Failure' in line:
            successes.append(0)
        elif 'Success' in line:
            successes.append(1)
        else:
            print(f'Unknown line: {line}')
print("successes", successes)
mean_success = np.mean(successes)
ste_success = np.std(successes) / np.sqrt(len(successes))

print(f'Mean success: {mean_success}, STE success: {ste_success}')