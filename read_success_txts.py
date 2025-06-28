import json
import os
import pathlib
import numpy as np
import pdb

experiment_name = 'train_diffusion_unet_clip'
experiment_tag = 'ST_OOD_DAgger'
task_name = 'CoffeePressButton'
finetune_type1 = 'dagger_episode_0_finetune_w_human_onlyFalse_useonlyoriginalFalse_fromscratchFalse_after_dagger'
finetune_type2 = 'dagger_episode_0_finetune_w_human_onlyTrue_useonlyoriginalTrue_fromscratchFalse_after_dagger'

finetune_type_epoch_list = {finetune_type1: ['eval_compute_rollout_scores_epoch_220_step_15031',
    'eval_compute_rollout_scores_epoch_240_step_16331', 
    'eval_compute_rollout_scores_epoch_260_step_17631',
    'eval_compute_rollout_scores_epoch_280_step_18931',
    'eval_compute_rollout_scores_epoch_300_step_20231'],
    finetune_type2:['eval_compute_rollout_scores_epoch_220_step_14737',
    'eval_compute_rollout_scores_epoch_240_step_15757', 
    'eval_compute_rollout_scores_epoch_260_step_16777',
    'eval_compute_rollout_scores_epoch_280_step_17797',
    'eval_compute_rollout_scores_epoch_300_step_18817'],
}

finetune_type_to_mean_success = {}
finetune_type_to_ste_success = {}
for finetune_type in [finetune_type1, finetune_type2]:

    epoch_list = finetune_type_epoch_list[finetune_type]


    epoch_to_mean_success = []
    epoch_to_ste_success = []
    for epoch_folder in epoch_list:
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

        epoch_to_mean_success.append(mean_success)
        epoch_to_ste_success.append(ste_success)

    finetune_type_to_mean_success[finetune_type] = epoch_to_mean_success
    finetune_type_to_ste_success[finetune_type] = epoch_to_ste_success

# pdb.set_trace()
# plot the results as a line graph with error bars
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(10, 6))

finetune_type_to_name = {
    finetune_type1: 'Finetune D_0+D_H',
    finetune_type2: 'Finetune D_0'
}

for finetune_type in [finetune_type1, finetune_type2]:
    mean_success = finetune_type_to_mean_success[finetune_type]
    ste_success = finetune_type_to_ste_success[finetune_type]
    plt.errorbar(
        [220, 240, 260, 280, 300], 
        mean_success, 
        yerr=ste_success, 
        label=finetune_type_to_name[finetune_type],
        marker='o'
    )
plt.xlabel('Epoch')
plt.ylabel('Mean Success Rate')
plt.title(f'Success Rate for {task_name}')
plt.xticks([220, 240, 260, 280, 300])
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(20))
plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(5))
plt.legend()
plt.tight_layout()

plt.show()















