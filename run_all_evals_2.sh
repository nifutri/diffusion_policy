#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_finetune_w_human_onlyFalse_useonlyoriginalFalse_fromscratchFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_220_step_15031'
# CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_finetune_w_human_onlyFalse_useonlyoriginalFalse_fromscratchFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_240_step_16331'
# CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_finetune_w_human_onlyFalse_useonlyoriginalFalse_fromscratchFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_260_step_17631'
# CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_finetune_w_human_onlyFalse_useonlyoriginalFalse_fromscratchFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_280_step_18931'
# CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_finetune_w_human_onlyFalse_useonlyoriginalFalse_fromscratchFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_300_step_20231'

# CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_finetune_w_human_onlyTrue_useonlyoriginalTrue_fromscratchFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_220_step_14737'
# CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_finetune_w_human_onlyTrue_useonlyoriginalTrue_fromscratchFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_240_step_15757'
# CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_finetune_w_human_onlyTrue_useonlyoriginalTrue_fromscratchFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_260_step_16777'
# CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_finetune_w_human_onlyTrue_useonlyoriginalTrue_fromscratchFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_280_step_17797'
# CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_finetune_w_human_onlyTrue_useonlyoriginalTrue_fromscratchFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_300_step_18817'


# CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_old0.25_new0.75_fromscratchFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_220_step_14947'
# CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_old0.25_new0.75_fromscratchFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_240_step_16167'
# CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_old0.25_new0.75_fromscratchFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_260_step_17387'
# CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_old0.25_new0.75_fromscratchFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_280_step_18607'
# CUDA_VISIBLE_DEVICES=2 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_old0.25_new0.75_fromscratchFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_300_step_19827'
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_lora_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_frozenobsTrue_loraonobsFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_220_step_15010'
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_lora_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_frozenobsTrue_loraonobsFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_240_step_16290'
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_lora_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_frozenobsTrue_loraonobsFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_260_step_17570'
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_lora_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_frozenobsTrue_loraonobsFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_280_step_18850'
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_lora_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_frozenobsTrue_loraonobsFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_300_step_20130'

CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_lora_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_frozenobsFalse_loraonobsFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_220_step_15010'
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_lora_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_frozenobsFalse_loraonobsFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_240_step_16290'
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_lora_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_frozenobsFalse_loraonobsFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_260_step_17570'
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_lora_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_frozenobsFalse_loraonobsFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_280_step_18850'
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_lora_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_frozenobsFalse_loraonobsFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_300_step_20130'


CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_old0.5_new0.5_fromscratchFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_220_step_15808'
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_old0.5_new0.5_fromscratchFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_240_step_17848'
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_old0.5_new0.5_fromscratchFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_260_step_19888'
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_old0.5_new0.5_fromscratchFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_280_step_21928'
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_old0.5_new0.5_fromscratchFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_300_step_23968'


CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_old0.75_new0.25_fromscratchFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_220_step_15094'
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_old0.75_new0.25_fromscratchFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_240_step_16454'
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_old0.75_new0.25_fromscratchFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_260_step_17814'
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_old0.75_new0.25_fromscratchFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_280_step_19174'
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_old0.75_new0.25_fromscratchFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_300_step_20534'




# CUDA_VISIBLE_DEVICES=1 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_finetune_w_human_onlyFalse_useonlyoriginalFalse_fromscratchTrue_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_280_step_18264'
# CUDA_VISIBLE_DEVICES=1 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_finetune_w_human_onlyFalse_useonlyoriginalFalse_fromscratchTrue_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_200_step_13064'

# CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_lora_finetuning_robocasa dagger.ckpt_folder='lora_dagger_episode_0_loraonobsFalse_after_dagger' task_name='CoffeePressButton' dagger.ckpt_name='epoch_300_step_20231'
