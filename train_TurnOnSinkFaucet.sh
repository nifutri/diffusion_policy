#!/bin/bash
CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python finetune.py --config-dir=. --config-name=finetune_after_dagger_robocasa.yaml training.seed=42 task.name='TurnOnSinkFaucet' finetuning.dagger_episode_folder='dagger_episode_0' finetuning.human_only=False finetuning.from_scratch=False finetuning.stop_at_epoch=201
CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python finetune.py --config-dir=. --config-name=finetune_after_dagger_robocasa.yaml training.seed=42 task.name='TurnOnSinkFaucet' finetuning.dagger_episode_folder='dagger_episode_0' finetuning.human_only=False finetuning.from_scratch=False finetuning.freeze_obs_encoder=True  finetuning.stop_at_epoch=201

CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python finetune.py --config-dir=. --config-name=finetune_lora_after_dagger_robocasa.yaml training.seed=42 task.name='TurnOnSinkFaucet' finetuning.dagger_episode_folder='dagger_episode_0' finetuning.apply_lora_on_obs_encoder=False finetuning.freeze_obs_encoder=True finetuning.stop_at_epoch=201
CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python finetune.py --config-dir=. --config-name=finetune_lora_after_dagger_robocasa.yaml training.seed=42 task.name='TurnOnSinkFaucet' finetuning.dagger_episode_folder='dagger_episode_0' finetuning.apply_lora_on_obs_encoder=False finetuning.stop_at_epoch=201


CUDA_VISIBLE_DEVICES=2 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_freezeobsTrue_finetune_w_human_onlyFalse_useonlyoriginalFalse_fromscratchFalse_after_dagger' task_name='TurnOnSinkFaucet' dagger.ckpt_name='epoch_120'
CUDA_VISIBLE_DEVICES=2 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_freezeobsTrue_finetune_w_human_onlyFalse_useonlyoriginalFalse_fromscratchFalse_after_dagger' task_name='TurnOnSinkFaucet' dagger.ckpt_name='epoch_140'
CUDA_VISIBLE_DEVICES=2 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_freezeobsTrue_finetune_w_human_onlyFalse_useonlyoriginalFalse_fromscratchFalse_after_dagger' task_name='TurnOnSinkFaucet' dagger.ckpt_name='epoch_160'
CUDA_VISIBLE_DEVICES=2 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_freezeobsTrue_finetune_w_human_onlyFalse_useonlyoriginalFalse_fromscratchFalse_after_dagger' task_name='TurnOnSinkFaucet' dagger.ckpt_name='epoch_180'
CUDA_VISIBLE_DEVICES=2 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_freezeobsTrue_finetune_w_human_onlyFalse_useonlyoriginalFalse_fromscratchFalse_after_dagger' task_name='TurnOnSinkFaucet' dagger.ckpt_name='epoch_200'



CUDA_VISIBLE_DEVICES=2 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_freezeobsFalse_finetune_w_human_onlyFalse_useonlyoriginalFalse_fromscratchFalse_after_dagger' task_name='TurnOnSinkFaucet' dagger.ckpt_name='epoch_120'
CUDA_VISIBLE_DEVICES=2 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_freezeobsFalse_finetune_w_human_onlyFalse_useonlyoriginalFalse_fromscratchFalse_after_dagger' task_name='TurnOnSinkFaucet' dagger.ckpt_name='epoch_140'
CUDA_VISIBLE_DEVICES=2 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_freezeobsFalse_finetune_w_human_onlyFalse_useonlyoriginalFalse_fromscratchFalse_after_dagger' task_name='TurnOnSinkFaucet' dagger.ckpt_name='epoch_160'
CUDA_VISIBLE_DEVICES=2 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_freezeobsFalse_finetune_w_human_onlyFalse_useonlyoriginalFalse_fromscratchFalse_after_dagger' task_name='TurnOnSinkFaucet' dagger.ckpt_name='epoch_180'
CUDA_VISIBLE_DEVICES=2 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_freezeobsFalse_finetune_w_human_onlyFalse_useonlyoriginalFalse_fromscratchFalse_after_dagger' task_name='TurnOnSinkFaucet' dagger.ckpt_name='epoch_200'






CUDA_VISIBLE_DEVICES=2 python evaluate.py --config-name=evaluate_lora_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_lora_rank256_scale0.2_freezeobsFalse_loraonobsFalse_after_dagger' task_name='TurnOnSinkFaucet' dagger.ckpt_name='epoch_120'
CUDA_VISIBLE_DEVICES=2 python evaluate.py --config-name=evaluate_lora_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_lora_rank256_scale0.2_freezeobsFalse_loraonobsFalse_after_dagger' task_name='TurnOnSinkFaucet' dagger.ckpt_name='epoch_140'
CUDA_VISIBLE_DEVICES=2 python evaluate.py --config-name=evaluate_lora_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_lora_rank256_scale0.2_freezeobsFalse_loraonobsFalse_after_dagger' task_name='TurnOnSinkFaucet' dagger.ckpt_name='epoch_160'
CUDA_VISIBLE_DEVICES=2 python evaluate.py --config-name=evaluate_lora_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_lora_rank256_scale0.2_freezeobsFalse_loraonobsFalse_after_dagger' task_name='TurnOnSinkFaucet' dagger.ckpt_name='epoch_180'
CUDA_VISIBLE_DEVICES=2 python evaluate.py --config-name=evaluate_lora_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_lora_rank256_scale0.2_freezeobsFalse_loraonobsFalse_after_dagger' task_name='TurnOnSinkFaucet' dagger.ckpt_name='epoch_200'



CUDA_VISIBLE_DEVICES=2 python evaluate.py --config-name=evaluate_lora_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_lora_rank256_scale0.2_freezeobsTrue_loraonobsFalse_after_dagger' task_name='TurnOnSinkFaucet' dagger.ckpt_name='epoch_120'
CUDA_VISIBLE_DEVICES=2 python evaluate.py --config-name=evaluate_lora_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_lora_rank256_scale0.2_freezeobsTrue_loraonobsFalse_after_dagger' task_name='TurnOnSinkFaucet' dagger.ckpt_name='epoch_140'
CUDA_VISIBLE_DEVICES=2 python evaluate.py --config-name=evaluate_lora_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_lora_rank256_scale0.2_freezeobsTrue_loraonobsFalse_after_dagger' task_name='TurnOnSinkFaucet' dagger.ckpt_name='epoch_160'
CUDA_VISIBLE_DEVICES=2 python evaluate.py --config-name=evaluate_lora_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_lora_rank256_scale0.2_freezeobsTrue_loraonobsFalse_after_dagger' task_name='TurnOnSinkFaucet' dagger.ckpt_name='epoch_180'
CUDA_VISIBLE_DEVICES=2 python evaluate.py --config-name=evaluate_lora_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_lora_rank256_scale0.2_freezeobsTrue_loraonobsFalse_after_dagger' task_name='TurnOnSinkFaucet' dagger.ckpt_name='epoch_200'

