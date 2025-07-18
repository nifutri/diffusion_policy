import os
import shutil
import time
import copy

home_path = os.path.expanduser("~")

dp_directory = os.path.join(home_path, "Code_robocasa", "robocasa_dagger", "diffusion_policy")
robocasa_directory = os.path.join(home_path, "Code_robocasa", "robocasa_dagger", "robocasa")

# now define some hyperaprameters:

initial_failure_file = f'{home_path}/Code_robocasa/robocasa_dagger/diffusion_policy/data/outputs/ST_OOD_DAgger_train_diffusion_unet_clip_OpenSingleDoor/eval_base_policy/merged_demos_fails_im256.hdf5'
# initial_failure_file = None

initialization = True
num_epochs = 50
# proposal_policy_path = '/home/niklasfunk/Code_robocasa/robocasa_dagger/diffusion_policy/data/outputs/ST_OOD_DAgger_train_diffusion_unet_clip_OpenSingleDoor/base_policy_2025.07.16-14.28.53/checkpoints/latest.ckpt'
proposal_policy_path = f'{home_path}/Code_robocasa/robocasa_dagger/diffusion_policy/data/outputs/PreTrainedModels/OpenSingleDoor/latest.ckpt'
# for initial policy configuration the bad policy is equal to the proposal policy
bad_policy_path = proposal_policy_path


num_epochs_iterative_procedure = 50
num_rollouts = 10
cuda_device = 0
pos_cropping = 0.1
rot_cropping = 0.1
num_epochs = 40 # 25
num_epochs_post_init = 10
lr_scheduler_epochs = 250
batch_size = 48
num_workers = 24

# make a folder with date and time in the name:
curr_experiment_name = f"blending_policy_experiment_{time.strftime('%Y.%m.%d-%H.%M.%S')}"
# curr_experiment_name = "blending_policy_experiment_2025.07.17-20.43.18"
output_dir = os.path.join(dp_directory, "data", "outputs", curr_experiment_name)
if os.path.exists(output_dir):
    print(f"Output directory {output_dir} already exists. Removing it.")
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

for i in range(num_epochs_iterative_procedure):
    # # first epoch -> create directory
    curr_folder_tmp = os.path.join(output_dir, f"epoch_{i+1}")
    os.makedirs(curr_folder_tmp, exist_ok=True)
    # Step 1: evaluate the current policy:
    cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} python evaluate.py --config-name=eval_robocasa_base_policy_blending proposal_policy_ckpt_path={proposal_policy_path} bad_policy_ckpt_path={bad_policy_path} blending_policy.pos_cropping={pos_cropping} blending_policy.rot_cropping={rot_cropping} num_rollouts={num_rollouts}"
    if not initialization:
        cmd += f" blending_policy.greedy_behavior=1"
    else:
        cmd += f" blending_policy.greedy_behavior=0"
    # finally append the output directory location
    cmd += f" hydra.run.dir={curr_folder_tmp}"

    print(f"Running command: {cmd}")
    os.system(cmd)

    # Step 2: create the full data, i.e., the image data
    cmd = f"cd {robocasa_directory}/robocasa/scripts && python original_robocasa_dataset_states_to_obs.py --dataset {curr_folder_tmp}/merged_demos_fails.hdf5"
    os.system(cmd)

    # now merge the datasets:
    if not initialization:
        if initial_failure_file is not None:
            # merge the files:
            cmd = f"python merge_dagger_datasets_simple.py --dataset1 {initial_failure_file} --dataset2 {curr_folder_tmp}/merged_demos_fails_im256.hdf5 --out_dataset {curr_folder_tmp}/merged_demos_fails_im256_complete.hdf5"
            os.system(cmd)
            initial_failure_file = f"{curr_folder_tmp}/merged_demos_fails_im256_complete.hdf5"
        else:
            initial_failure_file = f"{curr_folder_tmp}/merged_demos_fails_im256.hdf5"
    else:
        cmd = f"python merge_dagger_datasets_simple.py --dataset1 {initial_failure_file} --dataset2 {curr_folder_tmp}/merged_demos_fails_im256.hdf5 --out_dataset {curr_folder_tmp}/merged_demos_fails_im256_complete.hdf5"
        os.system(cmd)
        initial_failure_file = f"{curr_folder_tmp}/merged_demos_fails_im256_complete.hdf5"

    # now actually run the training:
    cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} python train.py --config-dir=. --config-name=train_robocasa_base_dp_clip_policy.yaml training.seed=42 task.name='OpenSingleDoor' task.dataset.human_path={initial_failure_file} training.num_epochs={num_epochs} hydra.run.dir={curr_folder_tmp}  dataloader.batch_size={batch_size} dataloader.num_workers={num_workers}"
    if initialization:
        cmd += f" training.lr_scheduler_epochs={lr_scheduler_epochs} training.restore_accelerator=False"
    if not initialization:
        cmd += f" training.lr_scheduler_epochs={lr_scheduler_epochs} training.restore_accelerator=True training.previous_accelerator_path={previous_accelerator_path}"

    os.system(cmd)

    # upon termination update the bad policy path
    bad_policy_path = f"{curr_folder_tmp}/checkpoints/epoch_{(num_epochs-1)}.ckpt"
    previous_accelerator_path = f"{curr_folder_tmp}/checkpoints/epoch_{(num_epochs-1)}_accelerator"

    # remove all unneeded checkpoints except the last one
    for file in os.listdir(f"{curr_folder_tmp}/checkpoints/"):
        if file.startswith("epoch_"):
            if not file.startswith(f"epoch_{(num_epochs-1)}"):
                # remove it no matter if it is a file or directory
                if (os.path.isdir(os.path.join(curr_folder_tmp, "checkpoints", file))):
                    shutil.rmtree(os.path.join(curr_folder_tmp, "checkpoints", file))
                else:
                    os.remove(os.path.join(curr_folder_tmp, "checkpoints", file))


    if initialization:
        num_epochs = num_epochs_post_init
        initialization = False



