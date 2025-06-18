# import subprocess
# import os
# import re

# # Constants
# AWS_PROFILE = "manip-cluster"
# BASE_PATH = "s3://robotics-manip-lbm/efs/data/tasks/BimanualBikeRotorInstall/ruggles/real/bc/rollout/"
# TARGET_DATE = "2025-06-16"
# LOCAL_SAVE_DIR = "bikerotor_faildetect"

# # Create local directory if it doesn't exist
# os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)

# # Step 1: List all directories in the rollout folder
# def list_s3_objects(path):
#     cmd = f"AWS_PROFILE={AWS_PROFILE} aws s3 ls {path}"
#     result = subprocess.check_output(cmd, shell=True).decode("utf-8")
#     return result.splitlines()

# # Step 2: Filter directories from the target date
# all_lines = list_s3_objects(BASE_PATH)
# target_dirs = [line.split()[-1] for line in all_lines if TARGET_DATE in line]

# # Step 3: For each matching directory, list episodes and download actions.npz
# for date_dir in target_dirs:
#     full_date_path = BASE_PATH + date_dir
#     diffusion_path = full_date_path + "diffusion_spartan/"

#     try:
#         episode_lines = list_s3_objects(diffusion_path)
#         episode_dirs = [line.split()[-1] for line in episode_lines if "episode_" in line]

#         for episode in episode_dirs:
#             actions_path = diffusion_path + f"{episode}processed/actions.npz"
#             local_filename = f"{TARGET_DATE}_{episode.strip('/').replace('episode_', 'ep')}_actions.npz"
#             local_path = os.path.join(LOCAL_SAVE_DIR, local_filename)

#             download_cmd = f"AWS_PROFILE={AWS_PROFILE} aws s3 cp {actions_path} {local_path}"
#             print(f"Downloading {actions_path} to {local_path}")
#             subprocess.run(download_cmd, shell=True, check=True)

#     except subprocess.CalledProcessError as e:
#         print(f"Failed to process {date_dir}: {e}")

# print("Download complete.")

import os
import subprocess
import pdb
# Constants
AWS_PROFILE = "manip-cluster"
S3_ROLLOUT_PATH = "s3://robotics-manip-lbm/efs/data/tasks/BimanualBikeRotorInstall/ruggles/real/bc/rollout/"
TARGET_DATE = "2025-06-16"
LOCAL_SAVE_DIR = "bikerotor_faildetect"

# Ensure save directory exists
os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)

def list_s3_dirs(base_path):
    """List subdirectories under the base rollout path."""
    cmd = f"AWS_PROFILE={AWS_PROFILE} aws s3 ls {base_path}"
    try:
        output = subprocess.check_output(cmd, shell=True).decode("utf-8")
        return [line.split()[-1].strip("/") for line in output.strip().splitlines()]
    except subprocess.CalledProcessError as e:
        print(f"Failed to list base path: {e}")
        return []

def list_s3_files_recursively(path):
    """List all files recursively under a date folder."""
    cmd = f"AWS_PROFILE={AWS_PROFILE} aws s3 ls {path} --recursive"
    try:
        output = subprocess.check_output(cmd, shell=True).decode("utf-8")
        return [line.split()[-1] for line in output.strip().splitlines()]
    except subprocess.CalledProcessError as e:
        print(f"Failed to list files under {path}: {e}")
        return []

def download_file(s3_path, local_path):
    """Download a single file from S3."""
    cmd = f"AWS_PROFILE={AWS_PROFILE} aws s3 cp {s3_path} {local_path}"
    try:
        print(f"Downloading {s3_path} -> {local_path}")
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {s3_path}: {e}")

# Step 1: Find folders for the target date
folders = list_s3_dirs(S3_ROLLOUT_PATH)
date_folders = [f for f in folders if TARGET_DATE in f]

# Step 2: Search and download rubric JSONs from each
for folder in date_folders:
    full_s3_prefix = S3_ROLLOUT_PATH + folder + "/"
    all_files = list_s3_files_recursively(full_s3_prefix)
    # pdb.set_trace()

    for key in all_files:
        if "rubric" in key and "json" in key:
            command = f"AWS_PROFILE={AWS_PROFILE} aws s3 cp {S3_ROLLOUT_PATH + key.split('efs/data/tasks/BimanualBikeRotorInstall/ruggles/real/bc/rollout/')[1]} {LOCAL_SAVE_DIR}"
            print("command", command)
            # run command
            os.system(command)
            # s3_full_path = full_s3_prefix + key
            # local_name = os.path.basename(key)
            # local_path = os.path.join(LOCAL_SAVE_DIR, local_name)
            # download_file(key, local_path)

print("\n✅ All rubric JSONs for 2025-06-16 downloaded.")


