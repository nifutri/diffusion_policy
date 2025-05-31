import os
import pdb
from pathlib import Path
import os
folder_path = Path('~/efs')
command = 'AWS_PROFILE=sagemaker aws s3 ls s3://robotics-manip-lbm/efs/data/tasks/AllLbmBimanual/'
# output = os.system(command)
# print(output)


import subprocess

try:
    result = subprocess.check_output(command, shell=True, text=True)
    output_lines = result.strip().split('\n')
    print("Command output as list:")
    print(output_lines)
    for elem in output_lines:
        # if 'Bimanual' in elem:
        pdb.set_trace()
except subprocess.CalledProcessError as e:
    print(f"Command failed with error: {e}")