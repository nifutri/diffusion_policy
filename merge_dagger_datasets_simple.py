import h5py
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
import scipy.stats
import os


def merge_h5_files_exact_dagger(file1_path, dagger_file2_path, output_path):
    with h5py.File(file1_path, 'r') as f1, \
        h5py.File(dagger_file2_path, 'r') as f2, \
        h5py.File(output_path, 'w') as out:

        # Copy all data from file1 as-is
        out.create_group('data')
        for key in f1['data'].keys():
            f1.copy(f'data/{key}', out['data'], name=key)
        print(f"Copied {len(f1['data'])} demos from {file1_path}")

        # Determine the starting index for file2 demos
        start_idx = len(f1['data'])

        for key in f2['data'].keys():
            new_key = f'demo_{start_idx}'
            f2.copy(f'data/{key}', out['data'], name=new_key)
            start_idx += 1
        print(f"Copied {len(f2['data'])} demos from {dagger_file2_path} with renamed keys")

    print(f"\n✅ Merged files saved to: {output_path}")



def main():
    parser = argparse.ArgumentParser(description="Merge dagger data for a specific task.")
    parser.add_argument("--dataset1", type=str, default='train', required=True)
    parser.add_argument("--dataset2", type=str, default='dagger_episode_0', required=True)
    parser.add_argument("--out_dataset", type=str, default='dagger_episode_0', required=True)

    args = parser.parse_args()

    dataset1 = args.dataset1
    dataset2 = args.dataset2
    out_dataset = args.out_dataset

    merge_h5_files_exact_dagger(dataset1, dataset2, out_dataset)
    print (f"✅ Merged {dataset1} and {dataset2} into {out_dataset}")



if __name__ == "__main__":
    main()