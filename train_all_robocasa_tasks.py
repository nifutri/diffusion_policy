import os
# OMP_NUM_THREADS=1 MPI_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python ../robocasa/robocasa/scripts/dataset_states_to_obs.py --dataset datasets/v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30/demo.hdf5


# data_dir = 'datasets/v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30/demo.hdf5'
dataset_dir = '../robocasa/datasets_first/v0.1/'
# list stages
stages = os.listdir(dataset_dir)
print(f'Found {len(stages)} stages in {dataset_dir}:', stages)
# for all stages, get kitchen domain
for stage in stages:
    
    # list kitchen domains
    kitchen_domains = os.listdir(os.path.join(dataset_dir, stage))
    print(f'Found {len(kitchen_domains)} kitchen domains in stage {stage}:', kitchen_domains)
    for kitchen_domain in kitchen_domains:
        # if 'kitchen' not in stage:
        #     continue
        print(f'Processing stage: {stage}, kitchen domain: {kitchen_domain}')
        # list tasks
        tasks = os.listdir(os.path.join(dataset_dir, stage, kitchen_domain))
        print(f'Found {len(tasks)} tasks in {kitchen_domain}:', tasks)
        for task in tasks:
            print("running task:", task)
            # if 'CloseDrawer' in task:
            #     continue
            # list dates
            dates = os.listdir(os.path.join(dataset_dir, stage, kitchen_domain, task))
            print(f'Found {len(dates)} demos in task {task}:', dates)
            for date in dates:
                print(f'Processing date: {date}')
                # list demos
                demos = os.listdir(os.path.join(dataset_dir, stage, kitchen_domain, task, date))
                print(f'Found {len(demos)} demos in date {date}:', demos)
                # convert each demo to im256
            
                for demo in demos:
                    if 'demo_gentex_im128_randcams_im256.hdf5' not in demo or 'zip' in demo:
                        continue
                    # convert to im256
                    dataset_converted_path = os.path.join(dataset_dir, stage, kitchen_domain, task, date, demo)
                    # demo_im256.hdf5 exists
                    # dataset_converted_path = data_path.replace('.hdf5', '_im256.hdf5')
                    print(f'TRAINING ON {dataset_converted_path} w im256...')
                    save_dir = 'data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}'
                    command = f'CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python train.py --config-dir=. --config-name=replicate_jb_robocasa.yaml training.seed=42 training.device=cuda:0 hydra.run.dir=\'{save_dir}_{kitchen_domain}_{task}\' task.name=train_dp_{task}'
                    
                    command += f' task.dataset.dataset_path=\'{dataset_converted_path}\' task.dataset_path=\'{dataset_converted_path}\' task.env_runner.dataset_path=\'{dataset_converted_path}\''
                    print(command)
                    output = os.system(command)
                    if output != 0:
                        print("error training task:", task, "with dataset:", dataset_converted_path)
                    else:
                        print("successfully trained task:", task, "with dataset:", dataset_converted_path)

