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
                    if 'demo.hdf5' not in demo:
                        continue
                    # convert to im256
                    data_path = os.path.join(dataset_dir, stage, kitchen_domain, task, date, demo)
                    # demo_im256.hdf5 exists
                    # if os.path.exists(data_path.replace('.hdf5', '_im256.hdf5')):
                    #     # if the size of the file is bigger than the original, skip it
                    #     if os.path.getsize(data_path.replace('.hdf5', '_im256.hdf5')) > os.path.getsize(data_path):
                    #         print(f'Skipping {data_path} as it is already converted and larger than the original.')
                    #         continue
                    #     else:
                    #         print(f"Overwriting existing im256 file for {data_path}.")
                    print(f'Converting {data_path} to im256...')
                    command = f'OMP_NUM_THREADS=1 MPI_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python ../robocasa/robocasa/scripts/dataset_states_to_obs.py --dataset {data_path}'
                    print(command)
                    output = os.system(command)
                    if output != 0:
                        print(f'Error converting {data_path} to im256.')
                    else:
                        print(f'Successfully converted {data_path} to im256.')

