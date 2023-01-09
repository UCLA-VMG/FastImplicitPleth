import os
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='/home/pradyumnachari/Documents/ImplicitPPG/SIGGRAPH_Data/rgb_files', type=str)
parser.add_argument('--pkl', required=True, type=str)
parser.add_argument('--cuda-dev', required=True, type=str)
parser.add_argument('--config-folder', default='configs/dataset', type=str)
args = parser.parse_args()

# ROOT_PATH = args.root
ROOT_PATH = args.root

# '/home/pradyumnachari/Documents/ImplicitPPG/SIGGRAPH_Data/fl_l1.pkl'
with open(args.pkl,'rb') as f:
    list_of_folders = pickle.load(f)

for folder in list_of_folders:
    # Path to the video
    filename = 'implicitpleth.scripts.train_double_hash_delta_motion'
    vp = os.path.join(ROOT_PATH, folder)
    # Start frame 0
    config = os.path.join(args.config_folder, f'dhd_motion_0.json')
    if os.path.exists(config):
        log = f'logs/{folder}_motion_log_0.txt'
        os.system(f'CUDA_VISIBLE_DEVICES={args.cuda_dev} nohup python -m {filename} -vp {vp} -config {config} --append_save_path _{folder} --verbose > {log} ')
    else:
        print(f'{config} does not exist')
    
    
    # Start frame 300
    config = os.path.join(args.config_folder, f'dhd_motion_300.json')
    if os.path.exists(config):
        log = f'logs/{folder}_motion_log_300.txt'
        os.system(f'CUDA_VISIBLE_DEVICES={args.cuda_dev} nohup python -m {filename} -vp {vp} -config {config} --append_save_path _{folder} --verbose > {log} ')
    else:
        print(f'{config} does not exist')
    
    
    # Start frame 600
    config = os.path.join(args.config_folder, f'dhd_motion_600.json')
    if os.path.exists(config):
        log = f'logs/{folder}_motion_log_600.txt'
        os.system(f'CUDA_VISIBLE_DEVICES={args.cuda_dev} nohup python -m {filename} -vp {vp} -config {config} --append_save_path _{folder} --verbose > {log} ')
    else:
        print(f'{config} does not exist')

    print(vp)
    print("-"*50)