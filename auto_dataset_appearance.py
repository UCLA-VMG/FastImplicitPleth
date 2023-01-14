import os
import pickle
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--root', required=True, type=str)
parser.add_argument('--pkl', required=True, type=str)
parser.add_argument('--cuda-dev', required=True, type=str)
args = parser.parse_args()

# ROOT_PATH = args.root
ROOT_PATH = '/home/pradyumnachari/Documents/ImplicitPPG/SIGGRAPH_Data/rgb_files'

# '/home/pradyumnachari/Documents/ImplicitPPG/SIGGRAPH_Data/fl_l1.pkl'
with open(args.pkl,'rb') as f:
    list_of_folders = pickle.load(f)

for folder in list_of_folders:
    # Path to the video
    filename = 'implicitpleth.scripts.train_cascaded_hash_appearance'
    vp = os.path.join(ROOT_PATH, folder)
    # Start frame 0
    config = 'configs/dataset/ch_appearance_0.json'
    log = f'logs/{folder}_appearance_log_0.txt'
    os.system(f'CUDA_VISIBLE_DEVICES={args.cuda_dev} nohup python -m {filename} -vp {vp} -config {config} --append_save_path _{folder} --verbose > {log} ')
    
    
    # Start frame 300
    config = 'configs/dataset/ch_appearance_300.json'
    log = f'logs/{folder}_appearance_log_300.txt'
    os.system(f'CUDA_VISIBLE_DEVICES={args.cuda_dev} nohup python -m {filename} -vp {vp} -config {config} --append_save_path _{folder} --verbose > {log} ')
    
    
    # Start frame 600
    config = 'configs/dataset/ch_appearance_600.json'
    log = f'logs/{folder}_appearance_log_600.txt'
    os.system(f'CUDA_VISIBLE_DEVICES={args.cuda_dev} nohup python -m {filename} -vp {vp} -config {config} --append_save_path _{folder} --verbose > {log} ')