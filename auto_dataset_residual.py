import os
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='/home/pradyumnachari/Documents/ImplicitPPG/SIGGRAPH_Data/rgb_files', type=str)
parser.add_argument('--pkl', required=True, type=str)
parser.add_argument('--cuda-dev', required=True, type=str)
args = parser.parse_args()

ROOT_PATH = args.root

# '/home/pradyumnachari/Documents/ImplicitPPG/SIGGRAPH_Data/fl_l1.pkl'
with open(args.pkl,'rb') as f:
    list_of_folders = pickle.load(f)

for folder in list_of_folders:
    # Path to the video
    filename = 'implicitpleth.scripts.train_residual'
    vp = os.path.join(ROOT_PATH, folder)
    # Start frame 0
    config = 'configs/dataset/residual_0.json'
    log = f'logs/{folder}_residual_log_0.txt'
    os.system(f'CUDA_VISIBLE_DEVICES={args.cuda_dev} nohup python -m {filename} -vp {vp} -config {config} --append_save_path _{folder}  --append_load_path _{folder}/epoch_10.pth --verbose > {log} ')
    
    
    # Start frame 300
    config = 'configs/dataset/residual_300.json'
    log = f'logs/{folder}_residual_log_300.txt'
    os.system(f'CUDA_VISIBLE_DEVICES={args.cuda_dev} nohup python -m {filename} -vp {vp} -config {config} --append_save_path _{folder}  --append_load_path _{folder}/epoch_10.pth --verbose > {log} ')
    
    
    # Start frame 600
    config = 'configs/dataset/residual_600.json'
    log = f'logs/{folder}_residual_log_600.txt'
    os.system(f'CUDA_VISIBLE_DEVICES={args.cuda_dev} nohup python -m {filename} -vp {vp} -config {config} --append_save_path _{folder}  --append_load_path _{folder}/epoch_10.pth --verbose > {log} ')
    
    print(vp)
    print("-"*50)