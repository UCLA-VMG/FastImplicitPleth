import os
from tqdm import tqdm
import numpy as np
import commentjson as json

import torch
import torch.utils.data
import tinycudann as tcnn
import argparse

from ..models.combinations import MotionNet
from ..data.datasets import VideoGridDataset
from ..utils.utils import trace_video, Dict2Class

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Siren with Hash Grid Encodings.')
    parser.add_argument('-vp', '--video_path', required=True, type=str, help='Path to the video.')
    parser.add_argument('-config', '--config_path', required=True, type=str, help='Path to the config file.')
    parser.add_argument('--verbose', action='store_true', help='Verbosity.')
    parser.add_argument('--prepend_save_path', default=None, type=str, help='Prepend the save paths for dataset automation.')
    parser.add_argument('--prepend_load_path', default=None, type=str, help='Prepend the load paths for dataset automation.')

    return parser.parse_args()

class PlethModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = torch.device("cpu")
    
    def forward(self, x):
        # All Custom models in the repo return 2 values
        return self.model(x.to(self.device)), None

    def set_device(self, device):
        self.device = device
        self.model.to(self.device)

class AdditionEnsemble(torch.nn.Module):
    def __init__(self, modelA, modelB):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.device = torch.device("cpu")
        
    def forward(self, x):
        a_out = self.modelA(x)
        if type(a_out) == tuple:
            a_out = a_out[0].to(self.device)
        b_out = self.modelB(x)
        if type(b_out) == tuple:
            b_out = b_out[0].to(self.device)
        return a_out + b_out
    
    def set_device(self, device):
        self.device = device

def main(args):
    dset = VideoGridDataset(args.video_path, verbose=args.verbose, 
                            num_frames=args.data["num_frames"], start_frame=args.data["start_frame"])
    dloader = torch.utils.data.DataLoader(dset, batch_size=args.data["batch_size"], shuffle=True)
    trace_loader = torch.utils.data.DataLoader(dset, batch_size=args.data["trace_batch_size"], shuffle=False)
    # Model 1: Capture Motion 
    with open(args.motion_model["config"]) as mmf:
        args.motion_model["config"] = json.load(mmf)
    motion_model = MotionNet(args.motion_model["config"]["spatiotemporal_encoding"], 
                             args.motion_model["config"]["spatiotemporal_network"],
                             args.motion_model["config"]["deltaspatial_encoding"], 
                             args.motion_model["config"]["deltaspatial_network"])
    # Load pre-trained weights
    if args.prepend_load_path is not None:
        args.motion_model["load_path"] = args.prepend_load_path + args.motion_model["load_path"]
    if args.verbose: print(f'Loading motion model from {args.motion_model["load_path"]}')
    motion_model.load_state_dict(torch.load(args.motion_model["load_path"])["model_state_dict"])
    # Freeze the model
    motion_model.eval()
    for params in motion_model.parameters():
        params.requires_grad = False
    # Set the model device
    motion_spatiotemporal_device = torch.device(args.spatiotemporal_device)
    motion_deltaspatial_device = torch.device(args.deltaspatial_device)
    motion_model.set_device(motion_spatiotemporal_device, motion_deltaspatial_device)
    # Model 2: Capture PPG
    pleth_model = PlethModelWrapper(tcnn.NetworkWithInputEncoding(args.pleth_encoding["input_dims"],
                                                                  args.pleth_network["output_dims"],
                                                                  args.pleth_encoding, 
                                                                  args.pleth_network))
    # To Device
    pleth_model.set_device(args.pleth_device)
    # Ensemble Adder
    # Trace takes in only a single model. Hence required.
    ensemble = AdditionEnsemble(motion_model, pleth_model)
    ensemble.set_device(args.io_device) # NOTE: This only stores the device does not move the models.
    # Print the models
    if args.verbose: print('-'*100, flush=True)
    if args.verbose: print(ensemble)
    if args.verbose: print('-'*100, flush=True)

    # Optimizer.
    # NOTE: When adding weight decay, make sure to split the Network and the Hash Encoding.
    #       The Motion model would need to be altered to separate the network and encoding models.
    #       Individual optimizers are needed - network opt (uses decay) and encoding opt (doesn't use decay).
    #       Compatibility has been ensured and this alteration will not break the code.
    #       Furthermore, 2 optimizers would be needed, one for the encodings and one for the network.
    opt = torch.optim.Adam(pleth_model.parameters(), lr=args.opt["lr"],
                           betas=(args.opt["beta1"], args.opt["beta2"]), eps=args.opt["eps"])
    # Folders for the traced video data and checkpoints.
    if args.prepend_save_path is not None:
        args.trace["folder"] = args.prepend_save_path + args.trace["folder"]
        args.checkpoints["dir"] = args.prepend_save_path + args.checkpoints["dir"]
    if args.trace["folder"] is not None:
        os.makedirs(args.trace["folder"], exist_ok=True)
        if args.verbose: print(f'Saving trace to {args.trace["folder"]}')
    if args.checkpoints["save"]:
        os.makedirs(args.checkpoints["dir"], exist_ok=True)
        if args.verbose: print(f'Saving checkpoints to {args.checkpoints["dir"]}')

    # Get info before iterating.
    epochs = args.train["epochs"]
    ndigits_epoch = int(np.log10(epochs)+1)
    latest_ckpt_path = os.path.join(args.checkpoints["dir"], args.checkpoints["latest"])
    if os.path.exists(latest_ckpt_path):
        if args.verbose: print('Loading latest checkpoint...')
        saved_dict = torch.load(latest_ckpt_path)
        pleth_model.load_state_dict(saved_dict["model_state_dict"])
        if "optimizer_state_dict" in saved_dict.keys():
            opt.load_state_dict(saved_dict["optimizer_state_dict"])
        start_epoch = saved_dict["epoch"] + 1
        if args.verbose: print(f'Continuing from epoch {start_epoch}.')
    else:
        if args.verbose: print('Start from scratch.')
        start_epoch = 1
    # Epoch iteration.
    for epoch in range(1,epochs+1):
        train_loss = 0
        pleth_model.train()
        for count, item in tqdm(enumerate(dloader),total=len(dloader)):
            loc = item["loc"].half()
            pixel = item["pixel"].half()/args.data["norm_value"]
            # Ensemble is not used since we want to use torch.np_grad()
            with torch.no_grad():
                motion_out, _ = motion_model(loc)
            ppg_res_out, _ = pleth_model(loc)
            output = motion_out.to(args.io_device) + ppg_res_out.to(args.io_device)
            # Since the model takes care of moving the data to different devices, move GT correspondingly.
            pixel = pixel.to(output.dtype)
            pixel = pixel.to(output.device)
            # Backpropagation.
            opt.zero_grad()
            l2_error = (output.to(args.io_device) - pixel)**2
            loss = l2_error.mean()
            loss.backward()
            opt.step()
        train_loss += loss.item()
        print(f'Epoch: {epoch}, Loss: {train_loss/len(dloader)}', flush=True)
        # Trace the video data and save/plot the video/frames.
        if epoch >= args.trace["trace_epoch"]:
            trace_video(ensemble, dset, trace_loader, args.io_device, \
                        save_dir=args.trace["folder"], \
                        save_file=f'ensemble_{args.trace["file_tag"]}{str(epoch).zfill(ndigits_epoch)}', \
                        save_ext=args.trace["ext"],\
                        plot=args.trace["plot"], verbose=args.verbose)
            trace_video(motion_model, dset, trace_loader, args.io_device, \
                        save_dir=args.trace["folder"], \
                        save_file=f'motion_{args.trace["file_tag"]}{str(epoch).zfill(ndigits_epoch)}', \
                        save_ext=args.trace["ext"],\
                        plot=args.trace["plot"], verbose=args.verbose)
        # Save the checkpoints
        if args.checkpoints["save"]:
            if args.verbose: print('Saving checkpoint.')
            if epoch % args.checkpoints["epoch_frequency"] == 0:
                checkpoint_file = f'{args.checkpoints["file_tag"]}{str(epoch).zfill(ndigits_epoch)}{args.checkpoints["ext"]}'
                # Save as dict to maintain uniformity
                torch.save({'model_state_dict': pleth_model.state_dict()}, 
                           os.path.join(args.checkpoints["dir"], checkpoint_file))
                if args.verbose: print(f'Saved model for epoch {epoch}.')
            torch.save({
                'epoch': epoch,
                'model_state_dict': pleth_model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                }, latest_ckpt_path)
            if args.verbose: print('Saved latest checkpoint.')
        # Epoch demarcation
        print('-'*100, flush=True)

if __name__ == '__main__':
    # Parse the command line arguments.
    args = parse_arguments()
    # Read the config file.
    with open(args.config_path) as f:
        json_config = json.load(f)
    # Convert the command line args to a dict format.
    dict_args = vars(args)
    # Update the args dict with the config dict.
    dict_args.update(json_config)
    # Convert back to a class (dot notation).
    args = Dict2Class(dict_args)
    if args.verbose: print('-'*100, flush=True)
    if args.verbose: print(dict_args)
    if args.verbose: print('-'*100, flush=True)
    # Convert device string to torch.device().
    args.pleth_device = torch.device(args.pleth_device)
    args.io_device = torch.device(args.io_device)
    # Main routine.
    main(args)