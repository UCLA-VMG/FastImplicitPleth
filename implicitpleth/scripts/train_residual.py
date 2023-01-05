import os
from tqdm import tqdm
import numpy as np
import commentjson as json
import imageio.v2 as iio2

import torch
import torch.utils.data
import tinycudann as tcnn
import argparse

from ..models.combinations import MotionNet
from ..data.datasets import VideoGridDataset
from ..utils.utils import Dict2Class

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Siren with Hash Grid Encodings.')
    parser.add_argument('-vp', '--video_path', required=True, type=str, help='Path to the video.')
    parser.add_argument('-config', '--config_path', required=True, type=str, help='Path to the config file.')
    parser.add_argument('--verbose', action='store_true', help='Verbosity.')
    parser.add_argument('--append_save_path', default=None, type=str, help='Prepend the save paths for dataset automation.')
    parser.add_argument('--append_load_path', default=None, type=str, help='Prepend the load paths for dataset automation.')

    return parser.parse_args()

def main(args):
    dset = VideoGridDataset(args.video_path, verbose=args.verbose, num_frames=args.data["num_frames"], 
                        start_frame=args.data["start_frame"], pixel_norm=args.data["norm_value"])
    dloader = torch.utils.data.DataLoader(range(len(dset)), 
                                          batch_size=args.data["batch_size"],
                                          shuffle=True, num_workers=1)
    # Pre-trained motion model
    print(args.motion_model)
    with open(args.motion_model["config"]) as mmf:
        config = json.load(mmf)
    motion_model = MotionNet(config["spatiotemporal_encoding"], 
                            config["spatiotemporal_network"],
                            config["deltaspatial_encoding"], 
                            config["deltaspatial_network"])
    if args.append_load_path is not None:
        args.motion_model["load_path"] = args.motion_model["load_path"] + args.append_load_path
    if args.verbose: print(f'Loading motion model from {args.motion_model["load_path"]}')
    motion_model.load_state_dict(torch.load(args.motion_model["load_path"])["model_state_dict"])
    motion_model.eval()
    motion_model.set_device(args.spatiotemporal_device, args.deltaspatial_device)
    # Pleth Residual Model
    pleth_enc = tcnn.Encoding(args.pleth_encoding["input_dims"], args.pleth_encoding)
    pleth_net = tcnn.Network(pleth_enc.n_output_dims, args.pleth_network["output_dims"], args.pleth_network)
    pleth_model = torch.nn.Sequential(pleth_enc, pleth_net)
    pleth_model.to(args.pleth_device)
    # Print the models
    if args.verbose: print('-'*100, flush=True)
    if args.verbose: print(pleth_model)
    if args.verbose: print('-'*100, flush=True)

    # Optimizer.
    opt_enc = torch.optim.Adam(pleth_enc.parameters(), lr=1e-3,
                       betas=(args.opt["beta1"], args.opt["beta2"]), eps=args.opt["eps"])
    opt_net = torch.optim.Adam(pleth_net.parameters(), lr=1e-3, weight_decay=args.opt["l2_reg"],
                        betas=(args.opt["beta1"], args.opt["beta2"]), eps=args.opt["eps"])
    # Folders for the traced video data and checkpoints.
    if args.append_save_path is not None:
        args.trace["folder"] =  args.trace["folder"] + args.append_save_path
        args.checkpoints["dir"] =  args.checkpoints["dir"] + args.append_save_path
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
        # TODO: Add code to load latest checkpoint and optimizer
        # start_epoch = saved_dict["epoch"] + 1
        # if args.verbose: print(f'Continuing from epoch {start_epoch}.')
    else:
        if args.verbose: print('Starting from scratch.')
        start_epoch = 1
    # Epoch iteration.
    for epoch in range(start_epoch,epochs+1):
        train_loss = 0
        pleth_model.train()
        motion_model.train()
        for count, item in enumerate(dloader):
            loc = dset.loc[item].half().to(args.pleth_device)
            pixel = dset.vid[item].half().to(args.pleth_device)
            motion_output, _ = motion_model(loc)
            pleth_output = pleth_model(loc)
            output = motion_output + pleth_output
            # Since the model takes care of moving the data to different devices, move GT correspondingly.
            pixel = pixel.to(output.dtype).to(output.device)
            # Backpropagation.
            opt_enc.zero_grad()
            opt_net.zero_grad()
            l2_error = (output - pixel)**2
            loss = l2_error.mean()
            loss.backward()
            opt_enc.step()
            opt_net.step()
            train_loss += loss.item()
        print(f'Epoch: {epoch}, Loss: {train_loss/len(dloader)}', flush=True)
        # Trace the video data and save/plot the video/frames.
        if args.trace["folder"] is not None:
            if epoch >= args.trace["trace_epoch"]:
                with torch.no_grad():
                    motion_model.eval()
                    pleth_model.eval()
                    trace_loc = dset.loc.half().to(args.pleth_device)
                    motion_output, _ = motion_model(trace_loc)
                    pleth_output = pleth_model(trace_loc)
                    
                    # TODO: Reduce the number of lines
                    trace = motion_output + pleth_output
                    trace = trace.detach().cpu().float().reshape(dset.shape).permute(2,0,1,3).numpy()
                    trace = (np.clip(trace, 0, 1)*255).astype(np.uint8)
                    save_path = os.path.join(args.trace["folder"], f'{args.trace["file_tag"]}{str(epoch).zfill(ndigits_epoch)}.avi')
                    iio2.mimwrite(save_path, trace, fps=30)
                    
                    trace = pleth_output.detach().cpu().float().reshape(dset.shape).permute(2,0,1,3).numpy()
                    trace = (trace - np.amin(trace, axis=(0,1,2), keepdims=True)) / (np.amax(trace, axis=(0,1,2), keepdims=True) - np.amin(trace, axis=(0,1,2), keepdims=True))
                    trace = (np.clip(trace, 0, 1)*255).astype(np.uint8)
                    save_path = os.path.join(args.trace["folder"], f'rescaled_residual_{args.trace["file_tag"]}{str(epoch).zfill(ndigits_epoch)}.avi')
                    iio2.mimwrite(save_path, trace, fps=30)
                    
                    trace = motion_output.detach().cpu().float().reshape(dset.shape).permute(2,0,1,3).numpy()
                    trace = (np.clip(trace, 0, 1)*255).astype(np.uint8)
                    save_path = os.path.join(args.trace["folder"], f'motion_{args.trace["file_tag"]}{str(epoch).zfill(ndigits_epoch)}.avi')
                    iio2.mimwrite(save_path, trace, fps=30)
        # Save the checkpoints
        if args.checkpoints["save"]:
            if args.verbose: print('Saving checkpoint.')
            if epoch % args.checkpoints["epoch_frequency"] == 0:
                checkpoint_file = f'{args.checkpoints["file_tag"]}{str(epoch).zfill(ndigits_epoch)}{args.checkpoints["ext"]}'
                # Save as dict to maintain uniformity
                torch.save({'model_state_dict': pleth_model.state_dict()}, 
                           os.path.join(args.checkpoints["dir"], checkpoint_file))
                if args.verbose: print(f'Saved model for epoch {epoch}.')
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': pleth_model.state_dict(),
            #     # 'optimizer_enc_state_dict': opt_enc.state_dict(),
            #     # 'optimizer_net_state_dict': opt_net.state_dict(),
            #     }, latest_ckpt_path)
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
    args.spatiotemporal_device = torch.device(args.spatiotemporal_device)
    args.deltaspatial_device = torch.device(args.deltaspatial_device)
    args.pleth_device = torch.device(args.pleth_device)
    # Main routine.
    main(args)