import os
from tqdm import tqdm
import numpy as np
import commentjson as json
import imageio.v2 as iio2
import matplotlib.pyplot as plt

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

    return parser.parse_args()

def main(args):
    # Create the Dataset object and the two dataloaders - one for train and one for tracing.
    dset = VideoGridDataset(args.video_path, verbose=args.verbose, num_frames=args.data["num_frames"], 
                        start_frame=args.data["start_frame"], pixel_norm=args.data["norm_value"])
    dloader = torch.utils.data.DataLoader(range(len(dset)), batch_size=args.data["batch_size"], shuffle=True)
    # Instantiate the model
    # NOTE: The model class will move the data between multiple devices (if applicable).
    # This format has been followed since certain hyperparameters would lead to very large networks.
    # Hence we can split the model between 2 devices. Alternatively, the same device can be specified in the config.
    model = MotionNet(args.spatiotemporal_encoding, args.spatiotemporal_network,
                      args.deltaspatial_encoding, args.deltaspatial_network)
    # NOTE: By default the models are on the CPU. Call set_device() to manually specify the devices.
    model.set_device(args.spatiotemporal_device, args.deltaspatial_device)
    if args.verbose: print('-'*100, flush=True)
    if args.verbose: print(model)
    if args.verbose: print('-'*100, flush=True)

    # Optimizer.
    opt1 = torch.optim.Adam(model.xyt_to_d_enc.parameters(), lr=args.opt["lr"],
                       betas=(args.opt["beta1"], args.opt["beta2"]), eps=args.opt["eps"])
    opt2 = torch.optim.Adam(model.xyt_to_d_net.parameters(), lr=args.opt["lr"],# weight_decay=1e-6,
                        betas=(args.opt["beta1"], args.opt["beta2"]), eps=args.opt["eps"])
    opt3 = torch.optim.Adam(model.d_to_rgb_enc.parameters(), lr=args.opt["lr"],
                        betas=(args.opt["beta1"], args.opt["beta2"]), eps=args.opt["eps"])
    opt4 = torch.optim.Adam(model.d_to_rgb_net.parameters(), lr=args.opt["lr"], #weight_decay=1e-6,
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
        saved_dict = torch.load(latest_ckpt_path)
        model.load_state_dict(saved_dict["model_state_dict"])
        # if "optimizer_state_dict" in saved_dict.keys():
        #     opt.load_state_dict(saved_dict["optimizer_state_dict"])
        start_epoch = saved_dict["epoch"] + 1
        if args.verbose: print(f'Continuing from epoch {start_epoch}.')
    else:
        if args.verbose: print('Start from scratch.')
        start_epoch = 1
    # Epoch iteration.
    for epoch in range(start_epoch,epochs+1):
        train_loss = 0
        model.train()
        for count, item in enumerate(dloader):
            loc = dset.loc[item].half()
            pixel = dset.vid[item].half()
            output, _ = model(loc)
            # Since the model takes care of moving the data to different devices, move GT correspondingly.
            pixel = pixel.to(output.dtype).to(output.device)
            # Backpropagation.
            opt1.zero_grad() ; opt2.zero_grad() ; opt3.zero_grad() ; opt4.zero_grad()
            l2_error = (output - pixel)**2
            loss = l2_error.mean()
            loss.backward()
            opt1.step() ; opt2.step() ; opt3.step() ; opt4.step()
            train_loss += loss.item()
        print(f'Epoch: {epoch}, Loss: {train_loss/len(dloader)}', flush=True)
        # Trace the video data and save/plot the video/frames.
        if args.trace["folder"] is not None:
            if epoch >= args.trace["trace_epoch"]:
                with torch.no_grad():
                    model.eval()
                    trace_loc = dset.loc.half()
                    trace, _ = model(trace_loc)
                    trace = trace.detach().cpu().float().reshape(dset.shape).permute(2,0,1,3).numpy()
                    trace = (np.clip(trace, 0, 1)*255).astype(np.uint8)
                    save_path = os.path.join(args.trace["folder"], f'{args.trace["file_tag"]}{str(epoch).zfill(ndigits_epoch)}.avi')
                    iio2.mimwrite(save_path, trace, fps=30)
        if args.checkpoints["save"]:
            if args.verbose: print('Saving checkpoint.')
            if epoch % args.checkpoints["epoch_frequency"] == 0:
                checkpoint_file = f'{args.checkpoints["file_tag"]}{str(epoch).zfill(ndigits_epoch)}{args.checkpoints["ext"]}'
                # Save as dict to maintain uniformity
                torch.save({'model_state_dict': model.state_dict()}, 
                            os.path.join(args.checkpoints["dir"], checkpoint_file))
                if args.verbose: print(f'Saved model for epoch {epoch}.')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                # 'optimizer_state_dict': opt.state_dict(),
                }, latest_ckpt_path)
            if args.verbose: print('Saved latest checkpoint.')
        # Epoch demarcation
        print('-'*100)

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
    # Main routine.
    main(args)