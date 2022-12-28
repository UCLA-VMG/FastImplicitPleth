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
    parser.add_argument('-vp', '--video_path', required=True, type=str, help='Path to the video')
    parser.add_argument('-config', '--config_path', required=True, type=str, help='Path to the config file')
    parser.add_argument('--verbose', action='store_true', help='Verbosity')
    parser.add_argument('--prepend_save_path', default=None, type=str, help='Verbosity')

    return parser.parse_args()

def main(args):
    # Create the Dataset object and the two dataloaders - one for train and one for tracing.
    dset = VideoGridDataset(args.video_path, verbose=args.verbose, 
                            num_frames=args.data["num_frames"], start_frame=args.data["start_frame"])
    dloader = torch.utils.data.DataLoader(dset, batch_size=args.data["batch_size"], shuffle=True)
    trace_loader = torch.utils.data.DataLoader(dset, batch_size=args.data["trace_batch_size"], shuffle=False)
    # Instantiate the model
    # NOTE: The model class will move the data between multiple devices (if applicable).
    # This format has been followed since certain hyperparameters would lead to very large networks.
    # Hence we can split the model between 2 devices. Alternatively, the same device can be specified in the config.
    model = MotionNet(args.spatiotemporal_encoding, args.spatiotemporal_network,
                      args.deltaspatial_encoding, args.deltaspatial_network)
    # NOTE: By default the models are on the CPU. Call set_device() to manually specify the devices.
    model.set_device(args.motion_spatiotemporal_device, args.motion_deltaspatial_device)
    if args.verbose: print(model)

    # Optimizer.
    # NOTE: When adding weight decay, make sure to split the Network and the Hash Encoding.
    #       The Motion model would need to be altered to separate the network and encoding models.
    #       Individual optimizers are needed - network opt (uses decay) and encoding opt (doesn't use decay).
    #       Compatibility has been ensured and this alteration will not break the code.
    #       Furthermore, 2 optimizers would be needed, one for the encodings and one for the network.
    opt = torch.optim.Adam(model.parameters(), lr=args.opt["lr"],
                           betas=(args.opt["beta1"], args.opt["beta2"]), eps=args.opt["eps"])
    # Folders for the traced video data and checkpoints.
    if args.prepend_save_path is not None:
        args.trace["folder"] = args.prepend_save_path + args.trace["folder"]
        args.checkpoints["dir"] = args.prepend_save_path + args.checkpoints["dir"]
    if args.trace["folder"] is not None:
        os.makedirs(args.trace["folder"], exist_ok=True)
        if args.verbose: print(f'Saving trace to {args.trace["folder"]}')
    if args.checkpoints["dir"] is not None:
        os.makedirs(args.checkpoints["dir"], exist_ok=True)
        if args.verbose: print(f'Saving checkpoints to {args.checkpoints["dir"]}')

    # Get info before iterating.
    epochs = args.train["epochs"]
    ndigits_epoch = int(np.log10(epochs)+1)
    # Epoch iteration.
    for epoch in range(1,epochs+1):
        train_loss = 0
        model.train()
        for count, item in tqdm(enumerate(dloader),total=len(dloader)):
            loc = item["loc"].half()
            pixel = item["pixel"].half()/args.data["norm_value"]
            output, _ = model(loc)
            # Since the model takes care of moving the data to different devices, move GT correspondingly.
            pixel = pixel.to(output.dtype)
            pixel = pixel.to(output.device)
            # Backpropagation.
            opt.zero_grad()
            l2_error = (output - pixel)**2
            loss = l2_error.mean()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        print(f'Epoch: {epoch}, Loss: {train_loss/len(dloader)}', flush=True)	
        # Trace the video data and save/plot the video/frames.
        if epoch >= args.trace["trace_epoch"]:
            trace_video(model, dset, trace_loader, args.motion_spatiotemporal_device, \
                        save_dir=args.trace["folder"], \
                        save_file=f'{args.trace["file_tag"]}{str(epoch).zfill(ndigits_epoch)}', \
                        save_ext=args.trace["ext"],\
                        plot=args.trace["plot"], verbose=args.verbose)
        # Save the checkpoints
        if args.checkpoints["dir"] is not None:
            if epoch % args.checkpoints["epoch_frequency"]:
                checkpoint_file = f'{args.checkpoints["latest"]}{args.checkpoints["ext"]}'
                # Save as dict to maintain uniformity
                torch.save({'model_state_dict': model.state_dict()}, 
                           os.path.join(args.checkpoints["dir"], checkpoint_file))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                }, os.path.join(args.checkpoints["dir"],args.checkpoints["latest"]))
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
    if args.verbose: print(dict_args)
    # Convert device string to torch.device().
    args.motion_spatiotemporal_device = torch.device(args.motion_spatiotemporal_device)
    args.motion_deltaspatial_device = torch.device(args.motion_deltaspatial_device)
    # Main routine.
    main(args)