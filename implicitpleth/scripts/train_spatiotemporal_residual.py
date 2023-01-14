import os
from tqdm import tqdm
import numpy as np
import commentjson as json

import torch
import torch.utils.data
import tinycudann as tcnn
import argparse

from ..models.combinations import AppearanceNet
from ..data.datasets import VideoGridDataset
from ..utils.utils import trace_video, Dict2Class

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Siren with Hash Grid Encodings.')
    parser.add_argument('-vp', '--video_path', required=True, type=str, help='Path to the video.')
    parser.add_argument('-config', '--config_path', required=True, type=str, help='Path to the config file.')
    parser.add_argument('--verbose', action='store_true', help='Verbosity.')
    parser.add_argument('--append_save_path', default=None, type=str, help='Prepend the save paths for dataset automation.')
    parser.add_argument('--append_load_path', default=None, type=str, help='Prepend the load paths for dataset automation.')

    return parser.parse_args()

class PlethSpatioTemporalModel(torch.nn.Module):
    def __init__(self, pleth_spatial_encoding, pleth_spatial_network, 
                 pleth_temporal_encoding, pleth_temporal_network):
        super().__init__()
        self.spatial_model = tcnn.NetworkWithInputEncoding(pleth_spatial_encoding["input_dims"],
                                                           pleth_spatial_network["output_dims"],
                                                           pleth_spatial_encoding, 
                                                           pleth_spatial_network)
        self.temporal_model = tcnn.NetworkWithInputEncoding(pleth_temporal_encoding["input_dims"],
                                                            pleth_temporal_network["output_dims"],
                                                            pleth_temporal_encoding, 
                                                            pleth_temporal_network)
        self.spatial_device = torch.device("cpu")
        self.temporal_device = torch.device("cpu")
    
    def forward(self, x, flag = False):
        # All custom models in the repo return 2 values
        if flag:
            x_temporal = x[...,2:3]
            x_temporal = torch.cat((x_temporal,torch.zeros_like(x_temporal)), dim=-1).to(self.temporal_device)
            temporal_out = self.temporal_model(x_temporal)
            return temporal_out, {"spatial_out": None, "temporal_out": temporal_out}
        else:
            x_spatial = x[...,0:2].to(self.spatial_device)
            spatial_out = self.spatial_model(x_spatial)
            x_temporal = x[...,2:3]
            x_temporal = torch.cat((x_temporal,torch.zeros_like(x_temporal)), dim=-1).to(self.temporal_device)
            temporal_out = self.temporal_model(x_temporal)
            return spatial_out * temporal_out, {"spatial_out": spatial_out, "temporal_out": temporal_out}

    def set_device(self, spatial_device, temporal_device):
        self.spatial_device = spatial_device
        self.temporal_device = temporal_device
        self.spatial_model.to(self.spatial_device)
        self.temporal_model.to(self.temporal_device)

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
    # Model 1: Capture Appearance 
    with open(args.appearance_model["config"]) as mmf:
        args.appearance_model["config"] = json.load(mmf)
    appearance_model = AppearanceNet(args.appearance_model["config"]["spatiotemporal_encoding"], 
                             args.appearance_model["config"]["spatiotemporal_network"],
                             args.appearance_model["config"]["deltaspatial_encoding"], 
                             args.appearance_model["config"]["deltaspatial_network"])
    # Load pre-trained weights
    if args.append_load_path is not None:
        args.appearance_model["load_path"] = args.appearance_model["load_path"] + args.append_load_path
    if args.verbose: print(f'Loading appearance model from {args.appearance_model["load_path"]}')
    appearance_model.load_state_dict(torch.load(args.appearance_model["load_path"])["model_state_dict"])
    # Freeze the model
    appearance_model.eval()
    for params in appearance_model.parameters():
        params.requires_grad = False
    # Set the model device
    appearance_spatiotemporal_device = torch.device(args.spatiotemporal_device)
    appearance_deltaspatial_device = torch.device(args.deltaspatial_device)
    appearance_model.set_device(appearance_spatiotemporal_device, appearance_deltaspatial_device)
    # Model 2: Capture PPG
    pleth_model = PlethSpatioTemporalModel(args.pleth_spatial_encoding, args.pleth_spatial_network, 
                                           args.pleth_temporal_encoding, args.pleth_temporal_network)
    # To Device
    pleth_model.set_device(args.pleth_spatial_device, args.pleth_temporal_device)
    # Ensemble Adder
    # Trace takes in only a single model. Hence required.
    ensemble = AdditionEnsemble(appearance_model, pleth_model)
    ensemble.set_device(args.io_device) # NOTE: This only stores the device does not move the models.
    # Print the models
    if args.verbose: print('-'*100, flush=True)
    if args.verbose: print(ensemble)
    if args.verbose: print('-'*100, flush=True)

    # Optimizer.
    # NOTE: When adding weight decay, make sure to split the Network and the Hash Encoding.
    #       The model would need to be altered to separate the network and encoding models.
    #       Individual optimizers are needed - network opt (uses decay) and encoding opt (doesn't use decay).
    #       Compatibility has been ensured and this alteration will not break the code.
    #       Furthermore, 2 optimizers would be needed, one for the encodings and one for the network.
    opt_spatial = torch.optim.Adam(pleth_model.spatial_model.parameters(), lr=args.opt["lr"],
                                   betas=(args.opt["beta1"], args.opt["beta2"]), eps=args.opt["eps"])
    opt_temporal = torch.optim.Adam(pleth_model.temporal_model.parameters(), lr=args.opt["lr"],
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
        pleth_model.load_state_dict(saved_dict["model_state_dict"])
        if "optimizer_spatial_state_dict" in saved_dict.keys():
            opt_spatial.load_state_dict(saved_dict["optimizer_spatial_state_dict"])
        if "optimizer_temporal_state_dict" in saved_dict.keys():
            opt_temporal.load_state_dict(saved_dict["optimizer_temporal_state_dict"])
        start_epoch = saved_dict["epoch"] + 1
        if args.verbose: print(f'Continuing from epoch {start_epoch}.')
    else:
        if args.verbose: print('Start from scratch.')
        start_epoch = 1
    # Epoch iteration.
    # TODO: Shift to config
    epoch_shift = 3
    # count_shift = 10
    ptc_flag = True
    spatial_temporal_bar = False
    for epoch in range(1,epochs+1):
        train_loss = 0
        pleth_model.train()
        if epoch % epoch_shift == 0:
            ptc_flag = False
            spatial_temporal_bar = not spatial_temporal_bar
            if args.verbose: print("Spatial Mode") if spatial_temporal_bar else print("Temporal Mode")
        for count, item in tqdm(enumerate(dloader, start=1),total=len(dloader)):
            # if count % count_shift == 0:
            #     spatial_temporal_bar = not spatial_temporal_bar
            loc = item["loc"].half()
            pixel = item["pixel"].half()/args.data["norm_value"]
            # Ensemble is not used since we want to use torch.np_grad()
            # with torch.no_grad():
            appearance_out, _ = appearance_model(loc)
            ppg_res_out, _ = pleth_model(loc,ptc_flag)
            output = appearance_out.to(args.io_device) + ppg_res_out.to(args.io_device)
            # Since the model takes care of moving the data to different devices, move GT correspondingly.
            pixel = pixel.to(output.dtype)
            pixel = pixel.to(output.device)
            # Backpropagation.
            if spatial_temporal_bar:
                opt_spatial.zero_grad()
                l2_error = (output.to(args.io_device) - pixel)**2
                loss = l2_error.mean()
                loss.backward()
                opt_spatial.step()
            else:
                opt_temporal.zero_grad()
                l2_error = (output.to(args.io_device) - pixel)**2
                loss = l2_error.mean()
                loss.backward()
                opt_temporal.step()
            train_loss += loss.item()
        # for name, param in pleth_model.named_parameters():
        #     print(name, param)
        print(f'Epoch: {epoch}, Loss: {train_loss/len(dloader)}', flush=True)
        # Trace the video data and save/plot the video/frames.
        if epoch >= args.trace["trace_epoch"]:
            trace_video(ensemble, dset, trace_loader, args.io_device, \
                        save_dir=args.trace["folder"], \
                        save_file=f'ensemble_{args.trace["file_tag"]}{str(epoch).zfill(ndigits_epoch)}', \
                        save_ext=args.trace["ext"],\
                        plot=args.trace["plot"], verbose=args.verbose)
            trace_map(pleth_model.spatial_model, dset, args.pleth_spatial_device, False,
                      save_dir=args.trace["folder"], \
                      save_file=f'saptial_{args.trace["file_tag"]}{str(epoch).zfill(ndigits_epoch)}', \
                      verbose=args.verbose)
            # trace_video(appearance_model, dset, trace_loader, args.io_device, \
            #             save_dir=args.trace["folder"], \
            #             save_file=f'appearance_{args.trace["file_tag"]}{str(epoch).zfill(ndigits_epoch)}', \
            #             save_ext=args.trace["ext"],\
            #             plot=args.trace["plot"], verbose=args.verbose)
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
                'optimizer_spatial_state_dict': opt_spatial.state_dict(),
                'optimizer_temporal_state_dict': opt_temporal.state_dict(),
                }, latest_ckpt_path)
            if args.verbose: print('Saved latest checkpoint.')
        # Epoch demarcation
        print('-'*100, flush=True)

import matplotlib.pyplot as plt
import imageio.v2 as iio2
def trace_map(model: torch.nn.Module, dataset: object, device: torch.device, plot: bool = True, 
              save_dir: str = None, save_file: str = "epoch_", save_ext: str = ".png", 
              verbose: bool = True):
    """ WARNING!!! Under development. Might be deprecated.
    
    Trace the map from the spatial model.
    NOTE: Single Batch. Since most of our frames are 128x128, we opted for a single batch computation
    Depending on the GPU capacity and single frame size, this might have to be changed by the user.


    Args:
        model (torch.nn.Module): _description_
        dataset (object): _description_
        device (torch.device): _description_
        plot (bool, optional): _description_. Defaults to True.
        save_dir (str, optional): _description_. Defaults to None.
        save_file (str, optional): _description_. Defaults to "epoch_".
        save_ext (str, optional): _description_. Defaults to ".png".
        verbose (bool, optional): _description_. Defaults to True.
    """
    import warnings
    warnings.warn("WARNING!!! Under development. Might be deprecated.")
    
    model.eval()
    if verbose: print('Tracing Image Grid Points')
    with torch.no_grad():
        inp = dataset.generate_spatial_tensor_grid().half().to(device)
        # print(inp.shape)
        output = model(inp) 
        if type(output) == tuple:
            output = output[0] 
        temp = output.squeeze().cpu().detach().float().numpy()
    if verbose: print('Arranging Tensor')
    temp = temp.reshape(dataset.shape[0],dataset.shape[1],3)
    print('-'*10,'   ', np.mean(temp), '   ', '-'*10)
    temp = np.clip(temp, a_min=0, a_max=1)
    print('-'*10,'   ', np.mean(temp), '   ', '-'*10)
    if plot:
        plt.figure(figsize=(20,20))
        plt.imshow(temp)
        plt.show()
    if save_dir is not None:
        save_path = f"{save_dir}/{save_file}{save_ext}"
        if verbose: print('Saving Traced Image')
        temp = (temp*255).astype(np.uint8)
        iio2.imwrite(save_path, temp)
        if verbose: print(f'Image Saved to {save_path}')
    del temp

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
    args.pleth_spatial_device = torch.device(args.pleth_spatial_device)
    args.pleth_temporal_device = torch.device(args.pleth_temporal_device)
    args.io_device = torch.device(args.io_device)
    # Main routine.
    main(args)