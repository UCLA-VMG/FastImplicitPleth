import os
from tqdm import tqdm
import numpy as np
import commentjson as json

import torch
import torch.utils.data
import tinycudann as tcnn
import argparse

from ..models.siren import Siren
from ..models.combinations import MotionNet
from ..data.datasets import VideoGridDataset
from ..utils.utils import trace_video, Dict2Class

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Siren with Hash Grid Encodings.')
    parser.add_argument('-config', '--config_path', required=True, type=str, help='Path to the config file.')
    parser.add_argument('--verbose', action='store_true', help='Verbosity.')
    parser.add_argument('--append_save_path', default=None, type=str, help='Prepend the save paths for dataset automation.')
    parser.add_argument('--append_load_path', default=None, type=str, help='Prepend the load paths for dataset automation.')

    return parser.parse_args()

class ToFloat(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.float()

# import pdb
class PlethSpatioModelPreTrain(torch.nn.Module):
    def __init__(self, pleth_spatial_encoding, pleth_spatial_network):
        super().__init__()
        # spatial_enc = tcnn.Encoding(pleth_spatial_encoding["input_dims"], pleth_spatial_encoding)
        # self.spatial_model = torch.nn.Sequential(
        #                             spatial_enc,
        #                             ToFloat(),
        #                             Siren(spatial_enc.n_output_dims, 
        #                                   pleth_spatial_network["n_neurons"],
        #                                   pleth_spatial_network["output_dims"], 
        #                                   pleth_spatial_network["n_hidden_layers"], 
        #                                   first_omega_0=1, hidden_omega_0=1),
        #                             torch.nn.Sigmoid()
        #                         )
        # self.spatial_device = torch.device("cpu")

        self.spatial_model = tcnn.NetworkWithInputEncoding(pleth_spatial_encoding["input_dims"],
                                                           pleth_spatial_network["output_dims"],
                                                           pleth_spatial_encoding, 
                                                           pleth_spatial_network)
        self.spatial_device = torch.device("cpu")

    def forward(self, x):
        x_spatial = x.to(self.spatial_device)
        # if pr: print("In", x_spatial)
        spatial_out = self.spatial_model(x_spatial)
        # if pr: 
        #     print("Out", spatial_out)
        #     pdb.set_trace()
        return spatial_out, {"spatial_out": spatial_out}

    def set_device(self, spatial_device):
        self.spatial_device = spatial_device
        self.spatial_model.to(self.spatial_device)

def generate_gaussian_ref(x,y,x_mid,y_mid,std):
    out = np.exp((-(x-x_mid)**2)/(2*std**2))*np.exp((-(y-y_mid)**2)/(2*std**2))
    return out

def generate_pretrain_mask(shape: tuple = (128,128),sigma: float = 50) -> np.array:
    """ Generate a Gaussian image for pre-training

    Args:
        shape (tuple, optional): Shape of the image. Defaults to (128,128).
        sigma (float, optional): Standard deviation of the Gaussian. Defaults to 50.

    Returns:
        np.array: Gaussian image.
    """
    X,Y = np.meshgrid(np.arange(shape[1]),np.arange(shape[0]))

    X = X.flatten()
    Y = Y.flatten()

    ref = np.zeros((shape[0],shape[1]))

    ref[X,Y] = generate_gaussian_ref(X,Y,shape[0]/2,shape[1]/2,sigma)
    ref = ref.flatten()
    # To RGB
    ref = np.stack([ref,ref,ref], axis=-1)
    # from 0.5 tp 0.5
    X = (X / shape[1]) - 0.5
    Y = (Y / shape[0]) - 0.5

    return ref, X, Y

def main(args):
    gau_shape = (128,128)
    gau_img, grid_X, grid_Y = generate_pretrain_mask(gau_shape,sigma=35)
    gau_img_tensor = torch.tensor(gau_img)
    grid_tensor  = torch.stack((torch.tensor(grid_X),torch.tensor(grid_Y)), dim=-1)
    plt.imshow(gau_img.reshape(*gau_shape,3))
    plt.show(block=False)
    print(gau_img_tensor.shape)
    print(grid_tensor.shape)
    # Model
    pleth_model = PlethSpatioModelPreTrain(args.pleth_spatial_encoding, args.pleth_spatial_network)
    # To Device
    pleth_model.set_device(args.pleth_spatial_device)
    # Print the models
    if args.verbose: print('-'*100, flush=True)
    if args.verbose: print(pleth_model)
    if args.verbose: print('-'*100, flush=True)

    # Optimizer.
    # NOTE: When adding weight decay, make sure to split the Network and the Hash Encoding.
    #       The model would need to be altered to separate the network and encoding models.
    #       Individual optimizers are needed - network opt (uses decay) and encoding opt (doesn't use decay).
    #       Compatibility has been ensured and this alteration will not break the code.
    #       Furthermore, 2 optimizers would be needed, one for the encodings and one for the network.
    opt = torch.optim.Adam(pleth_model.spatial_model.parameters(), lr=args.opt["pre_train_lr"],
                                   betas=(args.opt["beta1"], args.opt["beta2"]), eps=args.opt["eps"])
    # Folders for the traced video data and checkpoints.
    if args.append_save_path is not None:
        args.trace["folder"] =  args.trace["folder"] + args.append_save_path
        args.pre_train_checkpoints["dir"] =  args.pre_train_checkpoints["dir"] + args.append_save_path
    if args.trace["folder"] is not None:
        os.makedirs(args.trace["folder"], exist_ok=True)
        if args.verbose: print(f'Saving trace to {args.trace["folder"]}')
    if args.pre_train_checkpoints["save"]:
        os.makedirs(args.pre_train_checkpoints["dir"], exist_ok=True)
        if args.verbose: print(f'Saving checkpoints to {args.pre_train_checkpoints["dir"]}')

    # Get info before iterating.
    epochs = args.train["pre_train_epochs"]
    ndigits_epoch = int(np.log10(epochs)+1)
    latest_ckpt_path = os.path.join(args.pre_train_checkpoints["dir"], args.pre_train_checkpoints["latest"])
    if os.path.exists(latest_ckpt_path):
        if args.verbose: print('Loading latest checkpoint...')
        saved_dict = torch.load(latest_ckpt_path)
        pleth_model.spatial_model.load_state_dict(saved_dict["model_state_dict"])
        if "optimizer_state_dict" in saved_dict.keys():
            opt.load_state_dict(saved_dict["optimizer_state_dict"])
        start_epoch = saved_dict["epoch"] + 1
        if args.verbose: print(f'Continuing from epoch {start_epoch}.')
    else:
        if args.verbose: print('Start from scratch.')
        start_epoch = 1
    # Epoch iteration.
    for epoch in range(1,epochs+1):
        pleth_model.train()
        loc = grid_tensor.half()
        pixel = gau_img_tensor
        output, _ = pleth_model(loc)
        # Since the model takes care of moving the data to different devices, move GT correspondingly.
        pixel = pixel.to(output.dtype)
        pixel = pixel.to(output.device)
        # Backpropagation.
        opt.zero_grad()
        l2_error = (output.to(args.io_device) - pixel)**2
        loss = l2_error.mean()
        loss.backward()
        opt.step()
        train_loss = loss.item()
        # for name, param in pleth_model.named_parameters():
        #     print(name, param)
        print(f'Epoch: {epoch}, Loss: {train_loss}', flush=True)
        # Trace the video data and save/plot the video/frames.
        if epoch >= args.trace["trace_epoch"]:
            trace_map(pleth_model.spatial_model, grid_tensor.half(), gau_shape, args.pleth_spatial_device, False,
                      save_dir=args.trace["folder"], \
                      save_file=f'pre_train_{args.trace["file_tag"]}{str(epoch).zfill(ndigits_epoch)}', \
                      verbose=args.verbose)
        # Save the checkpoints
        if args.pre_train_checkpoints["save"]:
            if args.verbose: print('Saving checkpoint.')
            if epoch % args.pre_train_checkpoints["epoch_frequency"] == 0:
                checkpoint_file = f'{args.pre_train_checkpoints["file_tag"]}{str(epoch).zfill(ndigits_epoch)}{args.pre_train_checkpoints["ext"]}'
                # Save as dict to maintain uniformity
                torch.save({'model_state_dict': pleth_model.spatial_model.state_dict()}, 
                           os.path.join(args.pre_train_checkpoints["dir"], checkpoint_file))
                if args.verbose: print(f'Saved model for epoch {epoch}.')
            torch.save({
                'epoch': epoch,
                'model_state_dict': pleth_model.spatial_model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                }, latest_ckpt_path)
            if args.verbose: print('Saved latest checkpoint.')
        # Epoch demarcation
        print('-'*100, flush=True)

import matplotlib.pyplot as plt
import imageio.v2 as iio2
def trace_map(model: torch.nn.Module, grid: torch.tensor, shape: tuple, device: torch.device, plot: bool = True, 
              save_dir: str = None, save_file: str = "epoch_", save_ext: str = ".png", 
              verbose: bool = True):
    """ WARNING!!! Under development. Might be deprecated.
    
    Trace the map from the spatial model.
    NOTE: Single Batch. Since most of our frames are 128x128, we opted for a single batch computation
    Depending on the GPU capacity and single frame size, this might have to be changed by the user.
    """
    import warnings
    warnings.warn("WARNING!!! Under development. Might be deprecated.")
    
    model.eval()
    if verbose: print('Tracing Image Grid Points')
    with torch.no_grad():
        inp = grid.half().to(device)
        output = model(inp) 
        if type(output) == tuple:
            output = output[0] 
        temp = output.squeeze().cpu().detach().float().numpy()
    if verbose: print('Arranging Tensor')
    temp = temp.reshape(shape[0],shape[1],3)
    temp = np.clip(temp, a_min=0, a_max=1)
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