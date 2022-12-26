import os
from tqdm import tqdm
import numpy as np
import commentjson as json

import torch
import torch.utils.data
import tinycudann as tcnn
import argparse

from ..models.base import MLP
from ..data.datasets import VideoGridDataset
from ..utils.utils import trace_video, Dict2Class

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Siren with Hash Grid Encodings.')
    parser.add_argument('-vp', '--video_path', required=True, type=str, help='Path to the video')
    parser.add_argument('-config', '--config_path', required=True, type=str, help='Path to the config file')
    parser.add_argument('--verbose', action='store_true', help='Verbosity')

    return parser.parse_args()

class CombineNet(torch.nn.Module):
    def __init__(self, model_1, model_2):
        super().__init__()
        self.model_1 = model_1
        self.model_2 = model_2

    
    def forward(self, coords):
        # double_dims = torch.cat((coords[...,2:3],torch.zeros_like(coords[...,2:3]).to(coords.device)), dim=-1)
        # delta = self.model_1(double_dims)
        delta = self.model_1(coords)
        inp_model_2 = coords[...,0:2] + delta.float()
        out = self.model_2(inp_model_2.to(torch.device("cuda:1")))
        return out.to(torch.device("cuda:0")), None#{'delta': delta, 'pos_enc': pos_enc}
    
    def to_gpu(self):
        self.model_1.to(torch.device("cuda:0"))
        self.model_2.to(torch.device("cuda:1"))


def main(args):
    dset = VideoGridDataset(args.video_path, verbose=args.verbose, 
                            num_frames=args.data["num_frames"], start_frame=args.data["start_frame"])
    dloader = torch.utils.data.DataLoader(dset, batch_size=args.data["batch_size"], shuffle=True)
    trace_loader = torch.utils.data.DataLoader(dset, batch_size=args.data["trace_batch_size"], shuffle=False)

    time_to_phase = tcnn.NetworkWithInputEncoding(args.time_encoding["input_dims"], args.time_network["output_dims"], 
                                                  args.time_encoding, args.time_network)
    spatial_to_rgb = tcnn.NetworkWithInputEncoding(args.spatial_encoding["input_dims"], args.spatial_network["output_dims"], 
                                                  args.spatial_encoding, args.spatial_network)
    model = CombineNet(time_to_phase, spatial_to_rgb)
    model.to_gpu()
    if args.verbose: print(time_to_phase)
    if args.verbose: print(spatial_to_rgb)

    opt_time = torch.optim.Adam(time_to_phase.parameters(), lr=args.opt["lr"],
                               betas=(args.opt["beta1"], args.opt["beta2"]), eps=args.opt["eps"])
    opt_spatial = torch.optim.Adam(spatial_to_rgb.parameters(), lr=args.opt["lr"], #weight_decay=args.opt["l2_reg"],
                               betas=(args.opt["beta1"], args.opt["beta2"]), eps=args.opt["eps"])
    if args.trace["folder"] is not None:
        os.makedirs(args.trace["folder"], exist_ok=True)

    epochs = args.train["epochs"]
    ndigits_epoch = int(np.log10(epochs)+1)
    for epoch in range(1,epochs+1):
        train_loss = 0
        spatial_to_rgb.train()
        time_to_phase.train()
        for count, item in tqdm(enumerate(dloader),total=len(dloader)):
            loc = item["loc"].half().to(torch.device("cuda:0"))
            pixel = item["pixel"].half().to(torch.device("cuda:0"))/args.data["norm_value"]
            output, _ = model(loc)
            opt_time.zero_grad()
            opt_spatial.zero_grad()
            l2_error = (output - pixel.to(output.dtype))**2
            loss = l2_error.mean()
            loss.backward()
            opt_time.step()
            opt_spatial.step()
            train_loss += loss.item()
        print(f'Epoch: {epoch}, Loss: {train_loss/len(dloader)}', flush=True)	
        trace_video(model, dset, trace_loader, args.device, \
                    save_dir=args.trace["folder"], \
                    save_file=f'{args.trace["file_tag"]}{str(epoch).zfill(ndigits_epoch)}', \
                    save_ext=args.trace["ext"],\
                    plot=args.trace["plot"], verbose=args.verbose)
        print('-'*100, flush=True)

if __name__ == '__main__':
    args = parse_arguments()
    with open(args.config_path) as f:
        json_config = json.load(f)
    dict_args = vars(args)
    dict_args.update(json_config)
    args = Dict2Class(dict_args)
    if args.verbose: print(dict_args)
    args.device = torch.device(args.device)
    main(args)