import os
from tqdm import tqdm
import numpy as np
import commentjson as json

import torch
import torch.utils.data
import tinycudann as tcnn
import argparse

from ..models.base import LinearSineNet
from ..data.datasets import VideoGridDataset
from ..utils.utils import trace_video, trace_video_tqdm, Dict2Class

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Siren with Hash Grid Encodings.')
    parser.add_argument('-vp', '--video_path', required=True, type=str, help='Path to the video')
    parser.add_argument('-config', '--config_path', required=True, type=str, help='Path to the config file')
    parser.add_argument('--verbose', action='store_true', help='Verbosity')

    return parser.parse_args()

def main(args):
    dset = VideoGridDataset(args.video_path, num_frames=args.data["num_frames"],verbose=args.verbose)
    dloader = torch.utils.data.DataLoader(dset, batch_size=args.data["batch_size"], shuffle=True)
    trace_loader = torch.utils.data.DataLoader(dset, batch_size=args.data["trace_batch_size"], shuffle=False)

    encoding = tcnn.Encoding(args.encoding["input_dims"], args.encoding)
    network = LinearSineNet(encoding.n_output_dims, args.network["hidden_dims"], args.network["output_dims"])
    model = torch.nn.Sequential(encoding, network)
    model.to(args.device)
    if args.verbose: print(model)

    opt_enc = torch.optim.Adam(encoding.parameters(), lr=args.opt["lr"], \
                               betas=(args.opt["beta1"], args.opt["beta2"]), eps=args.opt["eps"])
    opt_net = torch.optim.Adam(network.parameters(), lr=args.opt["lr"], weight_decay=args.opt["l2_reg"], \
                               betas=(args.opt["beta1"], args.opt["beta2"]), eps=args.opt["eps"])
    if args.trace["folder"] is not None:
        os.makedirs(args.trace["folder"], exist_ok=True)

    epochs = args.train["epochs"]
    ndigits_epoch = int(np.log10(epochs)+1)
    for epoch in range(1,epochs+1):
        train_loss = 0
        model.train()
        for count, item in tqdm(enumerate(dloader),total=len(dloader)):
            loc = item["loc"].half().to(args.device)
            pixel = item["pixel"].half().to(args.device)/args.data["norm_value"]
            output, _ = model(loc)
            opt_enc.zero_grad()
            opt_net.zero_grad()
            relative_l2_error = (output - pixel.to(output.dtype))**2
            loss = relative_l2_error.mean()
            loss.backward()
            opt_enc.step()
            opt_net.step()
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