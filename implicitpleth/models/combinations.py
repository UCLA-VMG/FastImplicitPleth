import numpy as np
import torch
import tinycudann as tcnn

from .base import SineLayer
from ..utils.utils import positional_encoding, positional_encoding_phase

# class  OldMotionNet(torch.nn.Module):
#     def __init__(self, spatiotemporal_to_delta_encoding, spatiotemporal_to_delta_network, 
#                  deltaspatial_to_rgb_encoding, deltaspatial_to_rgb_network):
#         super().__init__()
#         self.spatiotemporal_to_delta = tcnn.NetworkWithInputEncoding(spatiotemporal_to_delta_encoding["input_dims"], 
#                                                                      spatiotemporal_to_delta_network["output_dims"], 
#                                                                      spatiotemporal_to_delta_encoding,
#                                                                      spatiotemporal_to_delta_network)
#         self.deltaspatial_to_rgb = tcnn.NetworkWithInputEncoding(deltaspatial_to_rgb_encoding["input_dims"], 
#                                                                  deltaspatial_to_rgb_network["output_dims"], 
#                                                                  deltaspatial_to_rgb_encoding, 
#                                                                  deltaspatial_to_rgb_network)
#         self.device_spatiotemporal_to_delta = torch.device("cpu")
#         self.device_deltaspatial_to_rgb_device = torch.device("cpu")
    
#     def forward(self, coords):
#         coords = coords.to(self.device_spatiotemporal_to_delta)
#         delta = self.spatiotemporal_to_delta(coords)
#         interim_out = coords[...,0:2] + delta.float()/2 # tanh is -1 to 1. But we need -0.5 to 0.5
#         out = self.deltaspatial_to_rgb(interim_out.to(self.device_deltaspatial_to_rgb_device))
#         return out.to(self.device_spatiotemporal_to_delta), interim_out
    
#     def set_device(self,device_spatiotemporal_to_delta, device_deltaspatial_to_rgb):
#         self.device_spatiotemporal_to_delta = device_spatiotemporal_to_delta
#         self.device_deltaspatial_to_rgb_device = device_deltaspatial_to_rgb
#         # Move to device
#         self.spatiotemporal_to_delta.to(self.device_spatiotemporal_to_delta)
#         self.deltaspatial_to_rgb.to(self.device_deltaspatial_to_rgb_device)


class  MotionNet(torch.nn.Module):
    def __init__(self, spatiotemporal_to_delta_encoding, spatiotemporal_to_delta_network, 
                 deltaspatial_to_rgb_encoding, deltaspatial_to_rgb_network):
        super().__init__()
        self.xyt_to_d_enc = tcnn.Encoding(spatiotemporal_to_delta_encoding["input_dims"], 
                                          spatiotemporal_to_delta_encoding)
        self.xyt_to_d_net = tcnn.Network(self.xyt_to_d_enc.n_output_dims, spatiotemporal_to_delta_network["output_dims"], 
                                         spatiotemporal_to_delta_network)

        self.d_to_rgb_enc = tcnn.Encoding(deltaspatial_to_rgb_encoding["input_dims"], 
                                          deltaspatial_to_rgb_encoding)
        self.d_to_rgb_net = tcnn.Network(self.d_to_rgb_enc.n_output_dims, deltaspatial_to_rgb_network["output_dims"], 
                                         deltaspatial_to_rgb_network)
        self.spatiotemporal_to_delta = torch.nn.Sequential(self.xyt_to_d_enc, self.xyt_to_d_net)
        self.deltaspatial_to_rgb = torch.nn.Sequential(self.d_to_rgb_enc, self.d_to_rgb_net)
        self.device_spatiotemporal_to_delta = torch.device("cpu")
        self.device_deltaspatial_to_rgb_device = torch.device("cpu")
    
    def forward(self, coords):
        coords = coords.to(self.device_spatiotemporal_to_delta)
        delta = self.spatiotemporal_to_delta(coords)
        interim_out = coords[...,0:2] + (delta/2) # tanh is -1 to 1. But we need -0.5 to 0.5
        out = self.deltaspatial_to_rgb(interim_out.to(self.device_deltaspatial_to_rgb_device))
        return out.to(self.device_spatiotemporal_to_delta), interim_out
    
    def set_device(self,device_spatiotemporal_to_delta, device_deltaspatial_to_rgb):
        self.device_spatiotemporal_to_delta = device_spatiotemporal_to_delta
        self.device_deltaspatial_to_rgb_device = device_deltaspatial_to_rgb
        # Move to device
        self.spatiotemporal_to_delta.to(self.device_spatiotemporal_to_delta)
        self.deltaspatial_to_rgb.to(self.device_deltaspatial_to_rgb_device)
