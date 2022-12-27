import numpy as np
import torch
import tinycudann as tcnn

from .base import SineLayer
from ..utils.utils import positional_encoding, positional_encoding_phase

class MotionNet(torch.nn.Module):
    def __init__(self, spatiotemporal_to_delta_input_dims, spatiotemporal_to_delta_output_dims, 
                 spatiotemporal_to_delta_encoding, spatiotemporal_to_delta_network, 
                 deltaspatial_to_rgb_device_input_dims, deltaspatial_to_rgb_device_output_dims, 
                 deltaspatial_to_rgb_device_encoding, deltaspatial_to_rgb_device_network,
                 ):
        super().__init__()
        self.spatiotemporal_to_delta = tcnn.NetworkWithInputEncoding(spatiotemporal_to_delta_input_dims, 
                                                                     spatiotemporal_to_delta_output_dims, 
                                                                     spatiotemporal_to_delta_encoding,
                                                                     spatiotemporal_to_delta_network)
        self.deltaspatial_to_rgb = tcnn.NetworkWithInputEncoding(deltaspatial_to_rgb_device_input_dims, 
                                                                 deltaspatial_to_rgb_device_output_dims, 
                                                                 deltaspatial_to_rgb_device_encoding, 
                                                                 deltaspatial_to_rgb_device_network)
    
    def forward(self, coords):
        # double_dims = torch.cat((coords[...,2:3],torch.zeros_like(coords[...,2:3]).to(coords.device)), dim=-1)
        # delta = self.model_1(double_dims)
        delta = self.spatiotemporal_to_delta(coords)
        interim_out = coords[...,0:2] + delta.float()
        out = self.deltaspatial_to_rgb(interim_out.to(self.device_deltaspatial_to_rgb_device))
        return out.to(self.device_spatiotemporal_to_delta), interim_out
    
    def to_device(self,spatiotemporal_to_delta_device, deltaspatial_to_rgb_device):
        self.device_spatiotemporal_to_delta = spatiotemporal_to_delta_device
        self.device_deltaspatial_to_rgb_device = deltaspatial_to_rgb_device
        # Move to device
        self.spatiotemporal_to_delta.to(self.device_spatiotemporal_to_delta)
        self.deltaspatial_to_rgb.to(self.device_deltaspatial_to_rgb_device)
