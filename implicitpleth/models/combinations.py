import numpy as np
import torch
import tinycudann as tcnn

from .base import SineLayer
from ..utils.utils import positional_encoding, positional_encoding_phase

class  MotionNet(torch.nn.Module):
    def __init__(self, spatiotemporal_to_delta_encoding, spatiotemporal_to_delta_network, 
                 deltaspatial_to_rgb_encoding, deltaspatial_to_rgb_network):
        super().__init__()
        self.spatiotemporal_to_delta = tcnn.NetworkWithInputEncoding(spatiotemporal_to_delta_encoding["input_dims"], 
                                                                     spatiotemporal_to_delta_network["output_dims"], 
                                                                     spatiotemporal_to_delta_encoding,
                                                                     spatiotemporal_to_delta_network)
        self.deltaspatial_to_rgb = tcnn.NetworkWithInputEncoding(deltaspatial_to_rgb_encoding["input_dims"], 
                                                                 deltaspatial_to_rgb_network["output_dims"], 
                                                                 deltaspatial_to_rgb_encoding, 
                                                                 deltaspatial_to_rgb_network)
        self.device_spatiotemporal_to_delta = torch.device("cpu")
        self.device_deltaspatial_to_rgb_device = torch.device("cpu")
    
    def forward(self, coords):
        coords = coords.to(self.device_spatiotemporal_to_delta)
        delta = self.spatiotemporal_to_delta(coords)
        interim_out = coords[...,0:2] + delta.float()
        out = self.deltaspatial_to_rgb(interim_out.to(self.device_deltaspatial_to_rgb_device))
        return out.to(self.device_spatiotemporal_to_delta), interim_out
    
    def set_device(self,device_spatiotemporal_to_delta, device_deltaspatial_to_rgb):
        self.device_spatiotemporal_to_delta = device_spatiotemporal_to_delta
        self.device_deltaspatial_to_rgb_device = device_deltaspatial_to_rgb
        # Move to device
        self.spatiotemporal_to_delta.to(self.device_spatiotemporal_to_delta)
        self.deltaspatial_to_rgb.to(self.device_deltaspatial_to_rgb_device)
