import numpy as np
import torch
from collections import OrderedDict

from .base import SineLayer
from ..utils.utils import positional_encoding, positional_encoding_phase

class Siren(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features, n_hidden,
                 outermost_linear=True, first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        self.net = []
        # First layer
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))
        # All the other layers
        for _ in range(n_hidden):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        if outermost_linear:
            final_layer = torch.nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_layer.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                                np.sqrt(6 / hidden_features) / hidden_omega_0)
        else:
            final_layer = SineLayer(hidden_features, out_features, 
                                        is_first=False, omega_0=hidden_omega_0)
        self.net.append(final_layer)
        self.net = torch.nn.Sequential(*self.net)

    def forward(self, coords, detach=False):
        out = self.net(coords)
        return out
    
    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations

class SirenPhaseEncodingMotionSpatioTemporal(torch.nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, \
                 out_features, outermost_linear=True, 
                 first_omega_0=30, hidden_omega_0=30., L_min=0, L_max=10):
        super().__init__()
        self.L_max = L_max
        self.L_min = L_min
        
        # Siren for phase encoding
        self.l1a = SineLayer((self.L_max-self.L_min)*2, hidden_features, 
                                      is_first=True, omega_0=first_omega_0)
        self.l2a = SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0)

        self.l3a  = torch.nn.Linear(hidden_features, (self.L_max-self.L_min)*2)
        with torch.no_grad():
            self.l3a .weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                            np.sqrt(6 / hidden_features) / hidden_omega_0)

        # Positional Implicit Network to generate x-y based color info based on a phase change
        self.l1b = SineLayer((self.L_max-self.L_min)*4, hidden_features, 
                                      is_first=True, omega_0=first_omega_0)
        self.l2b = SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0)
    
    def forward(self, coords): #mode=1: both, mode=0, only phase part
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
                
        # Time coordinates
        coords1 = positional_encoding(coords[...,2:3], L_max = self.L_max, L_min = self.L_min)
        l1a_o = self.l1a(coords1)
        l2a_o = self.l2a(l1a_o)
        phase = self.l3a(l2a_o)

        # Spatial Coordinates
        coords2 = positional_encoding_phase(coords[...,0:2], phase, L_max = self.L_max, L_min = self.L_min)
        l1b_o = self.l1b(coords2)
        l2b_o = self.l2b(l1b_o)
        
        return l2b_o, coords1, coords2

class SirenPhaseEncodingMotionSpatial(torch.nn.Module):
    def __init__(self, in_features, hidden_features, n_hidden, \
                 out_features, outermost_linear=True, 
                 first_omega_0=30, hidden_omega_0=30., L_min=0, L_max=10):
        super().__init__()
        self.L_max = L_max
        self.L_min = L_min
    
        # Positional Implicit Network to generate x-y based color info based on a phase change
        self.spatial_siren = Siren((self.L_max-self.L_min)*2*in_features, hidden_features,
                                    out_features, n_hidden, outermost_linear,
                                    first_omega_0, hidden_omega_0)

    def forward(self, coords, phase):
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        # Spatial Coordinates
        coords2 = positional_encoding_phase(coords, phase, L_max = self.L_max, L_min = self.L_min)
        out = self.spatial_siren(coords2)
        return out, coords2