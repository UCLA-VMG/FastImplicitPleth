import numpy as np
import torch

class LinearSineNet(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()

        self.hidden1 = torch.nn.Linear(in_features, hidden_features)
        self.hidden2 = torch.nn.Linear(hidden_features, hidden_features)
        self.final = torch.nn.Linear(hidden_features, out_features)

    
    def forward(self, coords):
        l1_o = torch.sin(self.hidden1(coords.float()))
        l2_o = torch.sin(self.hidden2(l1_o))
        out = self.final(l2_o)

        return out, {'sine1out':l1_o, 'sine2out':l2_o}

    def forward_with_intermediate(self, coords):
        hidden1_o = self.hidden1(coords.float())
        l1_o = torch.sin(hidden1_o)
        hidden2_o = self.hidden2(l1_o)
        l2_o = torch.sin(hidden2_o)
        out = self.final(l2_o)

        return out, {'sine1out':l1_o, 'sine2out':l2_o, 'hidden1out':hidden1_o, 'hidden2out':hidden2_o}

class SineLayer(torch.nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate

class SineAct(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inp):
        return torch.sin(inp)

class MLP(torch.nn.Module):
    def __init__(self, in_features, hidden_features, n_hidden, out_features,
                 activation="ReLU", act_kwargs=None, bias=True):
        super().__init__()
        if activation.lower() == "relu":
            self.act = torch.nn.ReLU()
        elif activation.lower() == "sine":
            self.act = SineAct()
        elif activation.lower() == "leakyrelu":
            self.act = torch.nn.LeakyReLU(act_kwargs["negative_slope"])
        elif activation.lower() == "sigmoid":
            self.act = torch.nn.Sigmoid()
        elif activation.lower() == "tanh":
            self.act = torch.nn.Tanh()
        else:
            NotImplementedError
        self.net = []
        # First layer
        self.net.append(torch.nn.Linear(in_features, hidden_features, bias=bias))
        self.net.append(self.act)
        # All the other layers
        for _ in range(n_hidden):
            self.net.append(torch.nn.Linear(hidden_features, hidden_features, bias=bias))
            self.net.append(self.act)
        self.net.append(torch.nn.Linear(hidden_features, out_features, bias=bias))
        self.net = torch.nn.Sequential(*self.net)

    def forward(self, coords, detach=False):
        out = self.net(coords)
        return out
