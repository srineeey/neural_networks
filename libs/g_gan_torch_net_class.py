"""Neural Network class"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np

from utils import *
#import matplotlib.pyplot as plt

"""
the neural network class inherits from nn.Module
the default functions
init (constructor)
forward
have to be defined manually
"""


class Generator(nn.Module):
    
    def __init__(self, name, net_struct, latent_size, output_size, device):
        super(Generator, self).__init__()
        self.name = name
        self.net_struct = net_struct
        self.input_size = latent_size
        self.output_size = output_size
        self.batch_size = 1
        self.device = device
        self.set_layers(net_struct)
        
        """
        self.layer_sizes = calc_layer_sizes(self.input_size, self.net_struct) #inherit from utils
        self.layers = nn.ModuleList()
        
        print(f"Initializing {self.name}:\n")
        for layer in self.net_struct:
            print(f"Adding {layer}\n")
            self.layers.append(layer["type"](**layer["layer_pars"]))
        """    
        #self.init_weights(torch.nn.init.xavier_normal_)  #inherit from AutoEncoder
        
        #self.set_latent_dist(func, kwargs)
        std_no_kwargs = {"loc" : 0.0, "scale" : 1.0}
        self.set_latent_dist(torch.distributions.Normal, std_no_kwargs)

    #def forward(self, batch_size):
    def forward(self, x):
        
        """
        iterate through all layers and perform calculation
        """
        
        for layer_i in range(len(self.layers)):
            #print(layer_i)
            #print(x.shape)
            
            z = self.layers[layer_i](x)
            if "act_func" in self.net_struct[layer_i]:
                x = self.net_struct[layer_i]["act_func"](z)
                #x = self.net_struct[layer_i]["act_func"](z)
            else:
                x = z

            #print(x)
            #print(x.shape)

        return x

    
    def set_latent_dist(self, method, method_kwargs):
        self.latent_dist = method(**method_kwargs)

    
    """
    initialize weights
    Using initialization routine from the torch.nn.init package
    """
    
    def init_weights(self, init_routine):
        print(f"Initializing weights of {self.name} with method {init_routine}\n")
        for i, layer in enumerate(self.layers):
            #if type(layer) == nn.Linear:
            if type(layer) in [nn.Linear, nn.Conv2d]:
                #torch.nn.init.xavier_normal_(layer.weight)
                init_routine(layer.weight)
                #if self.net_struct[i]["bias"] == True:
                    #layer.bias.data.fill_(0.01)

    """
    Basic method(s) to quickly display network structure, input, output
    """
    
    
    def show_net_struct(self):
        print(f"Network Architecture of {self.name}:\n")
        for layer in self.net_struct:
            print( "Layer: {}\n".format(layer))

    
    def get_net_struct(self):
        return self.net_struct

    def get_input_size(self):
        return self.input_size

    def get_output_size(self):
        return self.output_size
    
    
    def set_layer_sizes(self, layer_sizes):
        self.layer_sizes = layer_sizes
        
    def set_layers(self, net_struct):
        self.net_struct = net_struct
        self.layer_sizes = calc_layer_sizes(self.input_size, self.net_struct) #inherit from utils
        
        if np.all(self.layer_sizes[-1] != self.output_size):
            raise ValueError(f'calculated {self.layer_sizes[-1]} and given output size {self.output_size} do not match!')
        
        self.layers = nn.ModuleList()
        
        print(f"Initializing {self.name}:\n")
        for layer in self.net_struct:
            print(f"Adding {layer}\n")
            self.layers.append(layer["type"](**layer["layer_pars"]))

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

