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
init (constructor) and forward
have to be defined manually
"""

class GAN(nn.Module):
    """
    class constructor
    typically the neural networks variables and structure are initialized here
    super().__init__() is necessary to call the mother class constructor
    """
    
    def __init__(self, generator_struct, discriminator_struct, data_size, latent_size, device):
        super(GAN, self).__init__()
        self.name = "GenerativeAdversarialNetwork"
        self.generator_struct = generator_struct
        self.discriminator_struct = discriminator_struct
        self.data_size = data_size
        self.latent_size = latent_size
        self.batch_size = 1
        self.device = device
        
        """
        Construct neural network layers from subclasses
        """
        
        self.generator = Generator("Generator", self.generator_struct, self.latent_size, self.data_size, device)
        #self.discriminator = Discriminator("Discriminator", self.discriminator_struct, self.data_size, 2, device)
        self.discriminator = Discriminator("Discriminator", self.discriminator_struct, self.data_size, 1, device)
        

    """
    the forward function is called each the the network is propagating forward
    takes in input data and spits out the predicted output
    """
    
    #def forward(self, batch_size):
    def forward(self, x):

        #gen_sample = self.generator(batch_size)
        gen_sample = self.generator(x)
        
        prob = self.discriminator(gen_sample)
        
        
        return (gen_sample, prob)
    
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
        self.layers = nn.ModuleList()
        
        print(f"Initializing {self.name}:\n")
        for layer in self.net_struct:
            print(f"Adding {layer}\n")
            self.layers.append(layer["type"](**layer["layer_pars"]))

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        
    def get_latent_variables(self):
        return (self.latent_mu, self.log_latent_sigma)
    
"""
Encoder and Decoder have similar structure
so that one can reutilize the same class for both!
"""
#class Coder(nn.Module, AutoEncoder):
class Generator(GAN):
    
    def __init__(self, name, net_struct, latent_size, output_size, device):
        super(Generator, self).__init__()
        self.name = name
        #self.net_struct = net_struct
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
            else:
                x = z

            #print(x)
            #print(x.shape)

        return x

    def set_latent_dist(self, method, method_kwargs):
        self.dist = method(**method_kwargs)

class Discriminator(GAN):
    
    def __init__(self, name, net_struct, input_size, output_size, device):
        super(Discriminator, self).__init__()
        self.name = name
        #self.net_struct = net_struct
        self.input_size = input_size
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
        

    def forward(self, input_data):

        x = input_data
        #print(x.shape)
        
        """
        iterate through all layers and perform calculation
        """
        
        for layer_i in range(len(self.layers)):
            #print(layer_i)
            #print(x.shape)
            
            z = self.layers[layer_i](x)
            if "act_func" in self.net_struct[layer_i]:
                x = self.net_struct[layer_i]["act_func"](z)
            else:
                x = z

            #print(x)
            #print(x.shape)

        return x
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

