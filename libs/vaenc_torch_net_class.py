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

class AutoEncoder(nn.Module):
    """
    class constructor
    typically the neural networks variables and structure are initialized here
    super().__init__() is necessary to call the mother class constructor
    """
    
    def __init__( self, encoder_struct, decoder_struct, input_size, latent_size, device):
        super(AutoEncoder, self).__init__()
        self.name = "VariationalAutoEncoder"
        self.encoder_struct = encoder_struct
        self.decoder_struct = decoder_struct
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = self.input_size
        self.batch_size = 1
        self.device = device
        
        self.latent_mu = 0.
        self.latent_sigma = 1.
        
        
        """
        Construct neural network layers from subclasses
        """
        
        self.encoder = Coder("Encoder", self.encoder_struct, self.input_size, self.latent_size)
        self.decoder = Coder("Decoder", self.decoder_struct, self.latent_size, self.output_size)
        
        self.dist = torch.distributions.Normal(loc=0.0, scale=1.0)

    """
    the forward function is called each the the network is propagating forward
    takes in input data and spits out the predicted output
    """
    
    def forward(self, input_data):
        #print("Encoding...\n")
        latent_data = self.encoder(input_data)
        #print(latent_data)
        
        #print("Sampling...\n")
        self.latent_mu = latent_data[:,:int(self.latent_size/2)]
        self.log_latent_sigma = latent_data[:,int(self.latent_size/2):]
        
        #This creates a node through which backpropagation does not work
        #dist = torch.distributions.Normal(self.latent_mu, torch.exp(self.latent_sigma))
        #latent_sample = dist.sample((1,))
        
        #Use the reparametrization trick!
        latent_z = self.dist.sample(self.latent_mu.size()).to(self.device)
        #print(f"latenz_z {latent_z}")
        #print(f"mu {self.latent_mu}")
        #print(f"log_sigma {self.log_latent_sigma}")
        #print(f"sigma {torch.exp(self.log_latent_sigma)}")
        
        latent_sample = self.latent_mu + torch.exp(0.5*self.log_latent_sigma)*latent_z
        #print(latent_sample)
        
        #print("Decoding...\n")
        x = self.decoder(latent_sample)
        
        #x = self.decoder(latent_data)
        return x
    
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
class Coder(AutoEncoder):
    
    def __init__(self, name, net_struct, input_size, latent_size):
        super(AutoEncoder, self).__init__()
        self.name = name
        #self.net_struct = net_struct
        self.input_size = input_size
        self.output_size = latent_size
        self.batch_size = 1
        
        self.set_layers(net_struct)
        
        """
        self.layer_sizes = calc_layer_sizes(self.input_size, self.net_struct) #inherit from utils
        self.layers = nn.ModuleList()
        
        print(f"Initializing {self.name}:\n")
        for layer in self.net_struct:
            print(f"Adding {layer}\n")
            self.layers.append(layer["type"](**layer["layer_pars"]))
        """    
        self.init_weights(torch.nn.init.xavier_normal_)  #inherit from AutoEncoder
        

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
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

