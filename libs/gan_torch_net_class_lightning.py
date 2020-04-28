"""Neural Network class"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np

import pytorch_lightning as pl

from utils import *
#import matplotlib.pyplot as plt

"""
the neural network class inherits from nn.Module
the default functions
init (constructor) and forward
have to be defined manually
"""

class GAN(pl.LightningModule):
    """
    class constructor
    typically the neural networks variables and structure are initialized here
    super().__init__() is necessary to call the mother class constructor
    """
    
    #def __init__(self, generator_struct, discriminator_struct, data_size, latent_size, device):
    def __init__(self, hparams):
        super(GAN, self).__init__()
                
        self.hparams = hparams
        
        #self.name = "GenerativeAdversarialNetwork"
        self.name = self.hparams["name"]
        
        self.loss_type = self.hparams["loss"]
        self.optimizer_type = self.hparams["optimizer"]
        
        self.optimizer_kwargs = self.hparams["optimizer_kwargs"]
        self.loss_kwargs = self.hparams["loss_kwargs"]
        
        self.train_sampler = []
        self.val_sampler = []
        self.test_sampler = []
        
        self.g_optimizer = None
        self.d_optimizer = None
        self.g_loss = None
        self.d_loss = None
        
        self.dataset = None
        #self.dataset = dataset
        #self.dataset = hparams["dataset"]
        
        self.generator_struct = self.hparams["generator_struct"]
        self.discriminator_struct = self.hparams["discriminator_struct"]
        self.data_size = self.hparams["data_size"]
        self.latent_size = self.hparams["latent_size"]
        
        #self.device = device
        #self.tb_logger = None
        
        """
        Construct neural network layers from subclasses
        """
        
        self.generator = Generator("Generator", self.generator_struct, self.latent_size, self.data_size)
        #self.discriminator = Discriminator("Discriminator", self.discriminator_struct, self.data_size, 2)
        self.discriminator = Discriminator("Discriminator", self.discriminator_struct, self.data_size, 1)
        

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

        
    def set_layers(self, net_struct):
        self.net_struct = net_struct
        self.layer_sizes = calc_layer_sizes(self.input_size, self.net_struct) #inherit from utils
        self.layers = nn.ModuleList()
        
        print(f"Initializing {self.name}:\n")
        for layer in self.net_struct:
            print(f"Adding {layer}\n")
            self.layers.append(layer["type"](**layer["layer_pars"]))

        
    def get_latent_variables(self):
        return (self.latent_mu, self.log_latent_sigma)
    
    
    def prepare_dataset_splits(self, dataset, train_indices):
        
        self.dataset = dataset
        
        self.train_sampler = SubsetRandomSampler(train_indices)
            

    """Pytorch Lightning methods"""
    
    def prepare_data(self):
        
        pass
    


    def train_dataloader(self):
        self.train_loader = torch.utils.data.DataLoader(dataset=self.dataset,
                                           batch_size=self.hparams['bs'],
                                           sampler=self.train_sampler, num_workers=4)
        return self.train_loader

    
    
    def configure_optimizers(self):
        self.g_optimizer = self.optimizer_type(self.generator.parameters(), lr=self.hparams["lr"], **self.hparams["optimizer_kwargs"])
        
        self.d_optimizer = self.optimizer_type(self.discriminator.parameters(), lr=self.hparams["lr"], **self.hparams["optimizer_kwargs"])
        
        return [self.g_optimizer, self.d_optimizer], []
    
    def configure_loss(self):
        self.g_loss = self.loss_type(**self.hparams["loss_kwargs"])
        self.d_loss = self.loss_type(**self.hparams["loss_kwargs"])
        
        return (self.g_loss, self.d_loss)

    
    def training_step(self, batch, batch_idx, optimizer_idx):
        data, _ = batch
        self.data = data
        
        bs = data.shape[0]

    
        ##train generator
        if optimizer_idx == 0:
            latent_z = Variable(self.generator.latent_dist.sample( sample_shape=torch.Size([bs, self.latent_size]) ))
            label_real = Variable( torch.ones(bs) )

            generator_output = self.generator(latent_z)
            discriminator_output = self.discriminator(generator_output.float())
            generator_loss = self.g_loss(discriminator_output, label_real)
            
            log_dict = {'g_loss': generator_loss}
            loss_dict = {"loss": generator_loss, "progres_bar": log_dict, "log": log_dict}
            return loss_dict

        #train discriminator
        elif optimizer_idx == 1:
            # train discriminator on real
            data_real = data
            discriminator_output = self.discriminator(data_real.float())
            label_real = Variable( torch.ones(bs) )
            discriminator_real_loss = self.d_loss(discriminator_output, label_real)

            # train discriminator on fake
            latent_z = Variable(self.generator.latent_dist.sample( sample_shape=torch.Size([bs, self.latent_size]) ))
            data_fake =  self.generator(latent_z)
            label_fake = Variable( torch.zeros(bs) )
            discriminator_output =  self.discriminator(data_fake)
            discriminator_fake_loss = self.d_loss(discriminator_output, label_fake)
            discriminator_loss = 0.5*(discriminator_real_loss + discriminator_fake_loss)

            log_dict = {'d_loss': discriminator_loss}
            loss_dict = {"loss": discriminator_loss, "progres_bar": log_dict, "log": log_dict}
            return loss_dict

class Generator(GAN):
    
    def __init__(self, name, net_struct, latent_size, output_size):
        super(GAN, self).__init__()
        self.name = name
        #self.net_struct = net_struct
        self.input_size = latent_size
        self.output_size = output_size
        self.batch_size = 1
        #self.device = device
        self.set_layers(net_struct)
        
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
        self.latent_dist = method(**method_kwargs)

class Discriminator(GAN):
    
    def __init__(self, name, net_struct, input_size, output_size):
        super(GAN, self).__init__()
        self.name = name
        #self.net_struct = net_struct
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = 1
        #self.device = device
        
        self.set_layers(net_struct)
        
        

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
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

