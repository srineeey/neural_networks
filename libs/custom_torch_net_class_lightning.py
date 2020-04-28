"""Neural Network class"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch import cat
from torch import stack

import pytorch_lightning as pl

from utils import *
#import matplotlib.pyplot as plt

"""
the neural network class inherits from nn.Module
the default functions
init (constructor) and forward
have to be defined manually
"""

#class CustomNet(nn.Module):
class CustomNet(pl.LightningModule):
    """
    class constructor
    typically the neural networks variables and structure are initialized here
    super().__init__() is necessary to call the mother class constructor
    """
    
    #def __init__( self, hparams, dataset, cnn_struct, classifier1_struct, classifier2_struct, input_size, fc_input_size, output_size, device):
    def __init__(self, hparams):
                
        super(CustomNet, self).__init__()
        
        """ADD HYPER PARS"""
        
        self.hparams = hparams
        
        #self.name = "CustomNet"
        self.name = self.hparams["name"]
        
        self.loss_type = self.hparams["loss"]
        self.optimizer_type = self.hparams["optimizer"]
        
        self.optimizer_kwargs = self.hparams["optimizer_kwargs"]
        self.loss_kwargs = self.hparams["loss_kwargs"]
        
        self.train_sampler = []
        self.val_sampler = []
        self.test_sampler = []
        
        self.optimizer = None
        self.loss = None
        
        self.dataset = None
        #self.dataset = dataset
        #self.dataset = hparams["dataset"]
        
        
        #self.cnn_struct = None
        #self.classifier1_struct = None
        #self.classifier2_struct = None
        #self.cnn_struct = cnn_struct
        #self.classifier1_struct = classifier1_struct
        #self.classifier2_struct = classifier2_struct
        self.cnn_struct = self.hparams["cnn_struct"]
        self.classifier1_struct = self.hparams["classifier1_struct"]
        #self.classifier2_struct = self.hparams["classifier2_struct"]
        
        
        #self.input_size = None
        #self.fc_input_size = None
        #self.output_size = None
        #self.input_size = input_size
        #self.fc_input_size = fc_input_size
        #self.output_size = output_size
        self.input_size = self.hparams["input_size"]
        self.fc_input_size = self.hparams["fc_input_size"]
        self.output_size = self.hparams["output_size"]
        
        #self.device = None
        #self.device = device
        #self.device = hparams["device"]
        
        self.tb_logger = None
        
        """
        Construct neural network layers from subclasses
        """
        self.cnn = Net("CNN", self.cnn_struct, self.input_size, self.fc_input_size)
        
        #self.dense1 = Net("DenseClassifier1", self.classifier1_struct, self.fc_input_size, 1)
        self.dense1 = Net("DenseClassifier1", self.classifier1_struct, self.fc_input_size, 2)
        #self.dense2 = Net("DenseClassifier2", self.classifier2_struct, self.fc_input_size, 1)
        
    """
    the forward function is called each the the network is propagating forward
    takes in input data and spits out the predicted output
    """
    
    def forward(self, input_data):
        """
        fc_input = self.cnn(input_data)
        
        output1 = self.dense1(fc_input)
        output2 = self.dense2(fc_input)
        
        x = torch.cat((output1, output2),1)
        """
        fc_input = self.cnn(input_data)
        
        output1 = self.dense1(fc_input)
        #output2 = self.dense2(fc_input)
        
        x = output1
        return x
    
    """
    initialize weights
    Using initialization routine from the torch.nn.init package
    """
    
    def init_weights(self, init_routine):
        #print(f"Initializing weights of {self.name} with method {init_routine}\n")
        for i, layer in enumerate(self.layers):
            if type(layer) == nn.Linear:
                #torch.nn.init.xavier_normal_(layer.weight)
                init_routine(layer.weight)
                #if self.net_struct[i]["bias"] == True:
                    #layer.bias.data.fill_(0.01)


    def set_layers(self, net_struct):
        self.net_struct = net_struct
        self.layer_sizes = calc_layer_sizes(self.input_size, self.net_struct) #inherit from utils
        self.layers = nn.ModuleList()
        
        #print(f"Initializing {self.name}:\n")
        for layer in self.net_struct:
            #print(f"Adding {layer}\n")
            self.layers.append(layer["type"](**layer["layer_pars"]))
            
            
    def prepare_dataset_splits(self, dataset, split_indices):
        
        self.dataset = dataset
        
        self.train_sampler = SubsetRandomSampler(split_indices["train"])
        self.val_sampler = SubsetRandomSampler(split_indices["val"])
        self.test_sampler = SubsetRandomSampler(split_indices["test"])
            
    """Pytorch Lightning methods"""
    
        
    
    def prepare_data(self):
        
        pass
    


    def train_dataloader(self):
        self.train_loader = torch.utils.data.DataLoader(dataset=self.dataset,
                                           batch_size=self.hparams['bs'],
                                           sampler=self.train_sampler, num_workers=4)
        return self.train_loader

    
    def val_dataloader(self):
        self.val_loader = torch.utils.data.DataLoader(dataset=self.dataset,
                                           batch_size=self.hparams['bs'],
                                           sampler=self.val_sampler, num_workers=4)
        return self.val_loader
    
    def test_dataloader(self):
        self.test_loader = torch.utils.data.DataLoader(dataset=self.dataset,
                                           batch_size=self.hparams['bs'],
                                           sampler=self.test_sampler, num_workers=4)
        return self.test_loader
    
    
    def configure_optimizers(self):
        self.optimizer = self.optimizer_type(self.parameters(), lr=self.hparams["lr"], **self.hparams["optimizer_kwargs"])
        
        return self.optimizer
    
    def configure_loss(self):
        self.loss = self.loss_type(**self.hparams["loss_kwargs"])
        
        return self.loss

    
    def training_step(self, batch, batch_idx):
        
        feat_batch, label_batch = batch

        model_input_shape = tuple(np.concatenate( ([-1],self.input_size)) )
        model_output_shape = tuple(np.concatenate( ([-1],self.output_size)) )
        
        output = self(feat_batch.view(model_input_shape).float())
        loss = self.loss(output.view(model_output_shape).float(), label_batch.view(model_output_shape).float())
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        
        feat_batch, label_batch = batch

        model_input_shape = tuple(np.concatenate( ([-1],self.input_size)) )
        model_output_shape = tuple(np.concatenate( ([-1],self.output_size)) )
        
        output = self(feat_batch.view(model_input_shape).float())
        loss = self.loss(output.view(model_output_shape).float(), label_batch.view(model_output_shape).float())
        return {'val_loss': loss}
    
    def test_step(self, batch, batch_idx):
        
        feat_batch, label_batch = batch

        model_input_shape = tuple(np.concatenate( ([-1],self.input_size)) )
        model_output_shape = tuple(np.concatenate( ([-1],self.output_size)) )
        
        output = self(feat_batch.view(model_input_shape).float())
        loss = self.loss(output.view(model_output_shape).float(), label_batch.view(model_output_shape).float())
        return {'test_loss': loss}
    
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': logs}
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': logs}



class Net(CustomNet):
    
    def __init__(self, name, net_struct, input_size, output_size):
        super(CustomNet, self).__init__()
        self.name = name
        #self.net_struct = net_struct
        self.input_size = input_size
        self.output_size = output_size
        
        self.set_layers(net_struct)
        
        """
        self.layer_sizes = calc_layer_sizes(self.input_size, self.net_struct) #inherit from utils
        self.layers = nn.ModuleList()
        
        print(f"Initializing {self.name}:\n")
        for layer in self.net_struct:
            print(f"Adding {layer}\n")
            self.layers.append(layer["type"](**layer["layer_pars"]))
        """    
        self.init_weights(torch.nn.init.xavier_normal_)
        

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

        return x
    
    
        
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

