import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys

from sklearn.model_selection import train_test_split,KFold

from torch import reshape
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable


import misc_modules
import conv4d
import circ_padding


#from misc_modules import *
#from conv4d import *
#from circ_padding import *

import matplotlib.pyplot as plt

def np_complex_to_channel(x, channel_axis=0):
    re_x = x.real
    im_x = x.imag

    return np.concatenate((re_x, im_x), axis=channel_axis)
    
"""
#Return splitted data in the form of a list of Pytorch DataLoaders
#method = "holdout": method _pars = {"train" : 0.8, "val" : 0.1, "test" : 0.}
#test_size can be set to 0.
#returns [train_loader, val_loader, test_loader] list
#method = "kfold": method _pars = k
#returns [[train_indices1, test_indices2],[...],...]
"""

def load_split_data(dataset, batch_size=1, method="kfold", method_pars=None, random_seed=42, log_file=None):
    if method == "holdout":
        all_indices = list(range(len(dataset)))
        #all_indices = list(range(dataset.get_length()))
        split_ratio = method_pars
        
        if split_ratio["test"] == 0.:
            remaining_indices = all_indices
            
            test_sampler = None
            test_loader = None
            
        else:
            rem_test_ratio = (split_ratio["train"] + split_ratio["val"])/split_ratio["test"]
            remaining_indices, test_indices = train_test_split(all_indices, test_size=rem_test_ratio, shuffle=True, random_state=random_seed)
            
            test_sampler = SubsetRandomSampler(test_indices)
            test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
            print(f"size of test set :{len(test_loader)}\n")
            
            
        val_train_ratio = split_ratio["val"]/split_ratio["train"]
        train_indices, val_indices = train_test_split(remaining_indices, test_size=val_train_ratio, shuffle=True, random_state=random_seed)
        
        """UNCONTROLLED RANDOMNESS?"""
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
        print(f"size of val set :{len(val_loader)}\n")
        print(f"size of train set :{len(train_loader)}\n")
        
            
        return [[train_loader, val_loader, test_loader]]
    elif method == "kfold":
        """NOT WORKING ATM"""
        k = method_pars
        kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)
        kfold_indices = kf.split(all_indices)
        
        data_loader_list = []
        
        for indices in kfold_indices:
            train_indices = indices[0]
            val_indices = indices[1]
            
            """UNCONTROLLED RANDOMNESS?"""
            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)

            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
            val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
            
            data_loader_list.append([train_loader,val_loader])
            
        
        return data_loader_list
    
def load_split_indices(dataset, batch_size=1, method="kfold", method_pars=None, shuffle=True, random_seed=42, log_file=None):
    if method == "holdout":
        all_indices = list(range(len(dataset)))
        #all_indices = list(range(dataset.get_length()))
        split_ratio = method_pars
        
        if split_ratio["test"] == 0.:
            remaining_indices = all_indices
            test_indices = []
            
        else:
            rem_test_ratio = split_ratio["test"]/(split_ratio["train"] + split_ratio["val"])
            remaining_indices, test_indices = train_test_split(all_indices, test_size=rem_test_ratio, shuffle=shuffle, random_state=random_seed)
            print(f"size of test set :{len(test_indices)}\n")
            
            
        val_train_ratio = split_ratio["val"]/split_ratio["train"]
        train_indices, val_indices = train_test_split(remaining_indices, test_size=val_train_ratio, shuffle=shuffle, random_state=random_seed)
        
        print(f"size of val set :{len(val_indices)}\n")
        print(f"size of train set :{len(train_indices)}\n")
        
        split_indices = {"train": train_indices, "val": val_indices, "test": test_indices} 
        return split_indices    
        #return [[train_indices, val_indices, test_indices]]
#    elif method == "kfold":
#        """NOT WORKING ATM"""
#        k = method_pars
#        kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_seed)
#        kfold_indices = kf.split(all_indices)
#        return kfold_indices
    
def split_shuffle_indices(all_indices, fractions=[0.5,0.5], shuffle=True, random_seed=42, log_file=None):
    
    np.random.seed(random_seed)
    
    index_list = []
    
    norm = np.sum(fractions)
    if norm != 1.:
        for i, fraction in enumerate(fractions):
            fractions[i] = fraction/norm

    print(f"splitting into fraction {fractions}")
    
    num_indices = len(all_indices)
    print(f"number of indices: {num_indices}")
    
    if shuffle == True:
        np.random.shuffle(all_indices)
    
    start = 0
    for i in range(len(fractions)):
        stop = int(start + fractions[i] * num_indices)
        print(f"slicing from {start} to {stop}")
        index_list.append( all_indices[start:stop] )
        start = stop  
            
    return index_list

"""
Function for actual train/val step, standard NNs
"""
def step(model, input_batch_feat, input_batch_label, loss_func, optimizer, device, mode="val", log_file=None):
        #switch modes train and val so that in the latter case gradients are not calculated (unnecessary!)
        if mode == "train":
            model.train()
        elif mode == "val":
            model.eval()

        feat_batch = input_batch_feat.to(device)
        label_batch = input_batch_label.to(device)
    
        """ALWAYS SET GRADIENT TO ZERO  FOR STANDARD NN (NOT RNNs)"""
        model.zero_grad()
        optimizer.zero_grad()
        
        input_size = model.get_input_size()
        output_size = model.get_output_size()
        #print(input_size)
        #print(output_size)
        
        model_input_shape = tuple(np.concatenate(([-1],input_size)))
        output = model(feat_batch.view(model_input_shape).float())
        output = output.to(device)
        
        model_output_shape = tuple(np.concatenate(([-1],output_size)))
        
        """SWITCH BETWEEN TYPES OF LOSS FUNC"""
        """MSE"""
        loss = loss_func(output.view(model_output_shape).float(), label_batch.view(model_output_shape).float())
        """Cross Entropy"""
        #loss = loss_func(output.view(model_output_shape).float(), label_batch.view(-1).long())
        
        
        if mode == "train":
            loss.backward()
            optimizer.step()
    
        return loss, output

"""
Function for actual train/val step, VAEC
"""
def vstep(model, input_batch_feat, input_batch_label, loss_func, gamma, optimizer, device, mode="val", log_file=None):
        
        #switch modes train and val so that in the latter case gradients are not calculated (unnecessary!)
        if mode == "train":
            model.train()
        elif mode == "val":
            model.eval()

        feat_batch = input_batch_feat.to(device)
        label_batch = input_batch_label.to(device)
    
        #ALWAYS SET GRADIENT TO ZERO  FOR STANDARD NN (NOT RNNs)
        model.zero_grad()
        #optimizer.zero_grad()
        
        input_size = model.get_input_size()
        output_size = model.get_output_size()
        #print(input_size)
        #print(output_size)
        
        model_input_shape = tuple(np.concatenate(([-1],input_size)))
        output = model(feat_batch.view(model_input_shape).float())
        output = output.to(device)
        
        latent_mu, log_latent_sigma = model.get_latent_variables()
        #print(f"latent_mu {latent_mu}")
        #print(f"latent_sigma {log_latent_sigma}")
        
        model_output_shape = tuple(np.concatenate(([-1],output_size)))
        
        loss = vaec_loss(output, model_output_shape, label_batch, model_input_shape, model.get_latent_variables(), loss_func, gamma=gamma)
        
        if mode == "train":
            loss.backward()
            optimizer.step()
    
        return loss, output
    


"""VAEC Loss function"""
def vaec_loss(prediction, prediction_shape, label, label_shape, latent_variables, loss_func, gamma=1.0):
    
    latent_mu, log_latent_sigma = latent_variables
    
    #Reconstruction loss
    #loss = loss_func(prediction.view(prediction_shape).float(), label.view(label_shape).float())
    rec_loss = loss_func(prediction.view(prediction_shape).float(), label.view(label_shape).float())
    
    #KL divergence
    #loss += gamma * torch.sum( - log_latent_sigma + latent_mu.pow(2) + log_latent_sigma.exp() )
    kl_loss = gamma * torch.sum( - log_latent_sigma + latent_mu.pow(2) + log_latent_sigma.exp() )
    
    loss = rec_loss + kl_loss
    
    return loss







def d_train(discriminator, generator, loss_func, discriminator_optimizer, data, device):
    """Train discriminator"""
    ###net.discriminator.zero_grad()
    discriminator.zero_grad()

    # train discriminator on real
    data_real = data
    data_real = Variable(data_real.to(device))
    
    ###discriminator_output = net.discriminator(data_real)
    discriminator_output = discriminator(data_real)
    
    label_real = torch.ones(discriminator_output.size())
    label_real = Variable( label_real.to(device) )
    
    discriminator_real_loss = loss_func(discriminator_output, label_real)
    #discriminator_real_score = discriminator_output

    # train discriminator on fake
    #latent_z = Variable( torch.randn(batch_size,latent_size).to(device) )
    #latent_z = Variable( latent_dist.sample(tuple(latent_mini_batch_shape)).to(device) )
    latent_z = Variable( generator.latent_dist.sample(tuple(latent_mini_batch_shape)).to(device) )
    ###data_fake =  net.generator(latent_z)
    data_fake =  generator(latent_z)
    label_fake = Variable( torch.zeros(batch_size, 1).to(device) )
    
    ###discriminator_output =  net.discriminator(data_fake)
    discriminator_output =  discriminator(data_fake)
    discriminator_fake_loss = loss_func(discriminator_output, label_fake)
    discriminator_fake_score = discriminator_output

    #backpropagation and optimization of discriminator
    discriminator_loss = discriminator_real_loss + discriminator_fake_loss
    discriminator_loss.backward()
    discriminator_optimizer.step()
        
    return  discriminator_loss.data.item()

def g_train(discriminator, generator, loss_func, generator_optimizer, device):
    """Train generator"""
    ###net.generator.zero_grad()
    generator.zero_grad()

    #latent_z = Variable( torch.randn(batch_size,latent_size).to(device) )
    #latent_z = Variable( latent_dist.sample(tuple(latent_mini_batch_shape)).to(device) )
    latent_z = Variable( generator.latent_dist.sample(tuple(latent_mini_batch_shape)).to(device) )
    label_real = Variable( torch.ones(batch_size, 1).to(device) )

    ###generator_output = net.generator(latent_z)
    generator_output = generator(latent_z)
    ###discriminator_output = net.discriminator(generator_output)
    discriminator_output = discriminator(generator_output)
    generator_loss = loss_func(discriminator_output, label_real)

    #backpropagation and optimization of generator
    generator_loss.backward()
    generator_optimizer.step()
    
    return generator_loss.data.item(), generator_output








"""
function for calculating neuron activation layer sizes.
needed for torch_net_class
needs to be updated each time a new type of layer is being introduced
net_struct = list of dictionaries with layer attributes and parameters
"""
    
def calc_layer_sizes(input_shape, net_struct, log_file=None):
    #layer_sizes will have shape
    #[[channels,length,width],...,dense_neurons,...]
    layer_sizes = [input_shape]
    #print(f"input_shape: {input_shape}")
    
    #go through each layer
    for i in range(len(net_struct)):
        new_layer_size = []
        #print(i)
        #print(net_struct[i]["type"])
        
        if net_struct[i]["type"] == nn.Linear:
            new_layer_size = net_struct[i]["layer_pars"]["out_features"]
            
        elif net_struct[i]["type"] == nn.Flatten:
            new_layer_size = int(np.prod(layer_sizes[-1]))
        
        elif net_struct[i]["type"] == misc_modules.Reshape:
            new_layer_size = net_struct[i]["layer_pars"]["new_shape"]
            
        elif net_struct[i]["type"] == misc_modules.NpSplitReImToChannel:
            channel = net_struct[i]["layer_pars"]["channel_axis"]
            prev_layer_size = layer_sizes[-1].copy()
            new_layer_size = prev_layer_size
            new_layer_size[channel] *= 2
            
        elif net_struct[i]["type"] == misc_modules.PermuteAxes:
            perm = net_struct[i]["layer_pars"]["perm"]
            prev_layer_size = layer_sizes[-1].copy()
            new_layer_size = prev_layer_size[perm]
            
        elif net_struct[i]["type"] == circ_padding.CircularPadding:
            perm = net_struct[i]["layer_pars"]["padding"]
            prev_layer_size = layer_sizes[-1].copy()
            new_layer_size = prev_layer_size
            
            for layer_d in range(len(prev_layer_size)):
                new_layer_size[layer_d] += 2*net_struct[i]["layer_pars"]["padding"][layer_d]

            
        #elif (net_struct[i]["type"] == nn.Conv2d) or (net_struct[i]["type"] == nn.MaxPool2d):
        #elif net_struct[i]["type"] in [nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d]:
        elif net_struct[i]["type"] in [nn.Conv1d, nn.Conv2d, nn.Conv3d, conv4d.Conv4d, nn.MaxPool2d, nn.AvgPool2d]:
        #elif net_struct[i]["type"] in [nn.Conv1d, nn.Conv2d, nn.Conv3d, Conv4d, nn.MaxPool2d, nn.AvgPool2d]:
            
            kernel_shape = net_struct[i]["layer_pars"]["kernel_size"]
            
            #if net_struct[i]["type"] == nn.Conv2d:
            if net_struct[i]["type"] in [nn.Conv1d, nn.Conv2d, nn.Conv3d, conv4d.Conv4d]:
                stride = [1 for i in range(len(kernel_shape))]
            else:
                stride = kernel_shape
            
            padding_mode = "zeros"
            padding = [0 for i in range(len(kernel_shape))]
            dilation = [1 for i in range(len(kernel_shape))]
            
            if "stride" in net_struct[i]["layer_pars"]:
                stride = net_struct[i]["layer_pars"]["stride"]
                
            if "padding" in net_struct[i]["layer_pars"]:
                padding = net_struct[i]["layer_pars"]["padding"]

            if "padding_mode" in net_struct[i]["layer_pars"]:
                padding_mode = net_struct[i]["layer_pars"]["padding_mode"]
                
                #circular padding required specific values
                #bug of pytorch
                #works only for odd kernel sizes!!!
                if padding_mode == "circular":
                    padding = [int((kernel_l-1))//2 for kernel_l in kernel_shape]
                    
                    if not(np.all(padding == net_struct[i]["layer_pars"]["padding"])):
                           print("padding size possibly incorrect!")
            
            if "dilation" in net_struct[i]["layer_pars"]:
                dilation = net_struct[i]["layer_pars"]["dilation"]
                
            #print(kernel_shape)
            #new_layer_size = []
            
            for d in range(len(kernel_shape)):
                #get length of the previous layer in dimension d
                #layer_sizes[n][0] = number of channels!
                #print(f"last layer {layer_sizes[-1]}")
                prev_layer_l = int(layer_sizes[-1][d+1])
                kernel_l = int(kernel_shape[d])
                
                #padding_l = int(padding[d])
                
                if type(stride) == int:
                    stride_l = stride
                elif type(stride) == list:
                    stride_l = int(stride[d])
                    
                if type(padding) == int:
                    padding_l = padding
                elif type(padding) == list:
                    padding_l = int(padding[d])
                    
                if type(dilation) == int:
                    dilation_l = dilation
                elif type(dilation) == list:
                    dilation_l = int(dilation[d])
                    
                if (prev_layer_l + 2*padding_l - dilation_l*(kernel_l - 1) - 1) % stride_l != 0:
                    print(f"Input {layer_sizes[-1]} maybe not compatible with:")
                    print(net_struct[i]['layer_pars'])
                    #raise ValueError(f'Input {layer_sizes[-1]}, kernel {kernel_shape}, stride {stride} and padding {padding} in layer {i} not compatible!')

                #actual computation

                new_layer_size_l = int(np.floor( (prev_layer_l + 2*padding_l - dilation_l*(kernel_l - 1) - 1)/(stride_l) + 1.))
                new_layer_size.append( new_layer_size_l )
                
                #print(f"new layer {new_layer_size}")

            #if net_struct[i]["type"] == nn.Conv2d:
            if net_struct[i]["type"] in [nn.Conv1d, nn.Conv2d, nn.Conv3d, conv4d.Conv4d]:
                new_layer_size = [net_struct[i]["layer_pars"]["out_channels"]] + new_layer_size
                
            elif net_struct[i]["type"] in [nn.MaxPool2d, nn.AvgPool2d]:
                prev_channels = layer_sizes[-1][0]
                new_layer_size = [prev_channels] + new_layer_size
                
        elif net_struct[i]["type"] == nn.ConvTranspose2d:
            kernel_shape = net_struct[i]["layer_pars"]["kernel_size"]
            stride = net_struct[i]["layer_pars"]["stride"]
            padding = net_struct[i]["layer_pars"]["padding"]
            
            #new_layer_size = []
            for d in range(len(kernel_shape)):
                #get length of the previous layer in dimension d
                #layer_sizes[n][0] = number of channels!
                prev_layer_l = int(layer_sizes[-1][d+1])
                kernel_l = int(kernel_shape[d])
                if type(stride) == int:
                    stride_l = stride
                elif type(stride) == list:
                    stride_l = int(stride[d])
                    
                #actual conputation
                #for dilation = 1
                new_layer_size.append( (prev_layer_l - 1)*stride_l + kernel_l - 2*padding)
                
                #new_layer_size.append( (prev_layer_l - 1)*stride_l - 2*padding + 1)
            
            new_layer_size = [net_struct[i]["layer_pars"]["out_channels"]] + new_layer_size
        
        #elif (net_struct[i]["type"] == nn.BatchNorm1d) or (net_struct[i]["type"] == nn.Dropout) or (net_struct[i]["type"] == nn.Softmax):
        
        elif net_struct[i]["type"] in [nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]:
            """Append channel sizes from last layer"""
            new_layer_size = layer_sizes[-1].copy()
            d = len(net_struct[i]["layer_pars"]["output_size"])
            new_layer_size[-d:] = net_struct[i]["layer_pars"]["output_size"][:]
        
        elif net_struct[i]["type"] in [nn.BatchNorm1d, nn.Dropout, nn.Softmax]:
            new_layer_size = layer_sizes[-1]
        
        else:
            #print("custom layer operation not defined, assuming previous layer_size")
            new_layer_size = layer_sizes[-1]
        
        #append newly calculated neuron activation shape to layer_sizes
        if np.any(np.array(new_layer_size) <= 0):
            #raise ValueError(f'Negative layer size found in {new_layer_size}!')
            print(f'Negative layer size found in {new_layer_size}!')
            
        layer_sizes.append(new_layer_size)
    
    return layer_sizes