import torch.nn as nn
import numpy as np
from torch import reshape
from sklearn.model_selection import train_test_split,KFold
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class Reshape(nn.Module):
    def __init__(self, new_shape=[-1]):
        super(Reshape, self).__init__()
        self.shape = new_shape

    def forward(self, x):
        #return x.view(self.shape)
        return reshape(x, tuple(self.shape))

"""
Return splitted data in the form of a list of Pytorch DataLoaders
method = "holdout": method _pars = {"train" : 0.8, "val" : 0.1, "test" : 0.}
test_size can be set to 0.
returns [train_loader, val_loader, test_loader] list
method = "kfold": method _pars = k
returns [[train_indices1, test_indices2],[...],...]
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

"""
Function for actual train/val step
"""

def step(model, input_batch, output_size, loss_func, optimizer, device, mode="val", log_file=None):
        #switch modes train and val so that in the latter case gradients are not calculated (unnecessary!)
        if mode == "train":
            model.train()
        elif mode == "val":
            model.eval()
        
        #input data splitted into features and labels
        feat_batch = input_batch[0].to(device)
        #label_batch = input_batch[1].to(device)
        """FOR AUTOENCODER"""
        label_batch = input_batch[0].to(device)

        #print(feat_batch.shape)
        #print(label_batch.shape)
    
        """ALWAYS SET GRADIENT TO ZERO  FOR STANDARD NN (NOT RNNs)"""
        model.zero_grad()
        optimizer.zero_grad()
        
        input_size = model.get_input_size()
        
        """
        the peculiar shape (-1, sample_size) is needed, because an entire mini batch is passed on to the network
        initially it is not clear how large such a mini batch is
        the -1 acts as a placeholder in order to keep the number of processed items in one mini batch flexible
        """
        
        model_input_shape = tuple([-1] + list(input_size))
        output = model(feat_batch.view(model_input_shape).float())
        output = output.to(device)
        
        model_output_shape = tuple(np.concatenate(([-1],output_size)))
        
        """SWITCH BETWEEN TYPES OF LOSS FUNC"""
        """MSE"""
        #loss = loss_func(output.view(-1, output_size).float(), label_batch.view(-1, output_size).float())
        #loss = loss_func(output.view(model_output_shape).float(), label_batch.view(-1, output_size).float())
        loss = loss_func(output.view(model_output_shape).float(), label_batch.view(model_output_shape).float())
        """Cross Entropy"""
        #loss = loss_func(output.view(model_output_shape).float(), label_batch.view(-1).long())
        
        
        if mode == "train":
            loss.backward()
            optimizer.step()
    
        return loss, output
    

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
    
    #go through each layer
    for i in range(len(net_struct)):
        new_layer_size = []
        #print(i)
        #print(layer_sizes[-1])
        
        if net_struct[i]["type"] == nn.Linear:
            new_layer_size = net_struct[i]["layer_pars"]["out_features"]
            
        elif net_struct[i]["type"] == nn.Flatten:
            new_layer_size = int(np.prod(layer_sizes[-1]))
        
        elif net_struct[i]["type"] == Reshape:
            new_layer_size = net_struct[i]["layer_pars"]["new_shape"]
            
        #elif (net_struct[i]["type"] == nn.Conv2d) or (net_struct[i]["type"] == nn.MaxPool2d):
        elif net_struct[i]["type"] in [nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d]:
            kernel_shape = net_struct[i]["layer_pars"]["kernel_size"]
            stride = net_struct[i]["layer_pars"]["stride"]
            padding = net_struct[i]["layer_pars"]["padding"]
            #print(kernel_shape)
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
                if (prev_layer_l - kernel_l) % stride_l == 0:
                    new_layer_size.append( (prev_layer_l + 2*padding - kernel_l)//stride_l + 1 )
                else:
                    raise ValueError(f'Input {layer_sizes[-1]}, kernel {kernel_shape}, stride {stride} and padding {padding} not compatible!')
            
            if net_struct[i]["type"] == nn.Conv2d:
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
                new_layer_size.append( (prev_layer_l - 1)*stride_l + kernel_l - 2*padding)
            
            new_layer_size = [net_struct[i]["layer_pars"]["out_channels"]] + new_layer_size
        
        #elif (net_struct[i]["type"] == nn.BatchNorm1d) or (net_struct[i]["type"] == nn.Dropout) or (net_struct[i]["type"] == nn.Softmax):
        elif net_struct[i]["type"] in [nn.BatchNorm1d, nn.Dropout, nn.Softmax]:
                new_layer_size = layer_sizes[-1]
        
        #append newly calculated neuron activation shape to layer_sizes
        layer_sizes.append(new_layer_size)
    
    return layer_sizes

