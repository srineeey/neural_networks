import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Reshape(nn.Module):
    def __init__(self, new_shape=[-1]):
        super(Reshape, self).__init__()
        self.shape = new_shape
        self.batch_shape = np.concatenate( ([-1],self.shape) )

    def forward(self, x):
        #return x.view(self.shape)
        #print(tuple( np.concatenate(([-1],self.shape)) ))
        return reshape(x, tuple(self.batch_shape))
    
    
class PermuteAxes(nn.Module):
    def __init__(self, input_shape=[1], ignore_dims=1, perm=None):
        super(PermuteAxes, self).__init__()
        self.input_shape = input_shape
        self.ignore_dims = ignore_dims
        self.perm = perm
        if perm == None:
            perm == self.input_shape[::-1]
        
        #account for batch axis    
        perm += np.full(len(perm),ignore_dims,dtype=int)
        new_perm = np.arange(len(perm)+ignore_dims, dtype=int)
        new_perm[1:] = perm
        self.perm = new_perm
        
    def forward(self, x):
        #x.permute(*self.perm)
        #has to be torch tensor
        #return x
        return x.permute(*self.perm)
    
    
class NpSplitReImToChannel(nn.Module):
    def __init__(self, channel_axis=0):
        super(NpSplitReImToChannel, self).__init__()
        self.channel_axis = channel_axis

    def forward(self, x):
        re_x = x.real
        im_x = x.imag
        torch_x = torch.tensor(np.concatenate((re_x, im_x), axis=self.channel_axis))
        return torch_x
        #return np.concatenate((re_x, im_x), axis=self.channel_axis)
        
        

def np_complex_to_channel(x, channel_axis=0):
    re_x = x.real
    im_x = x.imag

    return np.concatenate((re_x, im_x), axis=channel_axis)