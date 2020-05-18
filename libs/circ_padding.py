import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""Circular Padding"""

class CircularPadding(nn.Module):
    
    def __init__(self, padding=1):
        super(CircularPadding, self).__init__()
        
        #self.padding = padding
        #batch axis
        self.padding = np.concatenate(([0],padding))
        
        self.dims = len(self.padding)

            
    def forward(self,x):
        
        padded_x = x
        
        #for d in dims:
        for d in range(1,self.dims):
            #print(d)
            
            pad_l = self.padding[d]
            if self.padding[d] != 0:
                end_strip = padded_x.narrow(d, 0, int(pad_l)).clone()
                start_strip = padded_x.narrow(d, int(-pad_l), int(pad_l)).clone()
                
                #print(padded_x.size())
                #print(end_strip.size())
                #print(start_strip.size())
                
                padded_x = torch.cat((start_strip, padded_x, end_strip), dim=d)
                
                #print(padded_x.size())
        
        return padded_x

#old implementation
"""       
class CircularPadding(nn.Module):
    
    def __init__(self, padding=1):
        super(CircularPadding, self).__init__()
        
        #self.padding = padding
        #batch axis
        self.padding = np.concatenate(([0],padding))
        
        self.dims = len(self.padding)

            
    def forward(self,x):
        
        padded_x = x
        
        #for d in dims:
        for d in range(1,self.dims):
            #print(d)
            
            pad_l = self.padding[d]
            if self.padding[d] != 0:
                end_strip = padded_x.narrow(d, 0, int(pad_l)).clone()
                start_strip = padded_x.narrow(d, int(-pad_l), int(pad_l)).clone()
                
                #print(padded_x.size())
                #print(end_strip.size())
                #print(start_strip.size())
                
                padded_x = torch.cat((start_strip, padded_x, end_strip), dim=d)
                
                #print(padded_x.size())
        
        return padded_x
"""