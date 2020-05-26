import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""Circular Padding"""

class CircularPadding(nn.Module):
    """
    Circular Padding for general n dimensional tensors
    
    Args:
        padding: a list containing the padding widths for each dimension
        disregarding batch dimension (this is being accounted for in self.padding)
        this also includes the channel dimension as well (which normally should not be padded -> padding = 0)
        the elements in list must be integer and specify the padding per face.
        Meaning: padding = [1] means a one dimensional input is padded with addition width of 1 on BOTH ENDS.
    
    """
    
    
    def __init__(self, padding=[1]):
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