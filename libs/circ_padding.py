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
        #print(f"padding {self.padding}")
        
        self.dims = len(self.padding)
        
        #slightly different implementation
        #removes all dimensions that are not padded
        #slightly faster?
        
        #print(f"np.where(self.padding != 0) {np.where(self.padding != 0)}")
        #self.non0_padding = np.where(self.padding != 0)[0].tolist()
        self.non0_padding = self.padding[self.padding != 0].tolist()
        #print(f"non0_padding {self.non0_padding}")

        self.padded_dims = np.argwhere(self.padding != 0).flatten().tolist()
        
        #print(f"padded_dims {self.padded_dims}")
        
            
    def forward(self,x):
        
        padded_x = x
        """
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
        """
        
        for i in range(len(self.padded_dims)):
            #print(i)
            
            dim = self.padded_dims[i]
            #print(f"padding dim {dim}")
            
            #pad_l = self.padding[padded_dims[d]]
            pad_l = self.non0_padding[i]
            #print(f"pad_l {pad_l}")
            
            end_strip = padded_x.narrow(dim, 0, int(pad_l)).clone()
            start_strip = padded_x.narrow(dim, int(-pad_l), int(pad_l)).clone()

            #print(padded_x.size())
            #print(end_strip.size())
            #print(start_strip.size())

            padded_x = torch.cat((start_strip, padded_x, end_strip), dim=dim)
                
            #print(padded_x.size())
        
        return padded_x
    
class AsymmetricCircularPadding(nn.Module):
    """
    Asymmetric Circular Padding for general n dimensional tensors
    
    Args:
        padding: a list containing the padding widths for each dimension
        disregarding batch dimension (this is being accounted for in self.padding)
        this also includes the channel dimension as well (which normally should not be padded -> padding = 0)
        the elements in list must be integer and specify the padding per dimension.
        Meaning: padding = [1] means a one dimensional input is padded with addition width of 1 on ONE END.
    
    """
    
    
    def __init__(self, padding=[1]):
        super(CircularPadding, self).__init__()
        
        #self.padding = padding
        #batch axis
        self.padding = np.concatenate(([0],padding))
        #print(f"padding {self.padding}")
        
        self.dims = len(self.padding)
        
        #slightly different implementation
        #removes all dimensions that are not padded
        #slightly faster?
        
        #print(f"np.where(self.padding != 0) {np.where(self.padding != 0)}")
        #self.non0_padding = np.where(self.padding != 0)[0].tolist()
        self.non0_padding = self.padding[self.padding != 0].tolist()
        #print(f"non0_padding {self.non0_padding}")

        self.padded_dims = np.argwhere(self.padding != 0).flatten().tolist()
        
        #print(f"padded_dims {self.padded_dims}")
        
            
    def forward(self,x):
        
        padded_x = x
        """
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
        """
        
        for i in range(len(self.padded_dims)):
            #print(i)
            
            dim = self.padded_dims[i]
            #print(f"padding dim {dim}")
            
            #pad_l = self.padding[padded_dims[d]]
            pad_l = self.non0_padding[i]
            #print(f"pad_l {pad_l}")
            
            end_strip = padded_x.narrow(dim, 0, int(pad_l)).clone()
            #start_strip = padded_x.narrow(dim, int(-pad_l), int(pad_l)).clone()

            #print(padded_x.size())
            #print(end_strip.size())
            #print(start_strip.size())

            #padded_x = torch.cat((start_strip, padded_x, end_strip), dim=dim)
            padded_x = torch.cat((padded_x, end_strip), dim=dim)
                
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