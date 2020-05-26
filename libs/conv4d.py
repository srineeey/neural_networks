import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""4D Convolution v2"""
"""No padding support in this convolution algorithm"""
"""Part of separate module"""
"""
Currently works only with [batch_index, channel_index, t, x, y, z] format
the four last index blocks are convolved
the 3d slices are cut with respect to the t-axis
"""

class Conv4d(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=1,
                 kernel_size=[1,1,1,1], stride=[1,1,1,1],
                 dilation=[1,1,1,1], groups=1, padding=[0,0,0,0], padding_mode="zeros",
                 bias=True,
                 kernel_initializer=None,
                 bias_initializer=None):
    
        #torch.nn.init.xavier_normal_
        super(Conv4d, self).__init__()
        
        #save all arguments
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        
        #(min_kernel_l, min_dims) = torch.min(self.kernel_size, 0, keepdim=True, out=None)
        
        #idea: separate a 4d convolution into a stack of 3d convolutions
        #intially the 4d kernel is stack split is done in the first dimension
        """CHANGE TO SHORTEST DIMENSION"""
        self.kernel_3d_num = self.kernel_size[0]
        self.kernel_3d_size = self.kernel_size[1:]
        
        self.padding_4d = self.padding[0]
        
        conv_3d_layer_pars = {}
        conv_3d_layer_pars["in_channels"] = self.in_channels
        conv_3d_layer_pars["out_channels"] = self.out_channels
        
        conv_3d_layer_pars["kernel_size"] = self.kernel_3d_size
        """NOT YET IMPLEMENTED"""
        conv_3d_layer_pars["stride"] = self.stride[1:]
        conv_3d_layer_pars["padding"] = self.padding[1:]
        #conv_3d_layer_pars["padding_mode"] = self.padding_mode
        
        """NOT YET IMPLEMENTED"""
        #conv_3d_layer_pars["dilation"] = self.dilation[1:]
        #conv_3d_layer_pars["groups"] = self.groups
        conv_3d_layer_pars["bias"] = self.bias
        
        #initialize conv_3d stack
        self.conv_3d_layers = nn.ModuleList()
        for kernel_3d_i in range(self.kernel_3d_num):
            #conv_3d_layer = nn.Conv3d(**conv3d_layer_pars)
            self.conv_3d_layers.append(nn.Conv3d(**conv_3d_layer_pars))
            
            if self.kernel_initializer != None:
                self.kernel_initializer(self.conv_3d_layers[-1].weight)
            if self.bias_initializer != None:
                self.bias_initializer(self.conv_3d_layers[-1].bias)
                
            

    def forward(self,x):
        
        x_size = np.array(x.size())
        #print(f"input_size: {x_size}")
        
        #calculate the size of the output tensor
        #output_size = np.zeros(len(x_size), dtype=int)
        output_size = x_size.copy()
        """BATCH AXIS?"""
        #output_size[0] = x_size[0]
        #output_size[0] = self.out_channels
        output_size[1] = self.out_channels
        #new_layer_size_l = int(np.floor( (prev_layer_l + 2*padding_l - dilation_l*(kernel_l - 1) - 1)/(stride_l) + 1.))
        """GENERALIZE TO ARBITRARY DIMENSION"""
        for d in range(len(self.kernel_size)):
            #print(output_size[d+1])
            #print(x_size[d+1])
            #print(self.padding[d])
            #print(self.dilation[d])
            #print(self.kernel_size[d])
            #print(self.stride)
            output_size[d+2] = (x_size[d+2] + 2*self.padding[d] - self.dilation[d]*(self.kernel_size[d] - 1) - 1)/(self.stride[d]) + 1
        
        #print(f"output_size: {output_size}")
        
        
        #keep track of the input row number in 4th dimension
        #disregarding padding!
        """CHANGE TO ARBITRARY DIMENSION"""
        #row_4d_num = x_size[1]
        row_4d_num = x_size[2]
        #print(f"row_4d_num: {row_4d_num}")
        
        
        #keep track of the output row number in 4th dimension
        #output_row_4d_num = output_size[1]
        output_row_4d_num = output_size[2]
        #print(f"output_row_4d_num: {output_row_4d_num}")
        
        
        #initialize output tensor
        #output = torch.zeros(tuple(output_size))
        output = torch.zeros(tuple(output_size)).to(torch.device("cuda:0"))
        
        #iterate through every 3d slice of the 4d kernel
        for kernel_3d_i in range(self.kernel_3d_num):
            #print(f"kernel_3d_i: {kernel_3d_i}")
            
            """CHECK FORMULA"""
            #perform a convolution with current 3d kernel slice on ALL of these rows
            rows = [kernel_3d_i, row_4d_num - self.kernel_3d_num + kernel_3d_i + 1]
            
            #print(f"convolving rows: [{rows[0]},{rows[1]})")
            
            #"""CHECK FORMULA"""
            #output_row = row_4d_i - kernel_3d_i
            #print(f"output_row: {output_row}")
            
            #the block with which the 3d conv slice should be convolved
            """GENERALIZE TO ARBITRARY DIMENSION"""
            x_4d_block = x.narrow(2, int(rows[0]), int(rows[1] - rows[0]))
            
            #print(f"slicing 4d block of size: {x_4d_block.size()}")
            
            #swap row axis and channel axis
            #so that convolution can sum up channels
            #but leave rows in 4d untouched
            #x_4d_block.transpose(2,1)
            x_4d_block_t = torch.transpose(x_4d_block, 1, 2)
            #print(f"transposing 4d block to size: {x_4d_block_t.size()}")
            
            #perform the actual operation
            #the input tensor has to have 5 dimensions in order for conv3d to work
            #squeeze the first two dimensions into the batch dimension
            #unsqueezing has to be done later
            #block_output = self.conv3d_layers[kernel_3d_i](x_4d_block_t.view(-1,*x_4d_block_t.size()[2:]))
            #print(f"convolving 4d block of size: {x_4d_block_t.view(-1,*x_4d_block_t.size()[2:]).size()}")
            
            block_output = self.conv_3d_layers[kernel_3d_i](x_4d_block_t.reshape(-1,*x_4d_block_t.size()[2:]))
            #block_output = self.conv_3d_layers[kernel_3d_i](x_4d_block_t.reshape(-1,*x_4d_block_t.size()[2:]).to(torch.device("cuda:0")))
            #print(f"convolving 4d block of size: {x_4d_block_t.reshape(-1,*x_4d_block_t.size()[2:]).size()}")

            #aggregate the block_output
            """UNSQUEEZING BATCH AND ROW DIMENSION CORRECT?"""
            #output[:,:] += block_output
            #output += block_output.view(*output_size).to(torch.device("cpu"))
            output += block_output.view(*output_size)

        
        
        return output
    
    
    
    
    
    
#old implementation   
"""

class MyConv4d(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=1,
                 kernel_size=[1,1,1,1], stride=[1,1,1,1], padding=[0,0,0,0], padding_mode="zeros",
                 dilation=[1,1,1,1], groups=1,
                 bias=True,
                 kernel_initializer=None,
                 bias_initializer=None):
    
        #torch.nn.init.xavier_normal_
        super(MyConv4d, self).__init__()
        
        #save all arguments
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        
        #(min_kernel_l, min_dims) = torch.min(self.kernel_size, 0, keepdim=True, out=None)
        
        #idea: separate a 4d convolution into a stack of 3d convolutions
        #intially the 4d kernel is stack split is done in the first dimension
        ###CHANGE TO SHORTEST DIMENSION
        self.kernel_3d_num = self.kernel_size[0]
        self.kernel_3d_size = self.kernel_size[1:]
        
        conv_3d_layer_pars = {}
        conv_3d_layer_pars["in_channels"] = self.in_channels
        conv_3d_layer_pars["out_channels"] = self.out_channels
        
        conv_3d_layer_pars["kernel_size"] = self.kernel_3d_size
        conv_3d_layer_pars["stride"] = self.stride[1:]
        conv_3d_layer_pars["padding"] = self.padding[1:]
        conv_3d_layer_pars["padding_mode"] = self.padding_mode
        
        #conv3d_layer_pars["dilation"] = self.dilation[1:]
        #conv3d_layer_pars["groups"] = self.groups
        conv_3d_layer_pars["bias"] = self.bias
        
        #initialize conv_3d stack
        self.conv_3d_layers = nn.ModuleList()
        for kernel_3d_i in range(self.kernel_3d_num):
            #conv_3d_layer = nn.Conv3d(**conv3d_layer_pars)
            self.conv_3d_layers.append(nn.Conv3d(**conv_3d_layer_pars))
            
            if self.kernel_initializer != None:
                self.kernel_initializer(self.conv_3d_layers[-1].weight)
            if self.bias_initializer != None:
                self.bias_initializer(self.conv_3d_layers[-1].bias)
                
            

    def forward(self,x):
        
        x_size = np.array(x.size())
        print(f"input_size: {x_size}")
        
        #calculate the size of the output tensor
        #output_size = np.zeros(len(x_size), dtype=int)
        output_size = x_size.copy()
        ###BATCH AXIS?
        #output_size[0] = x_size[0]
        #output_size[0] = self.out_channels
        output_size[1] = self.out_channels
        #new_layer_size_l = int(np.floor( (prev_layer_l + 2*padding_l - dilation_l*(kernel_l - 1) - 1)/(stride_l) + 1.))
        ###GENERALIZE TO ARBITRARY DIMENSION
        for d in range(len(self.kernel_size)):
            #print(output_size[d+1])
            #print(x_size[d+1])
            #print(self.padding[d])
            #print(self.dilation[d])
            #print(self.kernel_size[d])
            #print(self.stride)
            output_size[d+2] = (x_size[d+2] + 2*self.padding[d] - self.dilation[d]*(self.kernel_size[d] - 1) - 1)/(self.stride[d]) + 1
        
        print(f"output_size: {output_size}")
        
        
        #keep track of the input row number in 4th dimension
        ###CHANGE TO ARBITRARY DIMENSION
        #row_4d_num = x_size[1]
        row_4d_num = x_size[2]
        print(f"row_4d_num: {row_4d_num}")
        
        
        #keep track of the output row number in 4th dimension
        #output_row_4d_num = output_size[1]
        output_row_4d_num = output_size[2]
        print(f"output_row_4d_num: {output_row_4d_num}")
        
        
        #initialize output tensor
        output = torch.zeros(tuple(output_size))
        
        #iterate through every 3d slice of the 4d kernel
        for kernel_3d_i in range(self.kernel_3d_num):
            #print(f"kernel_3d_i: {kernel_3d_i}")
            
            ###CAN BE REDUCED?
            #iterate through all rows in the fourth dimension
            #for row_4d_i in range(row_4d_num-self.kernel_3d_num):
            for row_4d_i in range(kernel_3d_i, row_4d_num - self.kernel_3d_num + kernel_3d_i):
            #for row_4d_i in range(row_4d_num):
                #print(f"row_4d_i: {row_4d_i}")
                
                print(f"kernel_3d_i: {kernel_3d_i}, row_4d_i: {row_4d_i}")
            
                ###CHECK FORMULA
                output_row = row_4d_i - kernel_3d_i
                print(f"output_row: {output_row}")
                
                #check whether kernel slice is outside of the input tensor
                ###IF CHECK NECESSARY?

                
                
                ###GENERALIZE TO ARBITRARY DIMENSION
                #x_3d_slice = x[:,row_4d_i]
                x_3d_slice = x[:,:,row_4d_i]
                #print(f"x_3d_slice shape: {x_3d_slice.size()}")
                
                #compute the partial output row with the kernel slice
                #first two dimensions of x are batch size and channels
                ###VIEW NECESSARY?
                partial_output_row = self.conv_3d_layers[kernel_3d_i](x_3d_slice)
                #frame_conv3d = self.conv3d_layers[i](input[:, :, j, :].view(b, c_i, d_i, h_i, w_i))
                
                ###BATCH AND CHANNEL AXES?
                #output[:,output_row] += partial_output_row
                output[:,:,output_row] += partial_output_row
        
        
        return output
        




"""


class Pool4d(nn.Module):
    
    def __init__(self, kernel_size=[1,1,1,1], stride=[1,1,1,1],
                 pooling_mode="average"):
    
        #torch.nn.init.xavier_normal_
        super(Pool4d, self).__init__()
        
        #save all arguments
        
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.pooling_mode = pooling_mode
        
        #(min_kernel_l, min_dims) = torch.min(self.kernel_size, 0, keepdim=True, out=None)
        
        #idea: separate a 4d pooling layer into a stack of 3d pooling layer
        #intially the 4d kernel is stack split is done in the first dimension
        """CHANGE TO SHORTEST DIMENSION"""
        self.kernel_3d_num = self.kernel_size[0]
        self.kernel_3d_size = self.kernel_size[1:]
        
        self.pooling_mode = pooling_mode
        
        if self.pooling_mode == "average":
            self.pool_type = nn.AvgPool3d
        
        pool_3d_layer_pars = {}
        
        pool_3d_layer_pars["kernel_size"] = self.kernel_3d_size
        """NOT YET IMPLEMENTED"""
        pool_3d_layer_pars["stride"] = self.stride[1:]
        
        """NOT YET IMPLEMENTED"""
        #pool_3d_layer_pars["dilation"] = self.dilation[1:]
        
        #initialize conv_3d stack
        self.pool_3d_layers = nn.ModuleList()
        for kernel_3d_i in range(self.kernel_3d_num):
            #conv_3d_layer = nn.Conv3d(**conv3d_layer_pars)
            self.pool_3d_layers.append(self.pool_type(**pool_3d_layer_pars))


    def forward(self,x):
        
        x_size = np.array(x.size())
        #print(f"input_size: {x_size}")
        
        #calculate the size of the output tensor
        #output_size = np.zeros(len(x_size), dtype=int)
        output_size = x_size.copy()
        """BATCH AXIS?"""
        #output_size[0] = x_size[0]
        #output_size[0] = self.out_channels
        #output_size[1] = self.out_channels
        output_size[1] = x_size[1]
        #new_layer_size_l = int(np.floor( (prev_layer_l + 2*padding_l - dilation_l*(kernel_l - 1) - 1)/(stride_l) + 1.))
        """GENERALIZE TO ARBITRARY DIMENSION"""
        for d in range(len(self.kernel_size)):
            #print(output_size[d+1])
            #print(x_size[d+1])
            #print(self.padding[d])
            #print(self.dilation[d])
            #print(self.kernel_size[d])
            #print(self.stride)
            output_size[d+2] = (x_size[d+2] - (self.kernel_size[d] - 1) - 1)/(self.stride[d]) + 1
        
        #print(f"output_size: {output_size}")
        
        
        #keep track of the input row number in 4th dimension
        #disregarding padding!
        """CHANGE TO ARBITRARY DIMENSION"""
        #row_4d_num = x_size[1]
        row_4d_num = x_size[2]
        #print(f"row_4d_num: {row_4d_num}")
        
        
        #keep track of the output row number in 4th dimension
        #output_row_4d_num = output_size[1]
        output_row_4d_num = output_size[2]
        #print(f"output_row_4d_num: {output_row_4d_num}")
        
        
        #initialize output tensor
        output = torch.zeros(tuple(output_size))
        #output = torch.zeros(tuple(output_size)).to(torch.device("cuda:0"))
        
        """CHANGE TO ARBITRARY DIMENSION"""
        stride_4d = self.stride[0]
        
        """AVG INSTEAD OF KERNEL"""
        #iterate through every 3d slice of the 4d kernel
        for kernel_3d_i in range(self.kernel_3d_num):
            #print(f"kernel_3d_i: {kernel_3d_i}")
            
            """CHECK FORMULA"""
            """NOT SUITABLE FOR AVGPOOL"""
            #perform a convolution with current 3d kernel slice on ALL of these rows
            rows = [kernel_3d_i, row_4d_num - self.kernel_3d_num + kernel_3d_i + 1]
            
            #print(f"convolving rows: [{rows[0]},{rows[1]})")
            
            #"""CHECK FORMULA"""
            #output_row = row_4d_i - kernel_3d_i
            #print(f"output_row: {output_row}")
            
            #the block with which the 3d conv slice should be convolved
            """GENERALIZE TO ARBITRARY DIMENSION"""
            """NOT SUITABLE FOR AVGPOOL?"""
            #x_4d_block = x.narrow(2, int(rows[0]), int(rows[1] - rows[0]))
            
            #print(f"slicing 4d block of size: {x_4d_block.size()}")
            
            #stride other than 1 reduces number of rows in convolution
            #stride_rows = range(rows[0], rows[1], stride_4d)
            stride_rows = torch.arange(start=rows[0], end=rows[1], step=stride_4d, dtype=int)
            
            #print(stride_rows)
            
            """GENERALIZE TO ARBITRARY DIMENSION"""
            x_4d_block = torch.index_select(x, 2, stride_rows)
            #x_4d_block = torch.index_select(x_4d_block, 2, stride_rows)
            
            #print(f"slicing strided 4d block of size: {x_4d_block.size()}")
            
            #swap row axis and channel axis
            #so that convolution can sum up channels
            #but leave rows in 4d untouched
            #x_4d_block.transpose(2,1)
            x_4d_block_t = torch.transpose(x_4d_block, 1, 2)
            #print(f"transposing 4d block to size: {x_4d_block_t.size()}")
            
            #perform the actual operation
            #the input tensor has to have 5 dimensions in order for conv3d to work
            #squeeze the first two dimensions into the batch dimension
            #unsqueezing has to be done later
            #block_output = self.conv3d_layers[kernel_3d_i](x_4d_block_t.view(-1,*x_4d_block_t.size()[2:]))
            #print(f"convolving 4d block of size: {x_4d_block_t.view(-1,*x_4d_block_t.size()[2:]).size()}")
            
            block_output = self.pool_3d_layers[kernel_3d_i](x_4d_block_t.reshape(-1,*x_4d_block_t.size()[2:]))
            #block_output = self.conv_3d_layers[kernel_3d_i](x_4d_block_t.reshape(-1,*x_4d_block_t.size()[2:]).to(torch.device("cuda:0")))
            #print(f"convolving 4d block of size: {x_4d_block_t.reshape(-1,*x_4d_block_t.size()[2:]).size()}")

            #aggregate the block_output
            """UNSQUEEZING BATCH AND ROW DIMENSION CORRECT?"""
            #output[:,:] += block_output
            #output += block_output.view(*output_size).to(torch.device("cpu"))
            output += block_output.view(*output_size)
        
        """TAKE AVERAGE"""
        """OTHER POOLING OPERATIONS?"""
        output = output/self.kernel_3d_num

        
        
        return output
    
    
    
