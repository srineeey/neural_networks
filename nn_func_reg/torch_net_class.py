"""Neural Network class"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

"""
the neural network class inherits from nn.Module
the default functions
init (constructor) and forward
have to be defined manually
"""
class Net(nn.Module):
    """
    class constructor
    typically the neural networks variables and structure are initialized here
    super().__init__() is necessary to call the mother class constructor
    """
    def __init__( self, net_struct ):
        super(Net, self).__init__()
        self.net_struct = net_struct
        self.input_size = net_struct[0]["layer_pars"]["in_features"]
        self.output_size = net_struct[-1]["layer_pars"]["out_features"]
        self.batch_size = 1

        #self.fc = []
        """
        Using regular list results in an empty parameter list
        nn.ModuleList() is a way to save layers in a python type list
        to avoid this error
        """
        self.fc = nn.ModuleList()

        """
        Construct neural network layers from net_struct dictionary
        """

        for layer in self.net_struct:
            self.fc.append(layer["type"](**layer["layer_pars"]))
            print(layer["layer_pars"])
        #self.init_weights()


    """
    the forward function is called each the the network is propagating forward
    takes in input data and spits out the predicted output
    """
    def forward(self, input_data):

        x = input_data

        """
        iterate through all layers and perofrm calculation
        """
        for layer_i in range(len(self.fc)):
            z = self.fc[layer_i](x)
            if "act_func" in self.net_struct[layer_i]:
                x = self.net_struct[layer_i]["act_func"](z)
            else:
                x = z

            #print(x)

        return x

    def init_weights(self):
        for i, layer in enumerate(self.fc):
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_uniform_(layer.weight)
                if self.net_struct[i]["bias"] == True:
                    layer.bias.data.fill_(0.01)

    """
    Basic method(s) to quickly display network structure, input, output
    """
    def show_net_struct(self):
        print("Network Architecture:\n")
        for layer in self.net_struct:
            print( "Layer: {}".format(layer))

    def show_layers(self):
        for layer in self.fc:
            print(layer)

    def get_input_size(self):
        return self.input_size

    def get_output_size(self):
        return self.output_size

    def get_net_struct(self):
        return self.net_struct

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
