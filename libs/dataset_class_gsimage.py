"""data set class"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision

import numpy as np
import pandas as pd

import os

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import MinMaxScaler

from tqdm.notebook import tqdm

class image_dataset(torch.utils.data.Dataset):

    """
    class constructor
    typically all class variables are initialized here, but the data itself does not have to be loaded!
    the location of features (samples) and labels are stored as indices
    """

    def __init__(self, image_file_folder, image_file_paths, image_size, label_path, transform=None, device=torch.device("cpu")):
        #initialize basic dataset variables
        self.image_file_paths = image_file_paths
        self.image_file_folder = image_file_folder
        self.image_size = image_size
        self.label_path = label_path
        self.label_df = pd.read_csv(self.label_path)
        self.label_names = []
        self.length = len(self.label_df)
        self.device = device
        if transform == "default":
            #self.transform = self.min_max_scaling
            self.transform = torchvision.transforms.Compose([
                #torchvision.transforms.ToTensor(),
                min_max_scaler(device=self.device),
                #add_normal_noise(0.,0.1,device=self.device),
            ])
            
        else:
            self.transform = transform
        
        dataset_images_shape = np.concatenate([[self.length], self.image_size])
        
        self.images = np.zeros(shape=dataset_images_shape)
        
        for idx in tqdm(range(self.length)):
            image_file_path = self.image_file_folder  + str(idx)
            image = np.loadtxt(image_file_path).reshape(self.image_size)
            self.images[idx] = image
        
        self.images = torch.tensor(self.images).to(device)

    """function that spits out the databases length"""
    def __len__(self):
        return self.length
    
    def get_length(self):
        return self.length
    
    """
    the getitem function typically takes in an sample/label index and returns them both as a tuple (feature, label)
    this can be in the form of numpy arrays
    the DataLoader functions calls the getitem function in order to create a t
    rain_sampler/test_sampler list!
    """
    def __getitem__(self, idx):

        #load features
        #load labels
        #for one training example
        #and return it

        features = []
        labels = []
        
        """
        #image_file_path = self.image_file_folder + "/" + str(idx)
        image_file_path = self.image_file_folder  + str(idx)
        image = np.loadtxt(image_file_path).reshape(self.image_size)
        """
        
        image = self.images[idx]
        
        for label_name in self.label_names:
            #labels.append(np.array(self.label_df[label_name][idx]))
            labels.append( self.label_df.loc[idx,label_name] )
            #labels.append(self.label_df.loc[[idx],[label_name]])
        
        labels = np.array(labels)
        
        #labels = np.array(self.label_df.loc[ [idx],[self.label_names[0] ]])[0]
        
        #labels = np.array(self.label_df.iloc[idx,0])

        if self.transform is not None:
            return (self.transform(image), labels)
        else:
            return (image, labels)

    
    def get_image(self, idx):

        #load features
        #load labels
        #for one training example
        #and return it

        features = []
        labels = []
        
        """
        #image_file_path = self.image_file_folder + "/" + str(idx)
        image_file_path = self.image_file_folder  + str(idx)
        image = np.loadtxt(image_file_path).reshape(self.image_size)
        """
        
        image = self.images[idx]
        
        for label_name in self.label_names:
            #labels.append(np.array(self.label_df[label_name][idx]))
            labels.append( self.label_df.loc[idx,label_name] )
            #labels.append(self.label_df.loc[[idx],[label_name]])
        
        labels = np.array(labels)
        
        #labels = np.array(self.label_df.loc[ [idx],[self.label_names[0] ]])[0]
        
        #labels = np.array(self.label_df.iloc[idx,0])

        if self.transform is not None:
            return (self.transform(image), labels)
        else:
            return (image, labels)

    """
    Scaling procedures
    Maybe adaptable to images?
    """

    def set_label_names(self, label_names):
        self.label_names = label_names
        return label_names
    
    def get_input_size(self):
        return self.image_size
    
    def min_max_scaling(self, image):
        #mx = np.max(image)
        #mn= np.min(image)
        mx = torch.max(image)
        mn= torch.min(image)
        return (image-mn)/(mx-mn)
    
class min_max_scaler():
    
    def __init__(self, device=torch.device("cpu")):
        self.device = device
    
    def __call__(self, tensor):
        mx = torch.max(tensor).to(self.device)
        mn = torch.min(tensor).to(self.device)
        return (tensor-mn)/(mx-mn)

    def __repr__(self):
        return self.__class__.__name__ + 'min_max_scaler'
    
class add_normal_noise():

    def __init__(self, mu=0., std=1., device=torch.device("cpu")):
        self.device = device
        self.mu = torch.tensor(mu).to(self.device)
        self.sigma = torch.tensor(std).to(self.device)

    def __call__(self, tensor):
        return ( tensor + (torch.randn(tensor.size()).to(self.device) * self.sigma) + self.mu )

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mu}, std={self.sigma})'
        
        
class send_to_device():
    
    def __init__(self, target_device=torch.device("cpu")):
        self.target_device = target_device
    
    def __call__(self, tensor):
        
        return tensor.to(self.target_device)

    def __repr__(self):
        return self.__class__.__name__ + 'send_to_device'
    
class numpy():
    
    def __init__(self, typ=int):
        self.typ = typ
    
    def __call__(self, tensor):
        
        return np.array(tensor.detach().cpu().numpy(), dtype=self.typ)

    def __repr__(self):
        return self.__class__.__name__ + 'numpy'    

    
class torch_tensor():
    
    def __init__(self, typ=torch.float32):
        self.typ = typ
    
    def __call__(self, tensor):
        
        return torch.tensor(tensor.clone(), dtype=self.typ)

    def __repr__(self):
        return self.__class__.__name__ + 'tensor'  
    
class roll_channel_axis():
    
    def __init__(self, old_pos=0, new_pos=-1, device=torch.device("cpu")):
        self.device = device
        self.old_pos = old_pos
        self.new_pos = new_pos
    
    def __call__(self, tensor):
        
        return torch.tensor(np.rollaxis(tensor.detach().cpu().numpy(), self.old_pos, self.new_pos)).to(self.device)

    def __repr__(self):
        return self.__class__.__name__ + 'roll_channel_axis'  
        
