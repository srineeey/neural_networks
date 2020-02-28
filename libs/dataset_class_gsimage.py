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


class image_dataset(torch.utils.data.Dataset):

    """
    class constructor
    typically all class variables are initialized here, but the data itself does not have to be loaded!
    the location of features (samples) and labels are stored as indices
    """

    def __init__(self, image_file_folder, image_file_paths, image_size, label_path, transform=None):
        #initialize basic dataset variables
        self.image_file_paths = image_file_paths
        self.image_file_folder = image_file_folder
        self.image_size = image_size
        self.label_path = label_path
        self.label_df = pd.read_csv(self.label_path)
        self.label_names = []
        self.length = len(self.label_df)
        self.transform = transform

    """function that spits out the databases length"""
    def __len__(self):
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
        image = np.loadtxt(self.image_file_paths[idx]).reshape(self.image_size)
        
        for label_name in self.label_names:
            #labels.append(np.array(self.label_df[label_name][idx]))
            labels.append( np.array(self.label_df.loc[[idx],[label_name]]) )
            #labels.append(self.label_df.loc[[idx],[label_name]])
        
        labels = np.array(labels)
        
        #labels = np.array(self.label_df.loc[ [idx],[self.label_names[0] ]])[0]
        
        #labels = np.array(self.label_df.iloc[idx,0])

        if self.transform is not None:
            return (self.transform(image), labels)
        else:
            return (image, labels)
        """
        
        #image_file_path = self.image_file_folder + "/" + str(idx)
        image_file_path = self.image_file_folder  + str(idx)
        image = np.loadtxt(image_file_path).reshape(self.image_size)
        
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

    def get_length(self):
        return self.length
    
    def get_image(self, idx):

        #load features
        #load labels
        #for one training example
        #and return it

        features = []
        labels = []
        
        """
        image = np.loadtxt(self.image_file_paths[idx]).reshape(self.image_size)
        
        for label_name in self.label_names:
            #labels.append(np.array(self.label_df[label_name][idx]))
            labels.append( np.array(self.label_df.loc[[idx],[label_name]]) )
            #labels.append(self.label_df.loc[[idx],[label_name]])
        
        labels = np.array(labels)
        
        #labels = np.array(self.label_df.loc[ [idx],[self.label_names[0] ]])[0]
        
        #labels = np.array(self.label_df.iloc[idx,0])

        if self.transform is not None:
            return (self.transform(image), labels)
        else:
            return (image, labels)
        """
        
        #image_file_path = self.image_file_folder + "/" + str(idx)
        image_file_path = self.image_file_folder + str(idx)
        image = np.loadtxt(image_file_path).reshape(self.image_size)
        
        for label_name in self.label_names:
            #labels.append(np.array(self.label_df[label_name][idx]))
            labels.append( self.label_df.loc[idx,label_name] )
            #labels.append(self.label_df.loc[[idx],[label_name]])
        
        #print(np.shape(labels))
        labels = np.array(labels)
        #print(labels)
        
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
    
    def custom_scaling(self, attributes=None, scaling_func=lambda x: x):
        """
        Custom scaling according to function/lambda expression
        """
        pass
