"""data set class"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
from torchvision import transforms

import numpy as np
import pandas as pd

import os

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

import PIL
from PIL import Image


class image_dataset(torch.utils.data.Dataset):

    """
    class constructor
    typically all class variables are initialized here, but the data itself does not have to be loaded!
    the location of features (samples) and labels are stored as indices
    """

    def __init__(self, image_file_folder, image_file_paths, image_res=128, transform="default"):
        #initialize basic dataset variables
        self.image_file_folder = image_file_folder
        self.image_file_paths = image_file_paths
        self.image_res = image_res
        self.image_size = [3, image_res, image_res]
        self.length = len(image_file_paths)
        self.transform = transform
        
        if self.transform == "default":
            default_transforms = [transforms.RandomResizedCrop(self.image_res), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]
            self.transform = transforms.Compose(default_transforms)
        else:
            self.transform = transform
                
        self.raw_labels = []
        for image_file_path in image_file_paths:
            self.raw_labels.append(image_file_path.split("/")[-2])
        
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(self.raw_labels)
        self.labels = self.label_encoder.transform(self.raw_labels)

    """function that spits out the databases length"""
    def __len__(self):
        return self.length
    """
    the getitem function typically takes in an sample/label index and returns them both as a tuple (feature, label)
    this can be in the form of numpy arrays
    the DataLoader functions calls the getitem function in order to create a train_sampler/test_sampler list!
    """
    def __getitem__(self, idx):

        #load features
        #load labels
        #for one training example
        #and return it
        
        with PIL.Image.open(self.image_file_paths[idx]).convert("RGB") as image:
            labels = np.array(self.labels[idx])

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
        
        with PIL.Image.open(self.image_file_paths[idx]).convert("RGB") as image:
            labels = np.array(self.labels[idx])

            if self.transform is not None:
                return (self.transform(image), labels)
            else:
                return (image, labels)


    """
    Scaling procedures
    Maybe adaptable to images?
    """
    
    def get_image_size(self):
        return self.image_size
    
    def custom_scaling(self, attributes=None, scaling_func=lambda x: x):
        """
        Custom scaling according to function/lambda expression
        """
        pass
