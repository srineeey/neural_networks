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


class dataset(torch.utils.data.Dataset):

    """
    class constructor
    typically all class variables are initialized here, but the data itself does not have to be loaded!
    the location of features (samples) and labels are stored as indices
    """

    def __init__(self, file_path, target_attribute=[], pred_attributes=[], transform=None):
        #initialize basic dataset variables
        self.file_path = file_path

        self.pred_attributes = pred_attributes
        self.target_attribute = target_attribute
        self.df = pd.read_csv(self.file_path)
        self.attributes = self.df.columns
        self.length = len(self.df)
        self.transform = transform

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

        features = []
        labels = []

        for pred_attribute in self.pred_attributes:
            features.append(self.df[pred_attribute][idx])
        for target_attribute in self.target_attribute:
            labels.append(self.df[target_attribute][idx])

        features = np.array(features)
        labels = np.array(labels)

        if self.transform is not None:
            return self.transform( (features, labels) )
        else:
            return (features, labels)


    def get_length(self):
        return self.length

    def rename_columns(self, rename_dict = {}):
        """
        Takes in and saves a list of target/feature names which act as filters in the pandas dataframe
        """
        self.df.rename(columns=rename_dict, inplace=True)

    """
    Scaling procedures
    """
    def z_score_standardization(self, attributes=None):
        if attributes == None:
            attributes = self.pred_attributes
        mapper = DataFrameMapper([(attributes, StandardScaler())])
        scaled_features = mapper.fit_transform(self.df[attributes].copy(), 4)
        self.df[attributes] = pd.DataFrame(scaled_features, index=self.df[attributes].index, columns=attributes)

    def min_max_scaling(self, attributes=None):
        if attributes == None:
            attributes = self.pred_attributes
        mapper = DataFrameMapper([(attributes, MinMaxScaler(feature_range=(0, 1)))])
        scaled_features = mapper.fit_transform(self.df[attributes].copy(), 4)
        self.df[attributes] = pd.DataFrame(scaled_features, index=self.df[attributes].index, columns=attributes)

    def custom_scaling(self, attributes=None, scaling_func=lambda x: x):
        """
        Custom scaling according to function/lambda expression
        """
        if attributes == None:
            attributes = self.pred_attributes
        mapper = DataFrameMapper([(attributes, scaling_func)])
        scaled_features = scaling_func(self.df[attributes].copy())
        self.df[attributes] = pd.DataFrame(scaled_features, index=self.df[attributes].index, columns=attributes)

    """
    Manually reset target/pred attributes
    """
    def set_pred_attributes(self, pred_attributes):
        self.pred_attributes = pred_attributes

    def set_target_attribute(self, target_attribute):
        self.target_attribute = target_attribute

    def reload_df(self):
        """
        Reset DF to original
        """
        self.df = pd.read_csv(self.file_path)
