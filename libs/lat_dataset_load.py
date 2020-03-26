import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import glob
import time

import torch
from tqdm.notebook import tqdm

"""data set class"""

class kl_dataset(torch.utils.data.Dataset):

    """
    class constructor
    typically all class variables are initialized here, but the data itself does not have to be loaded!
    the location of features (samples) and labels are stored as indices
    """

    def __init__(self, conf_file_dir, file_format_list, conf_size, label_names, output_size, transform=None, device=torch.device("cpu")):
        #initialize basic dataset variables
        self.conf_file_dir = conf_file_dir
        self.file_format_list = file_format_list
        
        """INDEX ORDER CONFUSING"""
        #conf size given as array [degree of freedom, direction, z_l,y_l,x_l] for example
        self.conf_size = conf_size
        #output size is the size with which the configurations are loaded into the network (similar but not always identical to conf_size)
        self.output_size = output_size
        
        #latice size given as array [z_l,y_l,x_l] for example
        self.lat_size = conf_size[2:]
        #if lat_size is given in regular order x,y,z,... -> reverse it
        #self.lat_size = lat_size[::-1]
        #dimension of the lattice
        self.dim = len(self.lat_size)

        #number of total sites in the lattice
        self.n_sites = int(np.array(self.lat_size).prod())

        #one file contains multiple configurations! delimiter = \n
        self.conf_file_paths = []
        for file_format in self.file_format_list:
            self.conf_file_paths.append(glob.glob(self.conf_file_dir + file_format))
        
        self.label_names = label_names
        self.data = []
        
        for conf_file_path in tqdm(self.conf_file_paths):
            #read in configuration, reshape it and load it to device as a torch tensor
            conf = np.fromfile(conf_file_path, sep=" ", dtype=int)
            conf = conf.reshape(self.output_size)
            conf = torch.tensor(conf)
            conf.to(device)
            #separate entire path from filename
            file_name = conf_file_path.split("/")[-1]
            #isolate chemical potential out of filename.  delimiter = -
            #config-mu-id-mu_label-.ext
            mu = float(file_name.split("-")[-2])
            #load other labels
            #...
            
            #create dictionary for one configuration
            conf_dict = {}
            conf_dict["conf"] = conf
            conf_dict["mu"] = mu
            
            #save conf_dict to data. This is the actual dataset!
            self.data.append(conf_dict)
        
        self.length = len(self.data)

        #define custom transform
        if transform == "default":
            """define custom transform"""
            default_axes = list(range(2,len(self.lat_size)+2))
            print(f"setting default axes for transforms to {default_axes}")
            #self.transform = lambda x: self.lat_trans(self.lat_rot(x, axes=default_axes, random=True, rot_par=42), axes=default_axes, random=True, trans_par=42)
            self.transform = lambda x: self.lat_trans(x, axes=default_axes, random=True, trans_par=42)
        elif transform != None:
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
        
        conf_lat_links = self.data[idx]["conf"].reshape(tuple(self.conf_size))
        labels = []
        
        for label_name in self.label_names:
            labels.append(self.data[idx][label_name])
        
        labels = torch.tensor(labels)
            
        if self.transform is not None:
            return (self.transform(conf_lat_links), labels)
            #return (self.lat_translation(conf_lat_links, axes = [1,2]), label)
        else:
            return (conf_lat_links, labels)
        
    def get_conf(self, idx):

        #load features
        #load labels
        #for one training example
        #and return it
        
        conf_lat_links = self.data[idx]["conf"].reshape(tuple(self.conf_size))
        labels = []
        
        for label_name in self.label_names:
            labels.append(self.data[idx][label_name])
            
        if self.transform is not None:
            return (self.transform(conf_lat_links), labels)
            #return (self.lat_translation(conf_lat_links, axes = [1,2]), label)
        else:
            return (conf_lat_links, labels)

    def get_length(self):
        return self.length
    
    """converts array stream of lattice links to an array representing the lattice"""
    """shape is lat_links[link_dir,z,y,x], dimensions in reverse order"""
    def conv_links_to_lat(links, lat_size=[]):
        dim = len(lat_size)
        n_sites = int(lat_size.prod())
        sites = range(n_sites)
        lat_links_shape = [dim] + lat_size
        lat_links_shape = tuple(lat_links_shape)

        #links pointing right and up
        links_ru = np.array(links.reshape(lat_links_shape))

        #links pointing left and down
        #are determined by neighbours and periodic boundary conditions
        links_ld = np.zeros(lat_links_shape)

        #iterate through all dimensions
        for d in range(dim):
            #print(d)
            #print(int(lat_size[int(d)]))

            #links_ld is given by links_rd values, but shifted by a lattice constant
            # + periodic boudnary conditions
            #perm is and index array that specifies how array values should be permutated or swapped

            #possible indices for a particular dimension
            perm = list(range(int(lat_size[int(d)])))
            #shift perm indices by a lattice site and impose periodic boundary conditions
            perm = ([perm[-1]] + perm)[:-1]
            #print(perm)
            perm_indices = []
            #the index arrays for the lattice need to be specified for every dimension
            #for all dimensions other than d, dont perform any permutations (slice(None))
            for b in range(dim):
                if b == d:
                    perm_indices.append(perm)
                else:
                    perm_indices.append(slice(None))

            #perm_indices = perm_indices[::-1]
            #print(perm_indices)

            #lattice links for a given direction d.
            #index is -(int(d)+1) because the order of that array is reversed (due to numpy reshaping procedure)
            links = links_ru[-(int(d)+1)]
            #print(links)
            #print(links[tuple(perm_indices)])

            #left and downward links are given by a permuation of original links
            links_ld[d] = links[tuple(perm_indices)]
            #print(links_ld[d])

        #reverse order of directions in which links are stored
        #this way the order of link directions in links_ru is identical to links_ld
        """CHANGE ORDER OF LINKS_LD, LINKS_RU?"""
        links_ld = links_ld[::-1]

        #join all links and return enmtire lattice
        lat_links = np.concatenate((links_ru, links_ld))

        return lat_links

        

    """
    Scaling procedures
    """

    def lat_trans(self, lat_links, axes=[], random=True, trans_par=42):
        """
        Data augmentation: lattice translation
        """
        #print("performing translation")
        lat_links_shape = lat_links.shape
        trans_lat_links = lat_links
        
        for axis in range(len(lat_links_shape)):
            #print(f"axis {axis}")
            perm_indices = [slice(None)]*len(lat_links_shape)
            if axis in axes:
                perm = list( range(int(lat_links_shape[axis])) )
                shift = 0
                if random == True:
                    """use trans_par as random_seed?"""
                    #np.random.seed = trans_par
                    shift = np.random.randint(0,lat_links_shape[axis])
                else:
                    """use transpar as array for translation vector"""
                    shift = trans_par[axis]
                #print(f"shift {shift}")
                #print(perm)
                perm = (perm + perm)[shift:shift+len(perm)]
                #print(perm)
            
                perm_indices[axis] = perm
                #print(perm_indices)
                trans_lat_links = trans_lat_links[tuple(perm_indices)]
        
        return trans_lat_links
    
    def lat_rot(self, lat_links, axes=[], random=True, rot_par=42):
        """
        Data augmentation: lattice rotation
        """
        num_rot = rot_par
        if random == True:
            num_rot = np.random.choice([1,2,3,4])
        new_lat_links = lat_links
        if num_rot > 0:
            for i in range(num_rot):
                new_lat_links = np.rot90(new_lat_links, axes=axes)
        
        return new_lat_links

        
    def get_input_size(self):
        return self.output_size
    
    def set_label_names(self, label_names):
        self.label_names = label_names