import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import glob
import time

import torch
from tqdm.notebook import tqdm

import pytorch_lightning as pl

"""data set class"""

class kl_dataset(torch.utils.data.Dataset):

    """
    class constructor
    typically all class variables are initialized here, but the data itself does not have to be loaded!
    the location of features (samples) and labels are stored as indices
    """

    def __init__(self, conf_file_dir, file_format_list, label_names, labels_in_file_name, transform=None, device=torch.device("cpu")):
        self.device = device
        
        #save the firectory where conf files are saved
        self.conf_file_dir = conf_file_dir
        #save the file format to look for for example "*.dat
        self.file_format_list = file_format_list
        
        #get a list of all configurations to be loaded into the dataset
        self.conf_file_paths = []
        for file_format in self.file_format_list:
            #self.conf_file_paths.append(glob.glob(self.conf_file_dir + file_format))
            self.conf_file_paths += glob.glob(self.conf_file_dir + file_format)

        print(f"found file {len(self.conf_file_paths)} paths:\n{self.conf_file_paths}")
        
        #set the labels to look for in separate files
        self.label_names = label_names
        
        self.labels_in_file_name = labels_in_file_name
        
        #set labels used in training
        self.train_label_names = self.label_names
        
        #each example is saved as a dictionary with the label name as key to the label
        #examples are aggregated into this list
        self.data = []
        
        #load_perm = [0,3,2,1]
        
        #keep track of each conf with an id
        idx = 0
        
        self.file_confs = []
        #iterate through all conf_files
        for conf_file_i in tqdm(range(len(self.conf_file_paths))):
        #for conf_file_i in tqdm(range(0,3)):
            #read in configuration, reshape it and load it to device as a torch tensor
            conf_file_path = self.conf_file_paths[conf_file_i]
            print(f"Processing conf file {conf_file_path}")
            
            #confs = np.load(conf_file_path)
            
            #confs = np.fromfile(conf_file_path, sep=" ", dtype=int)
            #print(f"First read conf file {conf_file_path} with size :{confs.size()}")
            
            """RESHAPE!"""
            #confs = torch.tensor(confs, dtype=int)
            #print(f"First read conf file {conf_file_path} with size :{confs.size()}")
            
            #confs = confs.permute(load_perm)
            
            #send entire file to device, instead of one conf at a time
            #confs.to(device)
            
            #separate entire path from filename
            file_name = conf_file_path.split("/")[-1]
            #isolate chemical potential out of filename.  delimiter = -
            #configs-pars-mu.ext
            file_ext = "." + file_name.split(".")[-1]
            #split .ext from mu.ext
            
            """LOAD MU"""
            mu = torch.tensor(float( file_name.split("-")[-1].replace(file_ext,"") ))
            
            labels_in_file_name = {}
            
            split_file_name = file_name.split("-")
            split_file_name[-1] = split_file_name[-1].replace(file_ext,"")
            
            print(split_file_name)
            
            for i in range(len(self.labels_in_file_name)):
                
                label_val = split_file_name[i+1]

                labels_in_file_name[self.labels_in_file_name[i]] = float(label_val)
                
            print(f"loaded file name labels {self.labels_in_file_name}")
            
            lat_size = [int(labels_in_file_name["nx"]), int(labels_in_file_name["nt"])]
            
            #if "open" in file_name:
            if "open" in split_file_name[0]:
                open_conf = True
            else:
                open_conf = False
                    
            labels_in_file_name["open"] = open_conf
            
            #load labels
            #label file names should have identical name as conf file with "y" replacing "x"
            labels = {}
            conf_prefix = split_file_name[0]
            
            for label_name in self.label_names:
                print(f"for label {label_name}")
                label_prefix = label_name

                label_file_path = conf_file_path.replace(conf_prefix, label_prefix)
                label_file_path = "/".join( conf_file_path.split("/")[:-1] )
                label_file_path = label_file_path + "/" + file_name.replace(conf_prefix, label_prefix)

                print(f"loading labels file {label_file_path}")
                #labels[label_name] = np.load(label_file_path)
                labels[label_name] = np.fromfile(label_file_path, sep="\n", dtype=float)
            
            self.file_confs.append( np.fromfile(conf_file_path, sep=" ", dtype=int) )
            self.file_confs[-1] = torch.tensor(self.file_confs[-1])
            self.file_confs[-1] = self.file_confs[-1].reshape(-1, 2, 2, *lat_size)
            confs = self.file_confs[-1]
            confs = confs.reshape(-1, 4, *lat_size)
            
            print(f"First read conf file {conf_file_path} with size :{confs.shape}")
            
            #confs = torch.tensor(confs, dtype=int)
            #confs = confs.reshape(-1, 2, 2, *lat_size)
            
            print(f"Read conf file {conf_file_path} with size :{confs.size()}")
            num_confs = confs.size()[0]
            
            print(f"Loading {num_confs} confs")
            #iterate through all configurations from one file
            for num_conf in tqdm(range(num_confs)):
                #pick out conf and labels for one example
                conf = confs[num_conf]
                #conf.to(device)
                
                #for conf_size_l in conf.size():
                #    conf_size.append()
            
                #create dictionary for one configuration
                conf_dict = {}
                conf_dict["conf"] = conf
                
                """LOAD CORRECT MU"""
                conf_dict["mu"] = mu
                
                conf_dict["open"] = open_conf
                
                for key in labels_in_file_name:
                    conf_dict[key] = labels_in_file_name[key]
                
                mu_crit = torch.tensor(0.94)
                
                #conf_dict["phase"] = (torch.sign(mu - mu_crit)+1)//2
                conf_dict["phase"] = ((mu - mu_crit).sign()+1)//2 
                
                #if labels != None:
                #if type(labels) == torch.Tensor:

                #label = labels[num_conf]
                #attach all labels to dict
                for label_name in labels:
                    all_conf_labels = labels[label_name]
                    conf_dict[label_name] = all_conf_labels[num_conf]
                    
                for label_name in labels_in_file_name:
                    conf_dict[label_name] = labels_in_file_name[label_name]   
                    
                
                conf_dict["id"] = idx
            
                #save conf_dict to data. This is the actual dataset!
                self.data.append(conf_dict)
                
                if num_conf == 0 and conf_file_i == 0:
                    print("first example loaded:")
                    print(conf_dict)

                    
                #increase conf id by one for next example
                idx +=1
        
        print("last example loaded:")
        print(conf_dict)
        
        self.length = len(self.data)
        
        self.label_names = self.label_names + self.labels_in_file_name

        #define custom transform
        #if transform == "default":
        #    default_axes = list(range(2,len(self.lat_size)+2))
        #    print(f"setting default axes for transforms to {default_axes}")
        #    #self.transform = lambda x: self.lat_trans(self.lat_rot(x, axes=default_axes, random=True, rot_par=42), axes=default_axes, random=True, trans_par=42)
        #    self.transform = lambda x: self.lat_trans(x, axes=default_axes, random=True, trans_par=42)
        #elif transform != None:
        #    self.transform = transform

            
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
        
        #conf_lat_links = self.data[idx]["conf"].reshape(tuple(self.output_size))
        #conf_lat_links = torch.flatten(self.data[idx]["conf"], start_dim=0, end_dim=2)
        conf_lat_links = self.data[idx]["conf"]
        
        
        labels = torch.zeros(len(self.train_label_names))
        for label_name_i in range(len(self.train_label_names)):
            labels[label_name_i] = (self.data[idx][self.train_label_names[label_name_i]])

        
        return (conf_lat_links, labels)
            
        #if self.transform is not None:
        #    return (self.transform(conf_lat_links), labels)
        #    #return (self.lat_translation(conf_lat_links, axes = [1,2]), label)
        #else:
        #    return (conf_lat_links, labels)
        
    
    def filter_indices_label_vals(self, label_names, label_values, remove=False):
        
        filtered_indices = []
        removed_indices = []
        
        print(f"Filtering indices with respect to labels {label_names}, remove = {remove}")
        
        for ex_i, example in tqdm(enumerate(self.data)):
            """Include all examples with labels in label_values"""
            include = True
            if remove == False:
                for label_i, label_name in enumerate(label_names):
                    if example[label_name] not in label_values[label_i]:
                        include = False
                        """break out of label_i loop, because it is clear to not include this example"""
                        removed_indices.append(ex_i)
                        break
            
            """Exclude all examples with labels in label_values"""            
            #elif remove == True:
            #    for label_i, label_name in enumerate(label_names):
            #        if example[label_name] in label_values[label_i]:
            #            include = False
            #            """break out of label_i loop, because it is clear to not include this example"""
            #            removed_indices.append(ex_i)
            #            break
                    
            if include == True:
                filtered_indices.append(ex_i)
                
        return filtered_indices, removed_indices
            
        

def link_per_lat_point(links, lat_size, dims):
    """
    converts links in such a way,
    that ALL valid links corresponding to a lattice site
    can be accessed by indexing it.
    Formerly only positive links can be accessed,
    all other ones are mapped to different sites
    """        

    #empty channels where the linsk on other directions should be saved
    #empty_link_channels = np.zeros(links.shape)
    empty_link_channels = torch.zeros(*links.shape, dtype=int)
    for d in range(dims):
        #print(d)
        #print(links)
        d_links = links[d]
        #print(d_links)
        #perm_links = np.roll(d_links, shift=1, axis=d)
        perm_links = torch.roll(d_links, shifts=1, dims=d)
        #print(perm_links)
        
        #print(d_slices)
        #empty_link_channels[d] = links[zip(d_slices)]
        empty_link_channels[d] = perm_links
        
    #not supported
    #empty_link_channels = empty_link_channels[::-1]
    #torch.flip(input=empty_link_channels, dims=[0])
    
    #complete_links = np.concatenate((links,empty_link_channels), axis=0)
    complete_links = torch.cat((links,empty_link_channels), dim=0)
            
            
    return complete_links

        

    """
    Scaling procedures
    """

    def lat_trans(self, lat_links, axes=[], shifts=[]):
        """
        Data augmentation: lattice translation
        with periodic BC
        """
        trans_lat_links = torch.roll(lat_links, shifts=shifts, dims=axes)
        
        return trans_lat_links
    
    def lat_rot90(self, lat_links, axes=[]):
        """
        Data augmentation: lattice rotation by 90 degrees
        """
        
        return lat_links.transpose(*axes)
    
    
    def lat_flip(self, lat_links, axis=2):
        """
        Data augmentation: lattice parity (rotation by 180 degrees in 2D)
        """
        
        return lat_links.flip(axis)

