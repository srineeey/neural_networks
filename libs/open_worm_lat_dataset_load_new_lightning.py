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
            
        #print(self.conf_file_paths)
        
        #set the labels to look for in separate files
        self.label_names = label_names
        
        self.labels_in_file_name = labels_in_file_name
        
        #set labels used in training
        self.train_label_names = self.label_names
        
        #each example is saved as a dictionary with the label name as key to the label
        #examples are aggregated into this list
        self.data = []
        
        load_perm = [0,3,2,1]
        
        #keep track of each conf with an id
        idx = 0
        
        #iterate through all conf_files
        for conf_file_i in tqdm(range(len(self.conf_file_paths))):
        #for conf_file_i in tqdm(range(0,3)):
            #read in configuration, reshape it and load it to device as a torch tensor
            conf_file_path = self.conf_file_paths[conf_file_i]
            confs = np.load(conf_file_path)
            
            confs = torch.tensor(confs, dtype=int)
            confs = confs.permute(load_perm)
            
            #send entire file to device, instead of one conf at a time
            #confs.to(device)
            
            #separate entire path from filename
            file_name = conf_file_path.split("/")[-1]
            #isolate chemical potential out of filename.  delimiter = -
            #configs-pars-mu.ext
            file_ext = "." + file_name.split(".")[-1]
            #split .ext from mu.ext
            
            """LOAD MU"""
            #mu = torch.tensor(float( file_name.split("-")[-1].replace(file_ext,"") ))
            mu_i = int( file_name.split(".")[0][-1])
            mu_start = 0.90
            mu_stop = 1.20
            mu_delta = 0.05
            
            mu = torch.tensor(float(mu_start + float(mu_i)*mu_delta))
            #mu = torch.tensor(1.00)
            
            #load labels
            #label file names should have identical name as conf file with "y" replacing "x"
            labels_array = []
            conf_prefix = "x"
            label_prefix = "y"
            
            if "open" in file_name:
                open_conf = True
            else:
                open_conf = False
            
            #label_file_path = conf_file_path.replace(conf_prefix, label_prefix)
            label_file_path = "/".join( conf_file_path.split("/")[:-1] )
            label_file_path = label_file_path + "/" + file_name.replace(conf_prefix, label_prefix)
            
            labels_file_found = True
            
            try:
                print(f"trying to load labels file {label_file_path}")
                labels = np.load(label_file_path)
                labels = torch.tensor(labels)
                print(labels.size())
            except FileNotFoundError:
                print(f"labels file not found {label_file_path}")
                labels = None
                labels_file_found = False
            
            
            print(f"Read conf file {conf_file_path} with size :{confs.size()}")
            num_confs = confs.size()[0]
            
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
                
                mu_crit = torch.tensor(0.94)
                
                #conf_dict["phase"] = (torch.sign(mu - mu_crit)+1)//2
                conf_dict["phase"] = ((mu - mu_crit).sign()+1)//2 
                
                #if labels != None:
                #if type(labels) == torch.Tensor:
                if labels_file_found == True:
                    label = labels[num_conf]
                    #attach all labels to dict
                    for label_i in range(len(self.label_names)):
                        conf_dict[self.label_names[label_i]] = label[label_i]
                
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
    
    def set_train_label_names(self, train_label_names):
        self.train_label_names = train_label_names