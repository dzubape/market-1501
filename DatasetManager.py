#!/usr/bin/env python3
# -*- coding: utf-8 -*-
        

import os, re, h5py, gc
import random
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

def close_all_h5():
    '''Closes all h5py opend files'''
    
    ## Browse through all objects
    for obj in gc.get_objects():
        try:
            ## Just h5py files
            if isinstance(obj, h5py.File):
                try:
                    obj.close()
                    print("{} has been closed".format(obj.filename))
                except:
                    ## Has been already closed
                    pass 
        except:
            pass
        

class DatasetManager:
    
    def __init__(self, storage_file_path):
        self.filter = re.compile(r"^(\d{4}).+\.jpg$")
        self.storage_file_path = storage_file_path
        
        
    def __del__(self):
        self.stop_read_storage()
        #print("The End")
        
        
    def start_read_storage(self):
        if not hasattr(self, "hdh5"):
            #print("start_read")
            self.hdh5 = h5py.File(self.storage_file_path, 'r')
        
        
    def stop_read_storage(self):
        if hasattr(self, "hdh5") and isinstance(self.hdh5, h5py.File):
            #print("stop_read")
            self.hdh5.close()
            del self.hdh5
        
        
    def set_img_dir(self, img_dir):        
        self.img_dir = img_dir
        
        
    def get_img_dir(self):
        return self.img_dir
        
        
    def get_img_number(self, img_name):

        match = self.filter.search(img_name)
        if match is None:
            return
        return int(match.groups()[0])


    def filter_numbered(self, img_list):

        valid_img_list = []

        for img_name in img_list:
            number = self.get_img_number(img_name)
            if number is None:
                continue
            valid_img_list.append(img_name)

        return valid_img_list


    def __prepare_data(self):
        '''Prepares image list and groups image indices to person lists'''

        img_list = os.listdir(self.img_dir)
        img_list.sort()
        img_list = self.filter_numbered(img_list)

        print("Total images in {}: {}".format(self.img_dir, len(img_list) ))

        person_dict = {}
        for img_idx, img_name in enumerate(img_list):
            person_id = self.get_img_number(img_name)
            if person_id in person_dict:
                person_dict[person_id].append(img_idx)
            else:
                person_dict[person_id] = [img_idx]

        ## Distractors group
        if 0 in person_dict:
            self.distractor_group = person_dict.pop(0)
            
        self.person_groups = list(person_dict.values())
        self.img_list = img_list

        self.check_ds()
        
    
    def check_ds(self):
        '''Self-testing'''
        
        counter = 0
        check_set = set()
        for group in self.get_person_groups():
            counter += len((group))
            check_set |= set((*group,))

        print("Total img idx count: {}".format(counter))
        print("Unique img idx count: {}".format(len(check_set)))

    def export_to_h5py(self, rewrite=False):
        '''Export image files to h5py'''
        
        if os.path.exists(self.storage_file_path):
            if not rewrite:
                return
            
        self.__prepare_data()
        
        self.stop_read_storage()

        hdh5 = h5py.File(self.storage_file_path, "w")

        test_img = mpl.image.imread(os.path.join(
            self.img_dir,
            self.img_list[0]
        ))
        self.input_shape = test_img.shape
        h5ds_img_list = hdh5.create_dataset(
            "img_list",
            (len(self.img_list), *self.input_shape),
            dtype="uint8"
        )
        for i, img_name in enumerate(self.img_list):
            img = mpl.image.imread(os.path.join(self.img_dir, img_name))
            h5ds_img_list[i] = img
            
        dt_vlen_idx = h5py.special_dtype(vlen=np.dtype('uint16'))
        h5ds_person_img_groups = hdh5.create_dataset(
            "person_img_groups",
            (len(self.person_groups), ),
            dtype=dt_vlen_idx
        )
        for i, person_img_group in enumerate(self.person_groups):
            h5ds_person_img_groups[i] = person_img_group

        hdh5.flush()
        hdh5.close()
        
    def get_input_shape(self):
        while True:
            try:
                return self.input_shape
            except AttributeError:
                self.start_read_storage()
                self.input_shape = self.hdh5["img_list"][0].shape
        
        
    def get_img(self, idx):
        '''Return single image on idx'''
        
        while True:
            try:
                return self.hdh5["img_list"][idx]
            except AttributeError:
                self.start_read_storage()
                
                
    def get_img_list(self):
        if not hasattr(self, "img_list"):
            self.start_read_storage()
            self.img_list = self.hdh5["img_list"]
            
        return self.img_list
    
    
    def get_person_groups(self):
        if not hasattr(self, "person_groups"):
            self.start_read_storage()
            self.person_groups = self.hdh5["person_img_groups"]
            
        return self.person_groups
    
    
    def test_h5py(self):
        self.get_img_list()
        plt.figure(figsize=(3, 3), dpi=96, frameon=False)
        ax = plt.subplot(1, 1, 1)
        test_idx = random.randint(1, len(self.img_list)) - 1
        ax.imshow(self.get_img(test_idx))