#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import keras
import numpy as np


class SiamTestGenerator(keras.utils.Sequence):
    def __init__(self, ds):
        
        self._ds = ds
        self._person_groups = ds.get_person_groups()
        self._batch_size = 1
        
        ## список групп, содержащих по паре индексов
        ## изображений одного объекта
        ## [base:i, sim:i], i:0~n
        self._group_pairs = list()
        
        for base_group in self._person_groups:
            if len(base_group) < 2:
                continue
            [base_idx, same_idx] = random.sample(range(len(base_group)), 2)
            base_idx = base_group[base_idx]
            same_idx = base_group[same_idx]
            self._group_pairs.append((base_idx, same_idx))
        
        ## [base:i, sim:i] + [base, dis:i+1~n]
        self._pair_list = list()
        
        for i, (base_idx, same_idx) in enumerate(self._group_pairs):
            self._pair_list.append([base_idx, same_idx, 0])
            for j in range(i + 1, len(self._group_pairs)):
                diff_idx = self._group_pairs[j][0]
                self._pair_list.append([base_idx, diff_idx, 1])
                
        self._pair_list = np.asarray(self._pair_list)
                
    def get_samples_count(self):
        return len(self._pair_list)
    
    
    def get_group_count(self):
        return len(self._group_pairs)
        
    
    def __len__(self):
        return self.get_samples_count()
    
    
    def get_pair_list(self):
        return self._pair_list
    
    
    def get_label_list(self, start=0, count=-1):        
        assert start >=0 and start < len(self._pair_list)
        
        if count <= 0:
            count = len(self._pair_list)
        assert count < len(self._pair_list)
        
        return self._pair_list[start:start+count, 2]
    
    
    def get_sample(self, idx):
        a_idx, b_idx, label = self._pair_list[idx]
        img_a = self._ds.get_img(a_idx)
        img_b = self._ds.get_img(b_idx)
        
        return [img_a, img_b], label
    
    
    def __getitem__(self, idx):
        '''Returns batch [A, B], Label'''          
        A = []
        B = []
        Label = []

        for i in range(self._batch_size):
            [a, b], label = self.get_sample(idx * self._batch_size + i)
            A.append(a)
            B.append(b)
            Label.append(label)

        A = np.asarray(A, dtype=np.float32) / 255.
        B = np.asarray(B, dtype=np.float32) / 255.
        Label = np.asarray(Label, dtype=np.float32)

        return [A, B], Label
    