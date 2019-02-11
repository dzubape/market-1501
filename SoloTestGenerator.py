#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from SiamTestGenerator import SiamTestGenerator
from matplotlib import pyplot as plt
import numpy as np


class SoloTestGenerator(SiamTestGenerator):
        
    def get_samples_count(self):
        ## two images per one group ##
        return self.get_group_count() * 2
    
    
    def get_group_count(self):
        return len(self._group_pairs)
    
        
    def __len__(self):
        return self.get_samples_count()
    
        
    def __getitem__(self, idx):
        X = []
        Label = []
        for i in range(self._batch_size):
            [img, group_idx] = self.get_sample(self._batch_size * idx + i)
            X.append(img)
            Label.append(group_idx)
            
        X = np.asarray(X, dtype=np.float32) / 255.
            
        return X, Label
        
    def get_sample(self, idx):
        group_idx = int(idx / 2)
        inpair_idx= idx % 2
        img_idx = self._group_pairs[group_idx][inpair_idx]
        img = self._ds.get_img(img_idx)
        return [img, group_idx]
    
    
    def get_group(self, group_idx):
        assert group_idx >= 0 and group_idx < self.get_group_count()
        
        return self._group_pairs[group_idx]
    
    
    def test_group(self, group_idx=-1):        
        assert group_idx < self.get_group_count()
        
        if group_idx < 0:
            group_idx = random.randint(0, self.get_group_count())

        group = self.get_group(group_idx)
        group_size = len(group)
        img_height = 3
        plt.figure(figsize=(img_height, group_size * img_height * 3), dpi=96)
        
        for i in range(group_size):
            img_idx = group_idx * group_size + i
            [img, _] = self.get_sample(img_idx)
            ax = plt.subplot(1, group_size, i + 1)
            ax.imshow(img)
            ax.set_xlabel("{} from {}".format(i + 1, group_idx))
            ax.axis("image")