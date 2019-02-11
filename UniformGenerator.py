#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import keras
import random
from matplotlib import pyplot as plt


class UniformGenerator(keras.utils.Sequence):

    def __init__(self, ds, batch_size=256, epoch_batch_count=20):
        
        self._ds = ds
        
        self.img_list = self._ds.get_img_list()
        self.person_list = self._ds.get_person_groups()
        
        self.batch_size = batch_size
        self.batch_count = epoch_batch_count
        self.epoch_size = self.batch_size * self.batch_count
        
        self._epoch_sample_list = None
        self.generate_pairs()
        

    def __len__(self):
        '''Число обрабатываемых батчей на эпоху'''
        return self.batch_count
    
    
    def __getitem__(self, idx):
        '''Returns batch [A, B], Label'''
        
        X1 = np.ndarray(shape=(self.batch_size, *self._ds.get_input_shape()), dtype=np.float32)
        X2 = np.ndarray(shape=(self.batch_size, *self._ds.get_input_shape()), dtype=np.float32)
        Y = np.ndarray(shape=(self.batch_size,), dtype=np.float32)
        
        for i in range(self.batch_size):
            [X1[i], X2[i]], Y[i] = self.get_sample(self.batch_size * idx + i)
        return [X1, X2], Y
        
    
    def on_epoch_end(self):
        self.generate_pairs()
    
    
    def get_batch_count(self):
        return self.batch_count
    
    
    def get_epoch_size(self):
        return self.epoch_size
    
    
    def get_label_list(self, start=0, count=-1):
        assert start >= 0 and start < len(self._epoch_sample_list)
        
        if count <= 0:
            count = len(self._epoch_sample_list) - start
        assert start + count <= len(self._epoch_sample_list)
        
        return self._epoch_sample_list[start:start+count,2]
    
    
    def get_sample(self, idx):
        '''Работает через сгенерированное ранее хранилище h5py'''
        
        a_idx, b_idx, label = self._epoch_sample_list[idx]
        
        img_a = self._ds.get_img(a_idx)
        img_b = self._ds.get_img(b_idx)        
        
        img_a = img_a.astype(np.float32) / 255.
        img_b = img_b.astype(np.float32) / 255.
        label = float(label)
        
        return [img_a, img_b], label
    
    
    def check_epoch_samples(self):
        '''Тест парогенератора'''
        return
        
        print(self._epoch_sample_list)
        print("Total samples count: {}".format(len(self._epoch_sample_list)))

        label_acc = 0.
        for pair in self._epoch_sample_list:
            label_acc += pair[2]

        unique_samples = set()
        for i, sample in enumerate(self._epoch_sample_list):
            aligned_sample = list(sample[:2])
            aligned_sample.sort()
            aligned_sample.append(sample[2])
            elem = "{}|{}|{}".format(*aligned_sample)
            if elem in unique_samples:
                print("Duplicate: {}".format(elem))
            unique_samples.add(elem)
            
        #print(unique_samples)
            
        print("Unique samples count: {}".format(len(set(unique_samples))))

        pos_count = label_acc/self.epoch_size
        neg_count = (self.epoch_size - label_acc)/self.epoch_size
        print("pos/neg: %.2f / %.2f" % (pos_count, neg_count))
        

    def generate_pairs(self):
        '''Делаем новый парный замес'''
        
        neg_count = int(self.epoch_size * 0.9)
        pos_count = self.epoch_size - neg_count
        
        pos_samples = self.generate_pos(pos_count)
        neg_samples = self.generate_neg(neg_count)
        self._epoch_sample_list = np.asarray([*pos_samples, *neg_samples], dtype=np.int32)
        
        np.random.shuffle(self._epoch_sample_list)
        
        self.check_epoch_samples()
        
                    
    def generate_pos(self, samples_count):
        '''Генератор пар на одной персоне'''
        
        samples_a = []
        samples_b = []
        labels = []
        used_persons = set()
        
        persons_count = len(self.person_list)
        
        epoch_samples_count = self.batch_count * self.batch_size
        
        while len(samples_a) < samples_count:
            while True:
                person_idx = np.random.randint(persons_count)
                if person_idx not in used_persons:
                    break
            
            used_persons.add(person_idx)
            
            person_gallery = self.person_list[person_idx]
            
            ## Positive samples ##
            
            comb_list = []
            for i in range(len(person_gallery)):
                for j in range(i + 1, len(person_gallery)):
                    comb_list.append([i, j])
                    
            if len(comb_list) <= 1:
                continue
            
            max_same_count = max(1, np.random.randint(1, len(comb_list)))
            comb_indices = np.random.randint(0, len(comb_list), size=max_same_count)
            comb_indices = list(set(comb_indices))
            comb_list = [comb_list[i] for i in comb_indices]
            
            for a_idx, b_idx in comb_list:
                samples_a.append(person_gallery[a_idx])
                samples_b.append(person_gallery[b_idx])
                labels.append(0)
                
        return list(zip(samples_a, samples_b, labels))
    
    
    def generate_neg(self, samples_count):
        '''Генератор пар из двух персон'''
        
        samples_a = []
        samples_b = []
        labels = []
        used_persons = set()
        
        persons_count = len(self.person_list)
        
        while len(samples_a) < samples_count:
            while True:
                person_idx = np.random.randint(0, persons_count)
                if person_idx not in used_persons:
                    break
                    
            used_persons.add(person_idx)
            
            
            person_gallery = self.person_list[person_idx]
            
            max_diff_count = 50
            diff_b_indices = np.random.randint(0, len(self.img_list), size=max_diff_count)
            
            ## Filter off duplicates ##
            diff_b_indices = list(set(diff_b_indices))
            
            ## Filter off person gallery indices
            for idx in person_gallery:
                if idx in diff_b_indices:
                    diff_b_indices.remove(idx)
            
            diff_count = len(diff_b_indices)
            
            diff_a_idx_indices = np.random.randint(len(person_gallery), size=diff_count)
            diff_a_indices = [person_gallery[i] for i in diff_a_idx_indices]
            
            ## Filter off mirror pairs
            for b in diff_b_indices:
                if b in samples_a:
                    sample_idx = samples_a.index(b)
                    if samples_b[sample_idx] == b:
                        diff_a_indices.pop(sample_idx)
                        diff_b_indices.pop(sample_idx)
            
            samples_a.extend(diff_a_indices)
            samples_b.extend(diff_b_indices)
            labels.extend([1] * len(diff_a_indices))
            
        return list(zip(samples_a, samples_b, labels))
    
    
    def test_sample(self):
        [img_a, img_b], label = self.get_sample(random.randint(0, self.epoch_size - 1))
        plt.figure(figsize=(3,3), dpi=112)
        ax = plt.subplot(1, 1, 1)
        ax.imshow(img_a)

                