#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, getopt, os

from UniformGenerator import UniformGenerator
from SiamTestGenerator import SiamTestGenerator
from SoloTestGenerator import SoloTestGenerator
from DatasetManager import DatasetManager, close_all_h5
from NetModel import build_model, fit_this_feet, open_base_model, load_model, save_model, last_train_start, compile_model
from NetTest import test_base_model


def train_model(model, dataset_manager, epoch_count):
    
    train_generator = UniformGenerator(
        dataset_manager,
        batch_size=128,
        epoch_batch_count=20
    )
    train_generator.test_sample()
    
    fit_this_feet(model, train_generator, epoch_count=epoch_count)


def test_model(model_mark, dataset_manager, test_count):
    try:
        base_model = open_base_model(model_mark)
    except:
        print("Can't open base model {}".format(model_mark))
        return
    solo_test_generator = SoloTestGenerator(dataset_manager)
    test_base_model(base_model, solo_test_generator, test_count)

    
def print_help():
    print("Help yourself!")

    
def main(argv):
        
    task = None
    model_mark = None
    train_epoch_count = None
    test_group_count = None
    model_mark = None
    
    dataset_dir = "dataset"
    train_dir = os.path.join(dataset_dir, "bounding_box_train")
    test_dir = os.path.join(dataset_dir, "bounding_box_test")
    
    try:
        opts, args = getopt.getopt(argv, "htvug:e:m:", [
            "train",
            "test",
            "use",
            "test-group-count=",
            "train-epoch-count=",
            "model-mark="
        ])
    except getopt.GetoptError:
        print_help()
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-h':
            print_help()
            sys.exit()
        elif opt in ("-t", "--train"):
            ds_storage = "train.hdh5"
            ds_dir = train_dir
            task = "train"
            #return
        elif opt in ("-v", "--test"):
            ds_storage = "train.hdh5"
            ds_dir = test_dir
            task = "test"
        elif opt in ("-u", "--use"):
            task = "use"
            print("use is not implemented yet")
            sys.exit(0)
        elif opt in ("-e", "--train-epoch-count"):
            train_epoch_count = int(arg)
        elif opt in ("-g", "--test-group-count"):
            test_group_count = int(arg)
        elif opt in ("-m", "--model-mark"):
            model_mark = arg
            
        if task is None:
            sys.exit(2)
    
    #close_all_h5()
        
    dataset_manager = DatasetManager(storage_file_path=ds_storage)
    dataset_manager.set_img_dir(ds_dir)
    dataset_manager.export_to_h5py(rewrite=False)
    
    if task == "train":
        if model_mark is None:
            model = build_model(dataset_manager.get_input_shape())
        else:
            model = load_model(model_mark)
           
        compile_model(model)
        
        train_model(
            model,
            dataset_manager,
            epoch_count=train_epoch_count
        )
        save_model(model, last_train_start())
    elif task == "test":
        test_model(
            model_mark,
            dataset_manager,
            test_group_count
        )
    
    
if __name__ == '__main__':
    main(sys.argv[1:])