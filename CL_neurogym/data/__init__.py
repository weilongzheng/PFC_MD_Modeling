import os
import importlib


DATASET_FILENAMES = ['ngym']

all_datasets = {}
for dataset_filename in DATASET_FILENAMES:
    module = importlib.import_module('data.' + dataset_filename)
    class_name = {x.lower():x for x in module.__dir__()}[dataset_filename]
    all_datasets[dataset_filename] = getattr(module, class_name)

def get_dataset(dataset_filename, config):
    assert dataset_filename in DATASET_FILENAMES, "Please choose a dataset from "+str(DATASET_FILENAMES)
    dataset = all_datasets[dataset_filename](config)
    return dataset