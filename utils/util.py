"""
CUDA_VISIBLE_DEVICES=3 CUDA_CACHE_PATH="/gpu_cache/" \
python feature_extractor.py \
--data_path=/shared/data/sample/ \
--list_path=/shared/data/sample/meta/ \
--model_name=vgg_19


CUDA_VISIBLE_DEVICES=1 python feature_extractor.py \
--data_path=/shared/data/danbooru2017/256px/ \
--list_path=/shared/data/danbooru2017/256px/meta/ \
--model_name=vgg_19
"""

import pdb
import random, time, os, sys
import numpy as np
import scipy
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import scipy.io as sio

import argparse

def file_list(path, extensions, sort=True, path_label = False):
    if path_label == True:
        result = [(os.path.join(dp, f) + ' ' + os.path.join(dp, f).split('/')[-2])
        for dp, dn, filenames in os.walk(path) 
        for f in filenames if os.path.splitext(f)[1] in extensions]
    else:
        result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) 
        for f in filenames if os.path.splitext(f)[1] in extensions]
    if sort:
        result.sort()

    return result