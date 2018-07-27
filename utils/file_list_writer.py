"""
python file_list_writer.py \
--data_path=/home/siit/navi/data/sample \
--data_name=sample
"""

import os
#import h5py
#import numpy as np
#import matplotlib.pyplot as plt
#from datasets import imagenet

# ImageNet mapping class_index => class_name
#imagenet_classnames = imagenet.create_readable_names_for_imagenet_labels()
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, dest='data_path', default='/home/siit/navi/data/danbooru2017/256px/')
parser.add_argument('--data_name', type=str, dest='data_name', default='danbooru')
parser.add_argument('--save_path', type=str, dest='save_path', default=data_path+'meta/')
config, unparsed = parser.parse_known_args() 


def find_files(paths, extensions, sort=True):
    if type(paths) is str:
        paths = [paths]
    files = []
    for path in paths:
        for dirs in os.listdir(path):
            if dirs.endswith(extensions):
                files.append(os.path.join(path, dirs))
            else:
                if '.' not in dirs:
                    for file in os.listdir(path+dirs):
                        if file.endswith(extensions):
                            files.append(os.path.join(path+dirs, file))
    if sort:
        files.sort()
    return files



file_lst = find_files(config.data_path,('.jpg','.png'))
f = open(config.save_path + config.data_name + 'file_list.txt', 'w')
for line in file_lst:
    f.write(line + '\n' + ' 0')

f.close()
