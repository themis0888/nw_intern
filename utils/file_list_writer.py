"""
python file_list_writer.py \
--data_path=/home/siit/navi/data/sample/ \
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


# make the save dir if it is not exists
save_path = config.data_path + 'meta/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

# save the file inside of the meta/ folder
file_lst = find_files(config.data_path,('.jpg','.png'))
f = open(config.data_path + 'meta/' + 'path_label_list.txt', 'w')
for line in file_lst:
    f.write(line + ' 0\n')

f.close()
