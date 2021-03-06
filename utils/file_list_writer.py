"""
python -i file_list_writer.py \
--data_path=/home/siit/navi/data/input_data/mnist_png/ \
--save_path=/home/siit/navi/data/meta_data/mnist_png/ \
--path_label False
"""

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, dest='data_path', default='/home/siit/navi/data/input_data/mnist_png/')
parser.add_argument('--data_name', type=str, dest='data_name', default='danbooru')
parser.add_argument('--save_path', type=str, dest='save_path', default='/home/siit/navi/data/meta_data/mnist_png/')

parser.add_argument('--path_label', type=bool, dest='path_label', default=False)
parser.add_argument('--iter', type=int, dest='iter', default=1)
config, unparsed = parser.parse_known_args() 



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



# make the save dir if it is not exists
save_path = config.save_path
if not os.path.exists(save_path):
    os.mkdir(save_path)

file_lst = file_list(config.data_path, ('.jpg','.png'), True, config.path_label)
lenth = len(file_lst)


f = open(os.path.join(save_path, 'path_label_list.txt'), 'w')
for line in file_lst:
    f.write(line + '\n')
f.close()


"""
for itr in range(config.iter):
    # save the file inside of the meta/ folder
    f = open(os.path.join(save_path, 'path_label_list{0:03d}.txt'.format(itr)), 'w')
    for line in file_lst[int((itr)*lenth/config.iter):int((itr+1)*lenth/config.iter)]:
        f.write(line + '\n')
    f.close()
"""