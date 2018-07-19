import collections, itertools
import os 
import operator
import numpy as np 
import danbooru_data as dd

"""
Convert the tag-dictionary to label dictionary

ex) when dd.num_used_key = 10
tag_dict 	= {..., '2682608': ['1girl', 'bags_under_eyes', ..], ...}
label_dict 	= {..., '608/2682608.jpg': [1, 1, 0, 0, 0, 0, 0, 0, 0, 1], ...}
"""

tag_dict = {}
for i in range(6):
	tag_dict.update(dict(np.load('../metadata_{}.npy'.format(i)).item()))

key_list = dd.sorted_key_lst[:dd.num_used_key]

label_dict = {}
num_data = len(tag_dict)
for img_num in tag_dict:
	file_name = '{}/{}.jpg'.format((int(img_num)%1000), img_num)
	label_dict[file_name] = [0 for i in range(dd.num_used_key)]
	for tag in key_list: 
		if tag in tag_dict[img_num]:
			label_dict[file_name][dd.key_number_map[tag]] = 1


