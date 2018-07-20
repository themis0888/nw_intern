import collections, itertools
import os 
import operator
import numpy as np 

num_used_key = 100

# top 100 tags in the danbooru set. 
top_100_keys = {'1girl': 345674, 'bangs': 37471, 'greyscale': 39098, 'looking_at_viewer': 102725, 'monochrome': 50695, 'short_hair': 137511, 'solo': 287883, 'bad_id': 72023, 'bad_pixiv_id': 68063, 'highres': 169068, 'long_hair': 220143, 'purple_eyes': 37339, 'purple_hair': 33781, 'school_uniform': 55722, 'original': 56444, 'thighhighs': 77762, 'blush': 142716, 'comic': 42099, 'touhou': 97488, 'translated': 47231, 'panties': 44679, 'underwear': 51699, 'blue_hair': 47154, 'brown_eyes': 56534, 'serafuku': 24674, 'skirt': 86053, 'smile': 133734, 'blonde_hair': 97508, 'blue_eyes': 97282, 'hair_ornament': 60176, 'hat': 81433, 'twintails': 55694, 'wings': 27897, '1boy': 61644, 'boots': 26836, 'bow': 57729, 'breasts': 145838, 'gloves': 61352, 'heart': 23012, 'holding': 23556, 'large_breasts': 72305, 'nipples': 38840, 'nude': 24963, 'animal_ears': 44793, 'brown_hair': 90703, 'commentary_request': 25552, 'japanese_clothes': 24893, 'green_eyes': 48079, 'one_eye_closed': 24056, 'tail': 30567, 'absurdres': 33377, 'ahoge': 25505, 'long_sleeves': 38159, 'shirt': 40937, 'male_focus': 25817, 'red_eyes': 73911, 'bare_shoulders': 34732, 'jewelry': 37762, 'simple_background': 53045, 'white_background': 41454, 'barefoot': 20899, 'flower': 29384, 'food': 23303, 'green_hair': 27073, 'lying': 21074, 'multiple_girls': 112312, 'sitting': 43857, 'black_hair': 75189, 'closed_eyes': 38280, 'glasses': 25661, 'red_hair': 27388, 'translation_request': 28361, 'weapon': 38016, 'full_body': 24069, 'multiple_boys': 24319, 'standing': 24347, 'hair_ribbon': 36011, 'open_mouth': 108035, 'ribbon': 64490, 'yellow_eyes': 31647, 'hairband': 27527, 'kantai_collection': 43525, 'cleavage': 42535, '2girls': 71117, 'black_legwear': 32126, 'ponytail': 33924, 'very_long_hair': 36799, 'detached_sleeves': 23990, 'necktie': 21405, 'hair_bow': 30569, 'swimsuit': 29256, 'dress': 59495, 'd': 23827, 'navel': 46941, 'braid': 27955, 'ass': 28515, 'pink_hair': 32509, 'pantyhose': 26738, 'medium_breasts': 44779, 'silver_hair': 29798}

# sorted_key_lst: list of sorted 100 tags
sorted_key_lst_tuple = sorted(top_100_keys.items(), key=operator.itemgetter(1), reverse=True)
sorted_key_lst = [v[0] for v in sorted_key_lst_tuple]

# key_number_map: dictionary of the tags and numbers
# key_number_map = {'1girl': 0, 'solo': 1, 'long_hair': 2, ...}
key_number_map = {}
for i in range(num_used_key):
	key_number_map[sorted_key_lst[i]] = i

"""
Convert the tag-dictionary to label dictionary

ex) when num_used_key = 10
tag_dict 	= {..., '2682608': ['1girl', 'bags_under_eyes', ..], ...}
label_dict 	= {..., '608/2682608.jpg': [1, 1, 0, 0, 0, 0, 0, 0, 0, 1], ...}

"""
def convert_to(metadata_path, smoothing = 0):
	tag_dict = {}
	for i in range(6):
		tag_dict.update(dict(np.load(os.path.join(metadata_path, 'metadata_{}.npy'.format(i))).item()))

	key_list = sorted_key_lst[:num_used_key]

	label_dict = {}
	num_data = len(tag_dict)
	for img_num in tag_dict:
		file_name = '{0:04d}/{1}.jpg'.format((int(img_num)%1000), img_num)
		label_dict[file_name] = [smoothing for i in range(num_used_key)]
		for tag in key_list: 
			if tag in tag_dict[img_num]:
				label_dict[file_name][key_number_map[tag]] = 1
	
	return label_dict




def convert_to_300(metadata_path, smoothing = 0):
	tag_dict = {}
	for i in range(6):
		tag_dict.update(dict(np.load(os.path.join(metadata_path, 'metadata_{}.npy'.format(i))).item()))

	key_list = sorted_key_lst[:num_used_key]

	label_dict = {}
	num_data = len(tag_dict)
	for img_num in tag_dict:
		if (int(img_num)%1000) < 300:
			file_name = '{0:04d}/{1}.jpg'.format((int(img_num)%1000), img_num)
			label_dict[file_name] = [smoothing for i in range(num_used_key)]
			for tag in key_list: 
				if tag in tag_dict[img_num]:
					label_dict[file_name][key_number_map[tag]] = 1
	
	return label_dict

