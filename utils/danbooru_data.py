import collections, itertools
import os 
import operator


num_used_key = 10

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

top_100_tags 	= {'1girl', 'bangs', 'greyscale', 'looking_at_viewer', 'monochrome', 'short_hair', 'solo', 'bad_id', 'bad_pixiv_id', 'highres', 'long_hair', 'purple_eyes', 'purple_hair', 'school_uniform', 'original', 'thighhighs', 'blush', 'comic', 'touhou', 'translated', 'panties', 'underwear', 'blue_hair', 'brown_eyes', 'serafuku', 'skirt', 'smile', 'blonde_hair', 'blue_eyes', 'hair_ornament', 'hat', 'twintails', 'wings', '1boy', 'boots', 'bow', 'breasts', 'gloves', 'heart', 'holding', 'large_breasts', 'nipples', 'nude', 'animal_ears', 'brown_hair', 'commentary_request', 'japanese_clothes', 'green_eyes', 'one_eye_closed', 'tail', 'absurdres', 'ahoge', 'long_sleeves', 'shirt', 'male_focus', 'red_eyes', 'bare_shoulders', 'jewelry', 'simple_background', 'white_background', 'barefoot', 'flower', 'food', 'green_hair', 'lying', 'multiple_girls', 'sitting', 'black_hair', 'closed_eyes', 'glasses', 'red_hair', 'translation_request', 'weapon', 'full_body', 'multiple_boys', 'standing', 'hair_ribbon', 'open_mouth', 'ribbon', 'yellow_eyes', 'hairband', 'kantai_collection', 'cleavage', '2girls', 'black_legwear', 'ponytail', 'very_long_hair', 'detached_sleeves', 'necktie', 'hair_bow', 'swimsuit', 'dress', 'd', 'navel', 'braid', 'ass', 'pink_hair', 'pantyhose', 'medium_breasts', 'silver_hair'}
gan_tags		= {'aqua eyes', 'aqua hair', 'black eyes', 'black hair', 
       'blonde hair', 'blue eyes', 'blue hair', 'blush', 'brown eyes',
       'brown hair', 'drill hair', 'glasses', 'green eyes', 'green hair',
       'grey hair', 'hat', 'long hair', 'open mouth', 'orange eyes',
       'orange hair', 'pink eyes', 'pink hair', 'ponytail', 'purple eyes',
       'purple hair', 'red eyes', 'red hair', 'ribbon', 'short hair',
       'silver hair', 'smile', 'twintails', 'white hair', 'yellow eyes'}


please_dont 	= {'bangs', 'nipples', 'nude', 'ass'}
dont_know 		= {'translated', 'd'}
irrelevant 		= {'bad_id', 'bad_pixiv_id', 'translation_request', 'commentary_request',  'absurdres', 'original'}

tag_num_sex		= {'1girl','solo','multiple_girls','2girls','1boy','multiple_boys'}
tag_hair		= {"short_hair", "long_hair", "purple_hair", "blue_hair", "blonde_hair", "hair_ornament", "brown_hair", "green_hair", "black_hair", "red_hair", "hair_ribbon", "hairband", "very_long_hair", "hair_bow", "pink_hair", "silver_hair"}


obj_set = {'solo', '1girl', '1boy', '2girls', 'multiple_girls', 'multiple_boys'}
hair_set = {'long_hair', 'short_hair', 'twintails', 'bangs', 'very_long_hair', 'ponytail', 'hair_ribbon', 'hair_ornament', 'hair_bow', 'braid', 'hairband', 'ahoge'}

# grouped_tag_lst : list of list 
# [['solo', '1girl', ...], ['long_hair', 'short_hair', ...], ...]
# You can find the order of the list from 'group_order'
group_order = ['object', 'hair_feature']
grouped_tag_lst = [[] for i in range(len(group_order))]
group_set_lst = [obj_set, hair_set]

object_set = []
hair_feature_lst = []

for tag in sorted_key_lst:
	for num_group in range(len(group_order)):
		if tag in group_set_lst[num_group]:
			grouped_tag_lst[num_group].append(tag)

