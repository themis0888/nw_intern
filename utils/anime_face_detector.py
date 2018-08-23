"""
python anime_face_detector.py \
--data_path=/shared/data/nw_data/ \
--save_path=/shared/data/meta/nw_data/detect/ \
--draw_box True --write_list True
"""

import animeface as af 
import PIL.Image as im
import PIL.ImageDraw as dr
import os, random
import argparse
import numpy as np


parser = argparse.ArgumentParser()
# parser.add_argument('--data_storage', type=str, dest='data_storage', default='/shared/data/')
parser.add_argument('--data_path', type=str, dest='data_path', default='/shared/data/nw_data/')
parser.add_argument('--meta_path', type=str, dest='meta_path', default='/shared/data/meta/anime/attributes/')
parser.add_argument('--save_path', type=str, dest='save_path', default='/shared/data/meta/nw_data/detect/')
parser.add_argument('--draw_box', type=bool, dest='draw_box', default=False)
parser.add_argument('--write_list', type=bool, dest='write_list', default=False)
#parser.add_argument('--find_eye')
config, unparsed = parser.parse_known_args() 

key_list = ['face', 'skin', 'hair', 'left_eye', 'right_eye', 'mouth', 'nose', 'chin']


"""
find_face: str -> None
find the face parts from the image and draw a box around of it
you can choose the part that you want to detect
"""
def find_face(img_path, parts = ['face', 'eye']):
	img_name = os.path.basename(img_path)
	source_img = im.open(img_path)

	af_obj = af.detect(source_img)

	# if the image has no face or you don't choose any key to detect, it will stop
	if not af_obj or not parts:
		print('{} has no face'.format(img_name))
		return 1

	af_dict = af_obj[0]
	x, y, width, height = af_dict.face.pos.x, af_dict.face.pos.y, af_dict.face.pos.width, af_dict.face.pos.height
	draw = dr.Draw(source_img)

	if af_dict.likelihood < 0.9:
		return 1

	draw.rectangle(((x-1,y-1), (x+width-1, y+height-1)), outline = "rgb(160, 240, 60)")
	draw.rectangle(((x,y), (x+width, y+height)), outline = "rgb(160, 240, 60)")
	draw.rectangle(((x+1,y+1), (x+width+1, y+height+1)), outline = "rgb(160, 240, 60)")
	draw.text((x,y), 'face')


	draw.text((2,0), 'L={0:03f}'.format(af_dict.likelihood))

	b_x, b_y = 0, 10
	hair_color = af_dict.hair.color
	draw.rectangle(((b_x,b_y), (b_x+10, b_y+10)), 
		fill = (hair_color.r, hair_color.g, hair_color.b),
		outline = 'white')
	draw.text((b_x+12,b_y), 'hair_color')

	eye_color = af_dict.right_eye.color
	draw.rectangle(((b_x, b_y+10), (b_x+10, b_y+20)), 
		fill = (eye_color.r, eye_color.g, eye_color.b),
		outline = 'white')
	draw.text((b_x+12, b_y+10), 'eye_color')

	if 'eye' in parts:
		eye_pos = af_dict.right_eye.pos
		x, y, width, height = af_dict.right_eye.pos.x, af_dict.right_eye.pos.y, af_dict.right_eye.pos.width, af_dict.right_eye.pos.height
		draw.rectangle(((x,y), (x+width, y+height)), outline = "rgb(160, 240, 60)")
		eye_pos = af_dict.left_eye.pos
		x, y, width, height = af_dict.left_eye.pos.x, af_dict.left_eye.pos.y, af_dict.left_eye.pos.width, af_dict.left_eye.pos.height
		draw.rectangle(((x,y), (x+width, y+height)), outline = "rgb(160, 240, 60)")
		#text shadows the face too much
		#draw.text((x,y), 'right_eye')

	if not os.path.exists(config.save_path):
		os.mkdir(config.save_path)

	source_img.save(os.path.join(config.save_path, '{}'.format(img_name)), 'JPEG')
	return 0 

""" 
metadata looks like:
{ '0001.png': [(1,0,0), (252,239,232), (89,120,165), (204,68,101)]
'0002.png': [(0,1,0), (252,223,190), (96,132,180), (201,91,104)]
 ... }
each column stands for sex, skin color, hair color and eye color in RGB format.
"""
def feature_writer(img_path):
	img_name = os.path.basename(img_path)
	source_img = im.open(img_path)

	af_obj = af.detect(source_img)

	# if the image has no face or you don't choose any key to detect, it will stop
	
	if not af_obj:
		print('{} has no face'.format(img_name))
		return False

	af_dict = af_obj[0]

	hair_color = af_dict.hair.color
	eye_color = af_dict.right_eye.color
	skin_color = af_dict.skin.color

	feat_list = [
	skin_color.r, skin_color.g, skin_color.b,
	hair_color.r, hair_color.g, hair_color.b,
	eye_color.r, eye_color.g, eye_color.b]

	feat_list_norm = [value/256 for value in feat_list]

	return feat_list


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



input_path = os.path.join(config.data_path)
file_lst = file_list(input_path, ('.jpg','.png'), True)

if not os.path.exists(config.save_path):
	os.mkdir(config.save_path)


if config.draw_box:
	nw_files = [file for file in file_lst if 'free' not in file]
	random.shuffle(nw_files)

	num_file = 0
	num_no_face = 0
	for i in range(len(nw_files)):
		num_file += 1
		num_no_face += find_face(nw_files[i])
		if i % 100 == 0:
			print('no face images {}/{}'.format(num_no_face, num_file))


if config.write_list:
	feat_dict = {}

	i = 0
	num_no_face = 0
	num_file = len(file_lst)
	for file_path in file_lst:
		# find_face(file_path, ['face', 'eye'])
		img_name = file_path #os.path.basename(file_path)

		
		point_list = np.load(os.path.join(config.meta_path, img_name[:-4]+'_kpt.npy'))
		points = []
		for j in range(68):
			points.append(point_list[j,0])
			points.append(point_list[j,1])
		

		i += 1
		feat_list = feature_writer(file_path)
		if feat_list != False:
			feat_dict[img_name] = feat_list # + points
		else: 
			num_no_face += 1


		if i % 500 == 0:
			print('{0:.3f}% done'.format(100*i/num_file))
			print('no face images {}'.format(num_no_face))

	np.save('anime_feat_dict.npy', np.array(feat_dict))


