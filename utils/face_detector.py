"""
python face_detector.py \
--data_storage=/shared/data \
--data_path=danbooru2017/256px/0000/
"""

import animeface as af 
import PIL.Image as im
import PIL.ImageDraw as dr
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_storage', type=str, dest='data_storage', default='/shared/data/')
parser.add_argument('--data_path', type=str, dest='data_path', default='danbooru2017/256px/0000/')
config, unparsed = parser.parse_known_args() 

key_list = ['face', 'skin', 'hair', 'left_eye', 'right_eye', 'mouth', 'nose', 'chin']

"""
find_face: str -> None
find the face parts from the image and draw a box around of it
you can choose the part that you want to detect
"""
def find_face(img_path, parts = ['face']):
	img_name = os.path.basename(img_path)
	source_img = im.open(img_path)

	af_obj = af.detect(source_img)

	# if the image has no face or you don't choose any key to detect, it will stop
	if not af_obj or not parts:
		return
	af_dict = af_obj[0]
	x, y, width, height = af_dict.face.pos.x, af_dict.face.pos.y, af_dict.face.pos.width, af_dict.face.pos.height
	draw = dr.Draw(source_img)

	draw.rectangle(((x,y), (x+width, y+height)), outline = 'green')
	draw.text((x,y), 'face')

	if af_dict.likelihood < 0.95:
		return
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
		draw.rectangle(((x,y), (x+width, y+height)), outline = 'green')
		#text shadows the face too much
		#draw.text((x,y), 'right_eye')

	if not os.path.exists('./detect'):
		os.mkdir('./detect')

	source_img.save('./detect/{}'.format(img_name),'JPEG')


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

input_path = os.path.join(config.data_storage, config.data_path)
file_lst = file_list(input_path, ('.jpg','.png'), True)

i = 0
num_file = len(file_lst)
for file_path in file_lst[:1000]:
	find_face(file_path, ['face', 'eye'])
	i += 1
	"""
	if i % 500 == 0:
		print('{0:03f}% done'.format(i/num_file))
	"""