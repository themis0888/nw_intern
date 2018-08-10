import animeface as af 
import PIL.Image as im
import PIL.ImageDraw as dr

key_list = ['face', 'skin', 'hair', 'left_eye', 'right_eye', 'mouth', 'nose', 'chin']

# find_face: str, 
def find_face(img_path):
    source_img = im.open(img_path).convert('RGBA')

    cords = af.detect(source_img)
    cord = cord[0]
    draw = dr.Draw(source_img)
    draw.rectangle()