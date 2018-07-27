"""
CUDA_VISIBLE_DEVICES=1 python feature_extractor.py \
--data_path=/home/siit/navi/data/sample/ \
--list_path=/home/siit/navi/data/sample/meta/ \
--model_name=vgg_19
"""

import pdb
import random, time, os, sys
import numpy as np
import scipy
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import data_loader
import scipy.io as sio

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, dest='data_path', default='/home/siit/navi/data/danbooru2017/256px/')
parser.add_argument('--data_name', type=str, dest='data_name', default='danbooru')
parser.add_argument('--model_name', type=str, dest='model_name', default='vgg_19')
config, unparsed = parser.parse_known_args() 



flags = tf.app.flags
#################################################
########### model configuration #################
#################################################
flags.DEFINE_float('memory_usage', 0.96, 'GRU memory to use')
#################################################
########## network configuration ################
#################################################
flags.DEFINE_integer('n_classes', 50, 'MNIST dataset')

flags.DEFINE_integer('max_iter', 300000, '')
flags.DEFINE_integer('batch_size', 1, '')

flags.DEFINE_integer('train_display', 200, '')
flags.DEFINE_integer('val_display', 1000, '')
flags.DEFINE_integer('val_iter', 100, '')



FLAGS = flags.FLAGS

slim = tf.contrib.slim
vgg = nets.vgg

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.memory_usage)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

awa_train_path = config.data_path + 'meta/path_label_list.txt'
model_path = '/home/siit/navi/data/models/' + config.model_name + '.ckpt'

# num_file : int 
# count the number of input image files
with open(awa_train_path) as f:
    for num_file, l in enumerate(f):
        pass

"""
example) queue_data('/home/siit/navi/data/sample/meta/path_label_list.txt', 
50, 1, 'val',multi_label=False)
"""
trainX, trainY = data_loader.queue_data(
		awa_train_path, FLAGS.n_classes, FLAGS.batch_size, 'val', multi_label=False)

""" TRAINING """
x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
y_ = tf.placeholder(tf.float32, shape=[None, FLAGS.n_classes])
keep_prob = tf.placeholder(tf.float32)

print("\nLoding the model")

# NOTE Main Network
with slim.arg_scope(vgg.vgg_arg_scope()):
	logits, endpoints = vgg.vgg_19(x, num_classes=FLAGS.n_classes, is_training=False)
	feat_fc1 = endpoints['vgg_19/fc7']

all_vars = tf.all_variables()
var_to_restore = [v for v in all_vars if not v.name.startswith('vgg_19/fc8')]

tf.train.start_queue_runners(sess=sess)
saver = tf.train.Saver(var_to_restore)
saver.restore(sess, model_path)

print("\nStart Extracting features")

feat = []
lab = []
for i in range(num_file+1):
	batch_x, batch_y = sess.run([trainX, trainY])
	_, idx = np.nonzero(batch_y)

	feature = sess.run(feat_fc1, feed_dict={x: batch_x, y_: batch_y, keep_prob:1.0})
	feat.append(feature[0][0][0])
	lab.append(idx[0])
	if i%100 == 0:
		print("{} \% done".format(100*i/num_file))


save_path = config.data_path + 'meta/'
if not os.path.exists(save_path):
	os.mkdir(save_path)

sio.savemat(save_path + config.model_name + '_feature_prediction.mat', 
	{'feature': feat, 'label': lab})

print('end')
