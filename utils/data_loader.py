import numpy as np
import scipy.io as sio
import pdb
import tensorflow as tf
import imagenet_input as data_input

def queue_data(data_path, num_labels, batch_size, data_type, multi_label=False):
	"""
		* To get bigger image data, used queue.
		* Data preprocessing in 'data_input'
			- Random cropping
			- Normalization (by mean)
			- Random horizontal flip (training)

		* Data type
			- 'train'
			- 'val'
			- If not, print error

		* Multi label
			- True or False
			[ + ] num_labels : label + attributes
				  (for ex. in AWA attributes : 1 + 85 = 86)

		* Shape
			- images : [batch_size, 224, 224, 3] (as default)
			- labels : [batch_size, num_labels]

		* Usage:
			batch_images, one_hot_labels = queue_data(data_path, num_labels, batch_size, data_type, multi_label)
			images, labels = sess.run([batch_images, one_hot_labels])
	"""
	#with tf.device('/cpu:0'):
	if data_type=='train':
		print('\tLoading training data from %s' % data_path)
		with tf.variable_scope('train_image'):
			data_images, data_labels = data_input.distorted_inputs(data_path, num_labels, batch_size, multi_label, True)

	elif data_type=='val':
		print('\tLoading validation data from %s' % data_path)
		with tf.variable_scope('test_image'):
			data_images, data_labels = data_input.inputs(data_path, num_labels, batch_size, multi_label, False)
	else:
		raise IOError("Data type is not proper.")

	# make label to one-hot vector
	if multi_label:
		data_labels_one_hot = data_labels[:,1:]

	else:
		data_labels_one_hot = tf.one_hot(data_labels, num_labels)

	return data_images, data_labels_one_hot
