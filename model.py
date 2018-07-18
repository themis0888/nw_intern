import tensorflow as tf
import numpy as np

from tensorflow.contrib import layers

class Tag_Estimation_Network():
	def __init__(self, num_tags=10, trainable=False):
		print("\n Building Model...")
		self.image_height, self.image_width, self.image_channel = [512, 512, 3]
		self.class_num = num_tags

		self.images = tf.placeholder(tf.float32, shape=[None, self.image_height, self.image_width, self.image_channel])
		self.labels = tf.placeholder(tf.float32, shape=[None, self.class_num])
		self.is_training = tf.placeholder(tf.bool, shape=None)

		self.prediction, self.last_conv = self.inference(inputs=self.images, name='Dual_Path_Net_2D')

		is_correct = tf.equal(tf.argmax(self.prediction, axis=1), tf.argmax(self.labels, axis=1))
		self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32), name='Accuracy')
		
		if trainable:
			self.global_step = tf.Variable(-1, trainable=False, name='global_step')

			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.prediction, labels=self.labels), name='Loss')
			# self.loss = self.focal_loss(logits=self.prediction, labels=self.labels, name='Loss')
			tf.add_to_collection('tensorboard', self.loss)
			# tf.add_to_collection('tensorboard', self.accuracy)

			learning_rate = tf.train.exponential_decay(learning_rate=0.001, global_step=tf.train.get_global_step(), \
								decay_steps=10000, decay_rate=0.5, staircase=True)

			with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
				self.train = tf.train.AdamOptimizer(learning_rate, epsilon=0.01).minimize(self.loss, global_step=tf.train.get_global_step())


	def focal_loss(self, logits, labels, name, alpha=0.25, gamma=2):
		from tensorflow.python.ops import array_ops
		sigmoid_p = tf.nn.sigmoid(logits)
		zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

		pos_p_sub = array_ops.where(labels > zeros, labels - sigmoid_p, zeros)
		neg_p_sub = array_ops.where(labels > zeros, zeros, sigmoid_p)
		per_entry_cross_ent = -alpha*(pos_p_sub**gamma)*tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) - \
								(1 - alpha)*(neg_p_sub**gamma)*tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
		return tf.reduce_sum(per_entry_cross_ent, name=name)


	def dual_path_block(self, inputs, num_outputs, kernel_size, connection_depth, is_training, stride=1, activation_fn=tf.nn.leaky_relu):
		conv = layers.conv2d(inputs=inputs, num_outputs=num_outputs, kernel_size=kernel_size, stride=1, activation_fn=None)
		conv = layers.batch_norm(inputs=activation_fn(conv), center=True, scale=True, is_training=is_training)
		conv = layers.conv2d(inputs=inputs, num_outputs=num_outputs, kernel_size=kernel_size, stride=stride, activation_fn=None)	

		dense_connection = tf.concat([inputs[:,:,:,:connection_depth], conv[:,:,:,:connection_depth]], axis=-1)
		if not conv.shape[1:] == inputs.shape[1:]:
			inputs = layers.conv2d(inputs=inputs, num_outputs=num_outputs, kernel_size=stride, stride=stride, activation_fn=None)

		residual_connection = conv[:,:,:,connection_depth:] + inputs[:,:,:,connection_depth:]

		dual_path = activation_fn(tf.concat([dense_connection, residual_connection], axis=-1))
		dual_path = layers.batch_norm(inputs=dual_path, center=True, scale=True, is_training=is_training)
		return dual_path


	def inference(self, inputs, name, reuse=False):
		with tf.variable_scope(name) as scope:
			if reuse == True: scope.reuse_variables()
			
			inputs = layers.conv2d(inputs=inputs, num_outputs=32, kernel_size=7, stride=2, activation_fn=tf.nn.leaky_relu) # 512 512 3 -> 256 256 32

			for i in range(3):
				inputs = self.dual_path_block(inputs=inputs, num_outputs=inputs.shape[-1], kernel_size=3, connection_depth=16, is_training=self.is_training)
			inputs = layers.max_pool2d(inputs=inputs, kernel_size=2, stride=2) # 128 128
			
			for i in range(4):
				inputs = self.dual_path_block(inputs=inputs, num_outputs=inputs.shape[-1], kernel_size=3, connection_depth=24, is_training=self.is_training)
			inputs = layers.max_pool2d(inputs=inputs, kernel_size=2, stride=2) # 64 64

			for i in range(6):
				inputs = self.dual_path_block(inputs=inputs, num_outputs=inputs.shape[-1], kernel_size=3, connection_depth=24, is_training=self.is_training)
			inputs = layers.max_pool2d(inputs=inputs, kernel_size=2, stride=2) # 32 32

			for i in range(8):
				inputs = self.dual_path_block(inputs=inputs, num_outputs=inputs.shape[-1], kernel_size=3, connection_depth=32, is_training=self.is_training)

			last_conv = inputs # 32 32 256
			inputs = tf.reduce_mean(last_conv, axis=[1,2], keepdims=True) # GAP 1 1 256

			class_weights = tf.get_variable(name='class_weights', shape=[inputs.shape[-1], self.class_num], initializer=layers.xavier_initializer())
			inputs = tf.matmul(layers.flatten(inputs), class_weights)
			return inputs, last_conv


	"""
	def get_heat_map(self, inputs, labels):
		assert len(inputs.shape)==4 # DHWC
		inputs = tf.reshape(inputs, [1, np.prod(inputs.shape[:-1]), inputs.shape[-1]])
		
		with tf.variable_scope('Dual_Path_Network_3D', reuse=True):
			w = tf.gather(params=tf.transpose(tf.get_variable('class_weights')), indices=tf.argmax(labels, 1))
			w = tf.expand_dims(w, axis=-1)
				
		heat_map = tf.matmul(inputs, w)
		heat_map = tf.reshape(heat_map, [self.image_depth, self.image_height, self.image_width])
		return heat_map
	"""