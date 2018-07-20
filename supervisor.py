import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpuDevice', type=str, dest='gpu_device', default='0')
parser.add_argument('--dataDir', type=str, dest='data_path', default='/data/External/danbooru2017/512px/')
parser.add_argument('--metaDir', type=str, dest='metadata_path', default='/data/External/danbooru2017/metadata/')
parser.add_argument('--dataType', type=str, dest='data_type', default='valid')
parser.add_argument('--logDir', type=str, dest='log_path', default='./checkpoint/')

parser.add_argument('--numTags', type=int, dest='num_tags', default=10)
parser.add_argument('--train', type=lambda x: x.lower() in ('true', '1'), dest='training', default=False)
parser.add_argument('--numSteps', type=int, dest='num_steps', default=0)
parser.add_argument('--batchSize', type=int, dest='batch_size', default=1)
config, unparsed = parser.parse_known_args() 

# =========================================================================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_device

import tensorflow as tf
import tensorflow.contrib.tensorboard.plugins.projector as projector
import sys, glob, logging

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)

class Supervisor():
	def __init__(self, log_path):
		self.log_path = log_path
		self.summary_writer = tf.summary.FileWriter(self.log_path)
		if not os.path.exists(self.log_path):
			os.mkdir(self.log_path)
		self.logger = logging.getLogger()


	def init_scalar(self, scope=None):
		self.summary_scope = scope
		
		summary_variables = tf.get_collection(self.summary_scope)
		for var in summary_variables:
			tf.summary.scalar(var.name, var)
		
		log_files = glob.glob(os.path.join(self.log_path, 'events*')) + glob.glob(os.path.join(self.log_path, '*.log'))
		if len(log_files) >= 1:
			sys.stdout.write(' Previous log files are detected... Remove (y/n)? ')
			if input() == 'y': 
				for file in log_files: os.remove(file)

		self.logger.addHandler(logging.FileHandler(os.path.join(self.log_path, 'training_steps.log')))
		self.logger.addHandler(logging.StreamHandler())
		self.logger.setLevel(logging.DEBUG)


	def add_summary(self, sess, feed_dict):
		summary_variables = tf.get_collection(self.summary_scope)

		summary, values = sess.run([tf.summary.merge_all(), summary_variables], feed_dict=feed_dict)
		self.summary_writer.add_summary(summary, tf.train.global_step(sess, tf.train.get_global_step()))
		
		log_string = ' Step: {0:>4d}'.format(tf.train.global_step(sess, tf.train.get_global_step()))
		for var in zip(summary_variables, values):
			log_string += '   {0}: {1:>0.4f}'.format(var[0].name.split('_')[0].split(':')[0], var[1]) 
		self.logger.debug(log_string)


	def rename_ckpt_path(self):
		ckpt_path = os.path.join(self.log_path, 'model.ckpt')
		print(' Rename ckpt path...\n {0}'.format(ckpt_path))
		
		if os.path.exists(os.path.join(self.log_path, 'model.ckpt.index')):
			ckpt_path = ckpt_path.replace('\\', '/')
		
			with open(os.path.join(self.log_path, 'checkpoint'), 'w') as f:
				f.write('model_checkpoint_path: "' + ckpt_path + '"\n')
				f.write('all_model_checkpoint_paths: "' + ckpt_path + '"\n')

		projector_path = os.path.join(self.log_path, 'projector_config.pbtxt')

		if os.path.exists(projector_path):
			meta_path = os.path.join(self.log_path, 'labels.tsv')
			meta_path = meta_path.replace('\\', '/')

			with open(projector_path, 'w') as f:
				f.write('embeddings {\n\ttensor_name: "embedding:0"\n')
				if os.path.exists(meta_path):
					f.write('\tmetadata_path: "' + meta_path + '"\n}\n')


if __name__ == '__main__':
	supervisor = Supervisor(log_path=config.log_path)
	supervisor.rename_ckpt_path()