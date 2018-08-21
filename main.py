import tensorflow as tf

from supervisor import *
supervisor = Supervisor(log_path=config.log_path)

from model import *
model = Tag_Estimation_Network(num_tags=config.num_tags, trainable=config.training)

from setup_dataset import *
dataset = Dataset(data_path=config.data_path, metadata_path=config.metadata_path)


def training_model():

	test_loss = tf.get_variable(name='Test-Loss', shape=[], initializer=tf.zeros_initializer())
	test_acc = tf.get_variable(name='Test-Accuracy', shape=[], initializer=tf.zeros_initializer())
	tf.add_to_collection('tensorboard', test_loss)
	tf.add_to_collection('tensorboard', test_acc)

	supervisor.init_scalar(scope='tensorboard')
	sess.run(tf.global_variables_initializer())

	for step in range(config.num_steps):
		images, labels = dataset.train.next_batch(batch_size=config.batch_size)

		sess.run(model.train, feed_dict={model.images: images, model.labels: labels, model.is_training: True})
		sys.stdout.write(' Step: {0:>4d} ...\r'.format(step))

		if step % 50 == 0:
			loss_curr, acc_curr = [], []

			break_count = 0
			while True:
				if break_count >= 100: break

				try: test_images, test_labels = dataset.valid.next_batch(batch_size=10, num_epochs=None)
				except IndexError: break

				break_count += len(test_images)
				
				values = sess.run([model.loss, model.accuracy], feed_dict={model.images: test_images, model.labels: test_labels, model.is_training: False})
				loss_curr.append(values[0])
				acc_curr.append(values[1])
			
			sess.run([test_loss.assign(np.mean(loss_curr)), test_acc.assign(np.mean(acc_curr))])
			supervisor.add_summary(sess=sess, feed_dict={model.images: images, model.labels: labels, model.is_training: True})

		if step % 5000 == 0:
			saver = tf.train.Saver()	
			saver.save(sess=sess, save_path=os.path.join(supervisor.log_path, 'model.ckpt'))


	saver = tf.train.Saver()	
	saver.save(sess=sess, save_path=os.path.join(supervisor.log_path, 'model.ckpt'))
	
	print(' Training Complete...\n')
	sess.close()


def validation():

	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt

	saver = tf.train.Saver()
	saver.restore(sess=sess, save_path=os.path.join(supervisor.log_path, 'model.ckpt'))

	dataset_type = {'train': dataset.train, 'valid': dataset.valid, 'test': dataset.test}
	target_dataset = dataset_type.get(config.data_type)


	
if __name__ == '__main__':
	if config.training: training_model()
	else: validation()