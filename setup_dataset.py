<<<<<<< HEAD
import numpy as np
import sys, os, glob

import scipy.ndimage.interpolation
import scipy.misc

from sklearn.model_selection import train_test_split
from skimage.transform import resize as skresize

from tag_label_convertor import *

class Dataset():
	def __init__(self, data_path, metadata_path, split=True):
		self.data_path = data_path
		self.metadata_path = metadata_path
		
		print('\n Loading Dataset ...')
		self.metadata = convert(metadata_path=metadata_path)
		
		"""
		self.metadata = {}
		file_list = glob.glob(os.path.join(metadata_path, '*.npy'))
		file_list.sort()

		print(' Reading metadata ...')
		for file in file_list:
			print(' - {0}'.format(file))
			self.metadata.update(np.load(file).item())
		"""

		if split == True:
			train, valid_and_test = train_test_split(list(self.metadata.items()), test_size=0.1, shuffle=False)
			
			self.train = Sub_Dataset(data_path=data_path, meta_dict=dict(train))
			
			valid, test = train_test_split(valid_and_test, test_size=0.5, shuffle=False)
			self.valid = Sub_Dataset(data_path=data_path, meta_dict=dict(valid), shuffle=False)
			self.test = Sub_Dataset(data_path=data_path, meta_dict=dict(test), shuffle=False)
			
		else:
			self.whole_dataset = Sub_Dataset(data_path=data_path, meta_dict=list(self.metadata.items()), shuffle=False)


class Sub_Dataset():
	def __init__(self, data_path, meta_dict, shuffle=True):
		self.data_path, self.meta_dict = data_path, meta_dict
		
		self.key_list = list(self.meta_dict.keys())
		
		self.batch_index, self.num_epochs = 0, 0
		self.shuffle = shuffle
		if shuffle: 
			np.random.shuffle(self.key_list)

		self.data_length = len(self.meta_dict)

	def next_batch(self, batch_size, num_epochs=None):
		data, labels = [], []

		while len(data) < batch_size:
			try: 
				data.append(scipy.misc.imread(os.path.join(self.data_path, self.key_list[self.batch_index]), mode='RGB'))
				one_hot = self.meta_dict[self.key_list[self.batch_index]]
				labels.append(one_hot)

			except IndexError:
				if len(data) > 0: # return data if not empty 
					return np.asarray(data), np.asarray(labels)

				self.num_epochs += 1
				
				if self.num_epochs == num_epochs: # raise error if epoch matches
					self.num_epochs = 0
					raise IndexError('Finished ...')
				
				else:
					self.batch_index = 0
					if self.shuffle:
						np.random.shuffle(self.key_list)
				
			except FileNotFoundError: pass
				
			self.batch_index += 1

		p = np.random.permutation(len(data))
		return np.asarray(data)[p], np.asarray(labels)[p]


if __name__ == '__main__':

	dataset = Dataset(data_path='/data/External/danbooru2017/512px/', metadata_path='/data/External/danbooru2017/metadata/')

	count = 0
	while True:
		try: images, labels = dataset.valid.next_batch(batch_size=1, num_epochs=1)
		except IndexError: break

		count += len(images)
		print(' Image:{0}   Label:{1} -- {2}   Total Count: {3}'.format(images.shape, labels.shape, labels[0], count))
=======
import numpy as np
import sys, os, glob

import scipy.ndimage.interpolation
import scipy.misc

from sklearn.model_selection import train_test_split
from skimage.transform import resize as skresize

from tag_label_convertor import *

class Dataset():
	def __init__(self, data_path, metadata_path, split=True):
		self.data_path = data_path
		self.metadata_path = metadata_path
		
		self.metadata = convert(metadata_path=metadata_path)
		
		"""
		self.metadata = {}
		file_list = glob.glob(os.path.join(metadata_path, '*.npy'))
		file_list.sort()

		print(' Reading metadata ...')
		for file in file_list:
			print(' - {0}'.format(file))
			self.metadata.update(np.load(file).item())
		"""

		if split == True:
			train, valid_and_test = train_test_split(list(self.metadata.items()), test_size=0.1, shuffle=False)
			
			self.train = Sub_Dataset(data_path=data_path, meta_dict=dict(train))
			
			valid, test = train_test_split(valid_and_test, test_size=0.5, shuffle=False)
			self.valid = Sub_Dataset(data_path=data_path, meta_dict=dict(valid), shuffle=False)
			self.test = Sub_Dataset(data_path=data_path, meta_dict=dict(test), shuffle=False)
			
		else:
			self.whole_dataset = Sub_Dataset(data_path=data_path, meta_dict=list(self.metadata.items()), shuffle=False)


class Sub_Dataset():
	def __init__(self, data_path, meta_dict, shuffle=True):
		self.data_path, self.meta_dict = data_path, meta_dict
		
		self.key_list = list(self.meta_dict.keys())
		
		self.batch_index, self.num_epochs = 0, 0
		self.shuffle = shuffle
		if shuffle: 
			np.random.shuffle(self.key_list)

		self.data_length = len(self.meta_dict)

	def next_batch(self, batch_size, num_epochs=None):
		data, labels = [], []

		if self.num_epochs == num_epochs:
			self.num_epochs = 0
			raise IndexError('Finished ...')
			
		next_index = self.batch_index + batch_size
		if next_index >= self.data_length: next_index = self.data_length
		
		i = self.batch_index

		while True:
			if len(data)==batch_size: break
		
			if os.path.exists(os.path.join(self.data_path, self.key_list[i])):

				data.append(scipy.misc.imread(os.path.join(self.data_path, self.key_list[i]), mode='RGB')) # i change
				one_hot = self.meta_dict[self.key_list[i]]
				labels.append(one_hot)

			i += 1
		
		next_index = i

		if next_index == self.data_length:
			self.num_epochs += 1
			self.batch_index = 0
			if self.shuffle: 
				self.file_table = self.file_table.sample(frac=1)
		else:
			self.batch_index = next_index
		
		p = np.random.permutation(len(data))
		return np.asarray(data)[p], np.asarray(labels)[p]


if __name__ == '__main__':

	dataset = Dataset(data_path='/data/External/danbooru2017/512px/', metadata_path='/data/External/danbooru2017/metadata/')

	while True:
		try: images, labels = dataset.valid.next_batch(batch_size=5, num_epochs=1)
		except IndexError: break

		print(' Image:{0}   Label:{1}'.format(images.shape, labels.shape))
>>>>>>> 25dde7d6bb745cc2942cf08c3a746dc7a1c7c407
