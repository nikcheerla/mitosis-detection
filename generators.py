
#code for data generation from heatmaps

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
plt.ioff()
import glob, sys, os, random, time, logging, threading, subprocess
import scipy.io, scipy.misc

from sklearn.cross_validation import train_test_split
import progressbar
from keras.metrics import binary_crossentropy, binary_accuracy

from utils import evaluate_model_on_directory

import IPython


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


def totuple(a):
	try:
		return tuple(totuple(i) for i in a)
	except TypeError:
		return a


def tolist(a):
	try:
		return list(totuple(i) for i in a)
	except TypeError:
		return a

def normalize(arr):
	return (arr - arr.mean())/arr.std()





class AbstractGenerator(object):
	def __init__(self, train_image_list, val_image_list, samples_per_epoch=500, val_samples=100, batch_size=50, verbose=True):
		self.train_image_list = train_image_list
		self.val_image_list = val_image_list
		self.samples_per_epoch = samples_per_epoch
		self.val_samples = val_samples
		self.batch_size = batch_size
		self.verbose = verbose

		self.load()

	def load(self):
		raise NotImplementedError()

	def checkpoint(self):
		#cmd = ['python', 'evaluate.py', '-w', self.model.checkpoint, '-f', 
		#	'datasets/icpr/train/A01_03_image.jpg', '-o', 'datasets/icpr/train/A01_03_pred.jpg']
		print "HI Check"

	def data(self, mode='train'):
		raise NotImplementedError()




class TrainValSplitGenerator(AbstractGenerator):
	def __init__(self, directory, input_prefix="_image.jpg", output_prefix="_heatmap.jpg", output_type='binary', split=0.15, **kwargs):
		self.input_prefix = input_prefix
		self.output_prefix = output_prefix
		self.output_type = output_type
		self.directory = directory

		files_list = glob.glob(directory + "/*" + self.input_prefix)
		#print (directory + "/*" + self.input_prefix)

		train_list, val_list = train_test_split(files_list, test_size=split)
		super(TrainValSplitGenerator, self).__init__(train_list, val_list, **kwargs)




class AbstractHeatmapGenerator(TrainValSplitGenerator):
	def __init__(self, *args, **kwargs):
		super(AbstractHeatmapGenerator, self).__init__(*args, **kwargs)

	def load(self):
		self.img = {}
		self.hmap = {}
		for img_file in np.append(self.train_image_list, self.val_image_list):
			heatmap_file = img_file[:-len(self.input_prefix)] + self.output_prefix
			#print (img_file, heatmap_file)

			if img_file[-4:] == ".npy":
				self.img[img_file] = np.load(img_file)
			else:
				self.img[img_file] = (scipy.misc.imread(img_file))/255.0

			if self.output_type == 'binary':
				self.hmap[img_file] = (scipy.misc.imread(heatmap_file) > 128).astype(int)
			else:
				self.hmap[img_file] = (scipy.misc.imread(heatmap_file)/ 255.0)
				print (np.max(self.hmap[img_file]))

	def data(self, mode='train'):
		file_list = self.train_image_list if mode == 'train' else self.val_image_list
		num_samples = self.samples_per_epoch if mode == 'train' else self.val_samples

		

		generator_embed = self.gen_sample_pair(file_list, mode=mode)

		window, target = next(generator_embed)
		window, target = np.array(window), np.array(target)
		batch_data = np.zeros((num_samples, ) + window.shape, dtype=window.dtype)
		batch_target = np.zeros((num_samples, ) + target.shape, dtype=target.dtype)

		for i in range(0, num_samples):
			window, target = next(generator_embed)
			batch_data[i] = window
			batch_target[i] = target

		generator_embed.close()

		return batch_data, batch_target

	def checkpoint(self, model): pass

	def gen_sample_pair(self, files_list):

		# All generators must implement gen_sample pair -- returns a
		# pair (window, target) that represents a data/target pair
		raise NotImplementedError()






class BootstrapGenerator(AbstractHeatmapGenerator):
	def __init__(self, *args, **kwargs):
		self.mitosis_sampling = kwargs.pop("mitosis_sampling", None)
		self.normal_sampling = kwargs.pop("normal_sampling", None)
		self.source_image_size = kwargs.pop("source_image_size", None)
		self.pred_prefix = kwargs.pop("pred_prefix", "_inter.jpg")
		super(BootstrapGenerator, self).__init__(*args, **kwargs)

	def load(self):

		image_list = sorted(np.append(self.train_image_list, self.val_image_list))
		
		self.img = {}
		self.hmap = {}
		self.predmap = {}

		if self.source_image_size is not None:
			dtypes_float = []
			dtypes_int = []
			for img_file in image_list:
				dtypes_float.append((img_file, np.float32))
				dtypes_int.append((img_file, int))

			self.img = np.zeros((self.source_image_size, self.source_image_size, 3), dtype=dtypes_float)
			self.hmap = np.zeros((self.source_image_size, self.source_image_size), dtype=dtypes_int)
			self.predmap = np.zeros((self.source_image_size, self.source_image_size), dtype=dtypes_float)

		for img_file in image_list:
			print (img_file, len(self.img))
			heatmap_file = img_file[:-len(self.input_prefix)] + self.output_prefix
			pred_file = img_file[:-len(self.input_prefix)] + self.pred_prefix
			#print (img_file, heatmap_file)
			self.img[img_file] = (scipy.misc.imread(img_file))/255.0
			self.hmap[img_file] = (scipy.misc.imread(heatmap_file) > 128).astype(int)

			if os.path.exists(pred_file):
				self.predmap[img_file] = scipy.misc.imread(pred_file)/255.0
				

	def checkpoint(self, model, mode='train'):
		try:
			evaluate_model_on_directory(model, self.directory, suffix='_inter.jpg')
		except KeyboardInterrupt:
			pass
		self.load()







if __name__ == "__main__":
	print ("Hi")



		