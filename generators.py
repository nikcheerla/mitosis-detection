
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
	def __init__(self, directory, input_prefix="_image.jpg", output_prefix="_heatmap.jpg", split=0.15, **kwargs):
		self.input_prefix = input_prefix
		self.output_prefix = output_prefix

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
			self.img[img_file] = (scipy.misc.imread(img_file))/255.0
			self.hmap[img_file] = (scipy.misc.imread(heatmap_file) > 128).astype(int)

	def data(self, mode='train'):
		file_list = self.train_image_list if mode == 'train' else self.val_image_list
		num_samples = self.samples_per_epoch if mode == 'train' else self.val_samples

		batch_data, batch_target, multiple_sets, multiple_targets = [], [], False, False

		generator_embed = self.gen_sample_pair(file_list, mode=mode)
		for i in range(0, num_samples):
			window, target = next(generator_embed)
			multiple_sets = type(window) == list
			multiple_targets = type(target) == list
			batch_data.append(window)
			batch_target.append(target)

		generator_embed.close()

		if multiple_sets:
			batch_data_accumulate = []
			for i in range(0, len(batch_data[0])):
				cur_batch = []
				for j in range(0, len(batch_data)):
					cur_batch.append(batch_data[j][i])
				cur_batch = np.array(cur_batch)
				batch_data_accumulate.append(cur_batch)
			batch_data = batch_data_accumulate
		else:
			batch_data = np.array(batch_data)

		if multiple_targets:
			batch_target_accumulate = []
			for i in range(0, len(batch_target[0])):
				cur_batch = []
				for j in range(0, len(batch_target)):
					cur_batch.append(batch_target[j][i])
				cur_batch = np.array(cur_batch)
				batch_target_accumulate.append(cur_batch)
			batch_target = batch_target_accumulate
		else:
			batch_target = np.array(batch_target)

		return batch_data, batch_target

	def gen_sample_pair(self, files_list):

		# All generators must implement gen_sample pair -- returns a
		# pair (window, target) that represents a data/target pair
		raise NotImplementedError()






class BootstrapGenerator(AbstractHeatmapGenerator):
	def __init__(self, *args, **kwargs):
		self.erf = kwargs.pop("erf", lambda x:x**2)
		self.pred_prefix = kwargs.pop("pred_prefix", "_inter.jpg")
		super(BootstrapGenerator, self).__init__(*args, **kwargs)

	def load(self):
		self.img = {}
		self.hmap = {}
		self.predmap = {}
		for img_file in np.append(self.train_image_list, self.val_image_list):
			heatmap_file = img_file[:-len(self.input_prefix)] + self.output_prefix
			pred_file = img_file[:-len(self.input_prefix)] + self.pred_prefix
			#print (img_file, heatmap_file)
			self.img[img_file] = (scipy.misc.imread(img_file))/255.0
			self.hmap[img_file] = (scipy.misc.imread(heatmap_file) > 128).astype(int)

			if os.path.exists(pred_file):
				self.predmap[img_file] = scipy.misc.imread(pred_file)/255.0

	def data(self, mode='train'):
		batch_data, batch_target = super(BootstrapGenerator, self).data(mode=mode)
		self.load()
		return batch_data, batch_target





"""

class BootstrapGenerator(SimpleImageGenerator):


	def __init__(self, window_size, train_list, val_list, threshold=0.3, epochs_per_era=12, checkpoint=None, **kwargs):
		self.threshold = threshold
		self.queue = []
		self.error = []
		self.checkpoint_file = checkpoint
		self.epochs_per_era = epochs_per_era
		self.X, self.Y, _ = plt.imread(train_list[0]).shape
		self.img = []
		self.hmap = []
		for img_file in train_list:
			self.img.append(plt.imread(img_file))
			heatmap_file = img_file[:-9] + "heatmap.jpg"
			self.hmap.append((plt.imread(heatmap_file).sum(axis=2) < 700).astype(int))

		self.img = np.array(self.img)
		self.hmap = np.array(self.hmap)

		super(BootstrapGenerator, self).__init__(window_size, train_list, val_list, **kwargs)

	def checkpoint(self):
		if self.checkpoint_file != None:
			plt.figure(figsize=(8, 8))
			n = 3
			for i in range(n):
				ax = plt.subplot(3, n, i + 1)
				img = self.images_from_coords([self.queue[i]])[0]
				#print(img.shape)
				plt.imshow(img)
				plt.axis('off')
				plt.text(0, 40, "ACCURACY=" + str(round(self.error[i], 6)), fontsize=12, 
					weight="bold", color="black", backgroundcolor="gray")
				ax.get_xaxis().set_visible(False)
				ax.get_yaxis().set_visible(False)

				ax = plt.subplot(3, n, i + n + 1)
				img_rs = np.rollaxis(np.array([img]), 3, 1)
				preds = self.model.predict(img_rs)

				plt.imshow(preds[0, 0])
				plt.axis('off')
				ax.get_xaxis().set_visible(False)
				ax.get_yaxis().set_visible(False)

				ax = plt.subplot(3, n, i + 2*n + 1)
				hmap = self.heatmaps_from_coords([self.queue[i]])[0]

				plt.imshow(hmap[0])
				plt.axis('off')
				ax.get_xaxis().set_visible(False)
				ax.get_yaxis().set_visible(False)


			plt.savefig(self.checkpoint_file, bbox_inches='tight')
			if self.verbose:
				print ("Wrote checkpoint to: ", self.checkpoint_file)

	def random_train_coord(self):
		return (random.randint(self.window_size//2 + 1, self.X - self.window_size//2 - 1),
		                 random.randint(self.window_size//2 + 1, self.Y - self.window_size//2 - 1),
		                 random.randint(0, len(self.train_image_list) - 1))

	def images_from_coords(self, coords):
		patches = []
		for x, y, img_num in coords:
			x1, x2 = x - self.window_size//2, x + (self.window_size - self.window_size//2)
			y1, y2 = y - self.window_size//2, y + (self.window_size - self.window_size//2)
			patches.append(self.img[img_num, x1:x2, y1:y2])
		return np.array(patches)

	def heatmaps_from_coords(self, coords):
		patches = []
		for x, y, img_num in coords:
			x1, x2 = x - self.window_size/2, x + (self.window_size - self.window_size//2)
			y1, y2 = y - self.window_size/2, y + (self.window_size - self.window_size//2)
			patches.append([self.hmap[img_num, x1:x2, y1:y2]])
		return np.array(patches)

	def seed(self):
		self.queue_size = self.samples_per_epoch
		self.queue = []
		while len(self.queue) < self.queue_size:
			self.queue.append(self.random_train_coord())
			self.error.append(0.5)

	def scout(self):
		for i in range(0, self.queue_size):
			self.queue.append(self.random_train_coord())
		imgs, hmaps = self.images_from_coords(self.queue), self.heatmaps_from_coords(self.queue)
		imgs = np.rollaxis(imgs, 3, 1)
		self.error = []
		hmaps_predicted = self.model.predict(imgs)
		class_frac = float(np.sum(hmaps.astype(int)))/len(hmaps.flatten())
		predicted_class_frac = float(np.sum( (hmaps_predicted > class_frac).astype(int)))/len(hmaps_predicted.flatten())
		
		if self.verbose:
			print ("class frac/predicted ", class_frac, predicted_class_frac)

		for hmap, hmap_pred in zip(hmaps, hmaps_predicted):
			hmap[0, 0, 0] = 1.0
			hmap2 = (hmap_pred > class_frac).astype(float)
			hmap2[0, 0, 0] = 1.0
			precision_score = precision(hmap.astype(float), hmap2).eval() + 0.0
			recall_score = recall(hmap.astype(float), hmap2).eval() + 0.0
			f1score = 2*precision_score*recall_score/(precision_score + recall_score)
			print (precision_score, recall_score, f1score)
			self.error.append(f1score)

		self.error = np.array(self.error)
		self.initial_error = np.mean(self.error[0:self.queue_size])

		sort_idx = np.argsort(self.error)
		self.queue = np.array(self.queue)[sort_idx, :]
		self.queue = self.queue[0:self.queue_size]
		self.queue = tolist(self.queue)

		self.error = np.array(self.error)
		self.error = self.error[sort_idx]
		self.error = self.error[0:self.queue_size]

	@threadsafe_generator
	def train(self):
		self.seed()

		era = 1
		while True:
			if self.verbose:
				print ("\nStarting ERA {0} Scouting Stage:".format(era))
			k = 1; self.scout()
			while np.mean(self.error) > self.threshold:
				self.scout()

			if self.verbose:
				print ("Initial accuracy {0:.5f}, took {1} passes to decrease to {2:.5f}".format(self.initial_error, 
					k, np.mean(self.error)))
				print ("\n")

			#np.mean(error) > threshold
			#train in batches on stuff in queue

			for epoch in range(0, self.epochs_per_era):
				for i in range(0, self.queue_size, self.batch_size):
					start, end = i, i + self.batch_size
					img_snap = self.images_from_coords(self.queue[start:end])
					hmap_snap = self.heatmaps_from_coords(self.queue[start:end])
					
					img_snap = np.rollaxis(np.array(img_snap), 3, 1)
					hmap_snap = np.array(hmap_snap)

					self.update_progress(self.batch_size)

					yield img_snap, hmap_snap

			if self.verbose:
				print ("\nERA {0} FINISHED\n".format(era))

			era += 1


"""










if __name__ == "__main__":
	print ("Hi")



		