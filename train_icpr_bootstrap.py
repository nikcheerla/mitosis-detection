
#Trains a model on the ICPR database, checkpointing to the specified location

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import glob, sys, os, random, keras

from generators import AbstractHeatmapGenerator, BootstrapGenerator
from utils import bound, evaluate_model_on_directory

import IPython

from external.vgg19 import VGG19
from external.resnet50 import ResNet50
from models import FCNModel

import scipy.misc
from scipy.ndimage.interpolation import rotate
from sklearn.utils import shuffle

window_size = 80


class ICPRBootstrapGenerator(BootstrapGenerator):
	
	def __init_(self, directory, **kwargs):
		super(ICPRHeatmapGenerator, self).__init__(directory, **kwargs)

	def gen_sample_pair(self, files_list, mode='train'):

		image_size = self.hmap[random.choice(files_list)].shape[0]

		while True:
			coord_range = (window_size*1.25//2 + 1, image_size - window_size*1.25//2 - 1)

			normal_queue = self.sample_with_error(target=0, image_files=files_list, coord_range=coord_range, 
				initial_sample_num=self.normal_sampling[0], 
				final_sample_num=int(self.normal_sampling[0]*self.normal_sampling[1]))

			mitosis_queue = self.sample_with_error(target=1, image_files=files_list, coord_range=coord_range, 
				initial_sample_num=self.mitosis_sampling[0], 
				final_sample_num=int(self.mitosis_sampling[0]*self.mitosis_sampling[1]))

			queue = np.append(normal_queue, mitosis_queue, axis=0)
			target = np.append(np.zeros(len(normal_queue)), np.ones(len(mitosis_queue)))
			queue, target = shuffle(queue, target)

			for (image_file, x, y), target_val in zip(queue, target):

				#x, y = x + random.randint(-8, 8), y + random.randint(-8, 8)
				x, y = int(x), int(y)

				window_size_exp = window_size
				if random.randint(0, 1) == 0:
					window_size_exp = int(window_size * random.choice([0.75, 0.8, 1.0, 1.2, 1.25]))

				xs, ys = x - window_size_exp//2, y - window_size_exp//2
				xs, ys = bound((xs, ys), low=1, high=image_size - window_size_exp -1)

				pred = np.array([[[1 - target_val, target_val]]])
				img = self.img[image_file][xs:(xs + window_size_exp), ys:(ys + window_size_exp)]
				img = scipy.misc.imresize(img, (window_size, window_size, 3))/255.0

				if random.randint(0, 1) == 0:
					img = np.rot90(img)

				if random.randint(0, 1) == 0:
					img = np.fliplr(img)

				if random.randint(0, 1) == 0:
					img = np.flipud(img)

				if random.randint(0, 3) == 0:
					angle = random.randint(-25, 25)
					img = rotate(img, angle, reshape=False)

				yield img, pred

	def sample_with_error(self, image_files, target=0, initial_sample_num=100, final_sample_num=20, coord_range=(0, 5)):

		queue = []
		queue_error = []

		while len(queue_error) < initial_sample_num:

			image_file = random.choice(image_files)
			hmap = self.hmap[image_file]

			x = random.randint(*coord_range)
			y = random.randint(*coord_range)

			if target == hmap[x, y]:
				error = 1.0
				if image_file in self.predmap.dtype.names:
					error = abs(target - (self.predmap[image_file][x, y]))
					error += (random.random() - 0.5)*0.0001
				queue.append((image_file, x, y))
				queue_error.append(error)

		idxs = np.argsort(queue_error)[::-1]
		queue = np.array(queue)[idxs][0:final_sample_num]
		return queue





icpr_generator = ICPRBootstrapGenerator("datasets/icpr/train/", normal_sampling=(100000, 0.05), mitosis_sampling=(5000, 0.8),
		source_image_size=2084, split=0.1, samples_per_epoch=34000, val_samples=3000, batch_size=64)

"""
X, Y = icpr_generator.data(mode='train')
IPython.embed()
np.savez_compressed("results/train_data_icpr.npy", X=X, Y=Y)
"""

local_model = VGG19(weights=None, input_shape=(window_size, window_size, 3), 
		classes=2, filter_size=16, pooling='avg', dropout=0.3)
local_model.summary()
#sgd = keras.optimizers.SGD(lr=8e-3, decay=1e-6, momentum=0.8, nesterov=True)
local_model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

model = FCNModel(local_model)
"""
if os.path.exists("results/checkpoint.h5"):
	model = FCNModel.load("results/checkpoint.h5")
	print ("Loaded")
"""

model.train(icpr_generator, epochs=[8, 8, 10, 13, 15, 8, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 4, 3, 3, 3, 1, 1, 1, 1])

