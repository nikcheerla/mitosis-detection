
#Trains a model on the ICPR database, checkpointing to the specified location

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import glob, sys, os, random, keras

from generators import AbstractHeatmapGenerator, BootstrapGenerator
from utils import bound, evaluate_model_on_directory

import IPython

from external.vgg19 import VGG19
from models import FCNModel

import scipy.misc
from scipy.ndimage.interpolation import rotate


window_size = 80


class ICPRBootstrapGenerator(BootstrapGenerator):
	
	def __init_(self, directory, **kwargs):
		super(ICPRHeatmapGenerator, self).__init__(directory, **kwargs)

	def gen_sample_pair(self, files_list, mode='train'):
		while True:
			image_file = random.choice(files_list)
			hmap = self.hmap[image_file]
			target = np.random.choice([0, 1], p=[0.5, 0.5])

			window_size_exp = window_size
			if random.randint(0, 1) == 0:
				window_size_exp = int(window_size * random.choice([0.75, 0.8, 1.0, 1.2, 1.25]))
			
			while True:
				x = random.randint(window_size_exp//2 + 1, hmap.shape[0] - window_size_exp//2 - 1)
				y = random.randint(window_size_exp//2 + 1, hmap.shape[0] - window_size_exp//2 - 1)

				if image_file in self.predmap:
					d = (self.predmap[image_file][x, y])

				if target == hmap[x, y]:
					
					break

			#x, y = x + random.randint(-8, 8), y + random.randint(-8, 8)

			xs, ys = x - window_size_exp//2, y - window_size_exp//2
			xs, ys = bound((xs, ys), low=1, high=hmap.shape[0] - window_size -1)

			pred = np.array([[[1 - target, target]]])
			img = self.img[image_file][xs:(xs + window_size_exp), ys:(ys + window_size_exp)]
			img = scipy.misc.imresize(img, (window_size, window_size, 3))/255.0

			if random.randint(0, 1) == 0:
				img = np.rot90(img)

			if random.randint(0, 1) == 0:
				img = np.fliplr(img)

			if random.randint(0, 1) == 0:
				img = np.flipud(img)

			if random.randint(0, 2) == 0:
				angle = random.randint(-25, 25)
				img = rotate(img, angle, reshape=False)

			yield img, pred




icpr_generator = ICPRBootstrapGenerator("datasets/icpr/train/", erf=lambda x:x**2,
		split=0.15, samples_per_epoch=20000, val_samples=2000, batch_size=64)

X, Y = icpr_generator.data(mode='train')
IPython.embed()
np.savez_compressed("results/train_data_icpr.npy", X=X, Y=Y)

local_model = VGG19(weights=None, input_shape=(window_size, window_size, 3), 
		classes=2, filter_size=16, pooling='avg', dropout=0.3)
local_model.summary()
#sgd = keras.optimizers.SGD(lr=8e-3, decay=1e-6, momentum=0.8, nesterov=True)
local_model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

model = FCNModel(local_model)



model.train(icpr_generator, epochs=[5])
evaluate_model_on_directory(model, "datasets/icpr/train/", suffix="_inter.jpg")
model.train(icpr_generator, epochs=[10])
evaluate_model_on_directory(model, "datasets/icpr/train/", suffix="_inter.jpg")
