
#various Image<->Image Deep CNN models

import numpy as np
import random

from keras.layers import Input, Reshape, Permute, Flatten, Dense, Lambda, Dropout, merge
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, SpatialDropout2D, Cropping2D
from keras.models import Model, load_model
from keras.optimizers import SGD, adadelta
from keras.callbacks import ProgbarLogger, RemoteMonitor, ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K
from keras import objectives

import matplotlib
import matplotlib.pyplot as plt

from keras import backend as K
from models import VAEModel



data_len = 1000
window_size = 128
num_circles = 10
min_radius = 4
max_radius = 25



def load_data(cache='data/circles.npy'):
	if os.path.exists("data/circles.npy"):
		return np.load(cache)

	X = np.zeros((data_len, window_size, window_size))

	for i in range(0, data_len):
		for j in range(0, num_circles):
			cx, cy = random.randint(0, window_size), random.randint(0, window_size)
			radius = random.randint(min_radius, max_radius)

			x_min, x_max = min(0, cx - radius), max(cx - radius, window_size)
			y_min, y_max = min(0, cy - radius), max(cy - radius, window_size)
			for x in range(x_min, x_max):
				for y in range(y_min, y_max):
					if (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2:
						X[i, x, y] = 1

		print i,

	np.save(cache, X)

X = np.load_data()










