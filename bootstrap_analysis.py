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

data = np.load("results/train_data_era0.npz")
X_train = data["X_train"]
y_train = data["y_train"]

model = VGG19(weights=None, input_shape=(window_size, window_size, 3), classes=2, filter_size=16, pooling='avg', dropout=0.25)

model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5)

data2 = np.load("results/train_data_era1.npz")
X_train2 = data2["X_train"]
y_train2 = data2["y_train"]

print model.evaluate(X_train2, y_train2)

IPython.embed()