
#Trains a model on the ICPR database, checkpointing to the specified location

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import glob, sys, os, random

from generators import ImageHeatmapGenerator, PreferentialHeatmapGenerator
from models import SimpleConvFilter, SimpleAutoEncoder, VGGNetEncoder


window_size = 256

#icpr_generator = TrainValBootstrapGenerator(window_size, "datasets/icpr/train/", split=0.15, 
#		samples_per_epoch=1000, batch_size=100, threshold=0.3, verbose=True, checkpoint="boot_check.png")

icpr_generator = PreferentialHeatmapGenerator(window_size, "datasets/icpr/train/", split=0.15, 
		samples_per_epoch=1000, val_samples=250, batch_size=32, fraction=0.01)

model = VGGNetEncoder(window_size)

model.train(icpr_generator, epochs=[20, 15, 5, 5, 5])

#img, hmap = icpr_generator.val().next#print ("Score: ", model.model.evaluate(img, hmap))

model.save("model.h5")
