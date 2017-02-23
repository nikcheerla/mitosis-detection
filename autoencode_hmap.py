
#Trains a model on the ICPR database, checkpointing to the specified location

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import glob, sys, os, random

from generators import ImageHeatmapGenerator, PreferentialHeatmapGenerator
from models import SimpleConvFilter, SimpleAutoEncoder, VGGNetEncoder, VAEModel


window_size = 128

#icpr_generator = TrainValBootstrapGenerator(window_size, "datasets/icpr/train/", split=0.15, 
#		samples_per_epoch=1000, batch_size=100, threshold=0.3, verbose=True, checkpoint="boot_check.png")

icpr_generator = PreferentialHeatmapGenerator(window_size, "datasets/icpr/train/", split=0.15,
		input_type='heatmap', output_type='heatmap', samples_per_epoch=1000, val_samples=250, batch_size=25, fraction=0.05)

model = VAEModel(window_size=window_size, latent_dim=64, batch_size=25, std=0.04)

model.train(icpr_generator, epochs=300)

#img, hmap = icpr_generator.val().next#print ("Score: ", model.model.evaluate(img, hmap))

model.save("model.h5")



