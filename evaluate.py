# Evaluates on an image or on a group of images

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import glob, sys, os, random, argparse

from models import ImgToImgModel


parser = argparse.ArgumentParser(
        description='main script for evaluating mitoses heatmaps')
parser.add_argument('-w', '--weights-file', type=str, default=None, help='model weights file location')
parser.add_argument('-d', '--image-dir', type=str, default=None, help='directory of image files')
parser.add_argument('-f', '--image-file', type=str, default=None, help='single image file to evaluate')
parser.add_argument('-o', '--image-out-file', type=str, default=None, help='single image file to evaluate')

args = parser.parse_args()

model = ImgToImgModel.load(args.weights_file)

if args.image_dir == None:
	img = plt.imread(args.image_file)

	hmap = model.evaluate(img, overlap=model.window_size/2)
	plt.imsave(args.image_out_file, hmap, cmap=plt.get_cmap("Oranges"))

else:
	files = glob.glob(args.image_dir + "/*image.jpg")

	for image_file in files:
		hmap_file = image_file[:-9] + "heatmap_pred.jpg"
		plt.imsave(hmap_file, hmap, cmap=plt.get_cmap("Oranges"))