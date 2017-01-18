
#generate ICPR heatmap files

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import glob, sys, os, csv

from skimage.filters import threshold_otsu
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve

radius = 4
kernel = np.zeros((2*radius+1, 2*radius+1))
y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
mask = x**2 + y**2 <= radius**2
kernel[mask] = 1
cnt = 0

for image_file, csv_file in zip(sorted(glob.glob("icpr/train/*.bmp")), sorted(glob.glob("icpr/train/*.csv"))):
	
	#print (image_file, csv_file)

	img = plt.imread(image_file)
	img2 = convolve(img.sum(axis=2), kernel, mode='same')
	thresh = threshold_otsu(img2.astype(int)) - 50
	hmap = gaussian_filter(img2 > thresh, 0.5)
	
	#print (hmap)

	plt.imsave(image_file[:-4] + "_image.jpg", img)
	plt.imsave(image_file[:-4] + "_heatmap.jpg", hmap, cmap=plt.get_cmap("Oranges"))

	print (image_file[:-4] + "_image.jpg", image_file[:-4] + "_heatmap.jpg")