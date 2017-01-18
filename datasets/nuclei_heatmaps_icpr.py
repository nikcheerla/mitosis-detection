
#generate ICPR heatmap files

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import glob, sys, os, csv

from nuclei_detect import nuclei_detect_pipeline

for image_file, csv_file in zip(sorted(glob.glob("icpr/train/*.bmp")), sorted(glob.glob("icpr/train/*.csv"))):
	
	print (image_file, csv_file)

	img = plt.imread(image_file)
	hmap = nuclei_detect_pipeline(img)
	
	print (hmap)

	plt.imsave(image_file[:-4] + "_image.jpg", img)
	plt.imsave(image_file[:-4] + "_heatmap.jpg", hmap, cmap=plt.get_cmap("Oranges"))