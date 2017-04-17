
#generate ICPR heatmap files

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import glob, sys, os, csv, scipy
import scipy.misc

for image_file, csv_file in zip(sorted(glob.glob("icpr/train/*.bmp")), sorted(glob.glob("icpr/train/*.csv"))):
	
	img = scipy.misc.imread(image_file)
	hmap = np.zeros(img.shape[:-1]).astype(int)
	
	csv_link = csv.reader(open(csv_file, 'rb'))
	for row in csv_link:
		for i in range(0, len(row) / 2):
			xv, yv = (int(row[2 * i]), int(row[2 * i + 1]))
			hmap[yv, xv] = 1

	print (hmap)

	scipy.misc.imsave(image_file[:-4] + "_image.jpg", img)
	scipy.misc.imsave(image_file[:-4] + "_heatmap.jpg", hmap)