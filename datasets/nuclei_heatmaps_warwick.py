
#generate WARWICK heatmap files

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import glob, sys, os, csv
import scipy.io, scipy.misc

import IPython


from nuclei_detect import nuclei_detect_pipeline


RADIUS =5

for image_file, csv_file in zip(sorted(glob.glob("warwick_nuclei/train/*/*.bmp")), sorted(glob.glob("warwick_nuclei/train/*/*.mat"))):
	
	print (image_file, csv_file)

	img = scipy.misc.imread(image_file)
	print (img.shape)
	coords = scipy.io.loadmat(csv_file)
	hmap = np.zeros(img.shape[:-1]).astype(int)
	
	for xv, yv in coords['detection']:
		xv = int(xv)
		yv = int(yv)

		for dx in range(-RADIUS, RADIUS):
			for dy in range(-RADIUS, RADIUS):
				xc, yc = xv + dx, yv + dy
				if xc < 0 or xc >= hmap.shape[0] or yc < 0 or yc >= hmap.shape[0]:
					continue
				if (xc - xv)**2 + (yc - yv)**2 < RADIUS**2:
					hmap[yc, xc] = 1

		plt.show()

	scipy.misc.imsave("warwick_nuclei/train/" + os.path.basename(image_file)[:-4] + "_image.jpg", img)
	scipy.misc.imsave("warwick_nuclei/train/" + os.path.basename(image_file)[:-4] + "_heatmap.jpg", hmap)

	"""
	img = plt.imread(image_file)
	hmap = nuclei_detect_pipeline(img)
	
	print (hmap)

	plt.imsave(image_file[:-4] + "_image.jpg", img)
	plt.imsave(image_file[:-4] + "_heatmap.jpg", hmap, cmap=plt.get_cmap("Oranges"))"""