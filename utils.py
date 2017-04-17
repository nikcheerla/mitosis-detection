
#Misc utils, including nuclei detection and image processing

import numpy as np

from scipy import ndimage as nd
import scipy.misc
import skimage as ski
import skimage.io as skio
from skimage import filters
from skimage.exposure import rescale_intensity
from skimage import morphology
import scipy.ndimage.morphology as smorphology
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from scipy.signal import convolve

from skimage.morphology import binary_dilation, binary_erosion

import matplotlib.pyplot as plt
import glob
import IPython

def nuclei_detect_pipeline(img, MinPixel = 200, MaxPixel=2500):
    return nuclei_detection(img, MinPixel, MaxPixel)


def nuclei_detection(img, MinPixel, MaxPixel):
    img_f = ski.img_as_float(img)
    adjustRed = rescale_intensity(img_f[:,:,0])
    roiGamma = rescale_intensity(adjustRed, in_range=(0, 0.5));
    roiMaskThresh = roiGamma < (250 / 255.0) ;

    roiMaskFill = morphology.remove_small_objects(~roiMaskThresh, MinPixel);
    roiMaskNoiseRem = morphology.remove_small_objects(~roiMaskFill,150);
    roiMaskDilat = morphology.dilation(roiMaskNoiseRem, morphology.disk(3));
    roiMask = smorphology.binary_fill_holes(roiMaskDilat)

    hsv = ski.color.rgb2hsv(img);
    hsv[:,:,2] = 0.8;
    img2 = ski.color.hsv2rgb(hsv)
    diffRGB = img2-img_f
    adjRGB = np.zeros(diffRGB.shape)
    adjRGB[:,:,0] = rescale_intensity(diffRGB[:,:,0],in_range=(0, 0.4))
    adjRGB[:,:,1] = rescale_intensity(diffRGB[:,:,1],in_range=(0, 0.4))
    adjRGB[:,:,2] = rescale_intensity(diffRGB[:,:,2],in_range=(0, 0.4))

    gauss = gaussian_filter(adjRGB[:,:,2], sigma=3, truncate=5.0);

    bw1 = gauss>(100/255.0);
    bw1 = bw1 * roiMask;
    bw1_bwareaopen = morphology.remove_small_objects(bw1, MinPixel)
    bw2 = smorphology.binary_fill_holes(bw1_bwareaopen);

    bwDist = nd.distance_transform_edt(bw2);
    filtDist = gaussian_filter(bwDist,sigma=5, truncate=5.0);

    L = label(bw2)
    R = regionprops(L)
    coutn = 0
    for idx, R_i in enumerate(R):
        if R_i.area < MaxPixel and R_i.area > MinPixel:
            r, l = R_i.centroid
            #print(idx, filtDist[r,l])
        else:
            L[L==(idx+1)] = 0
    BW = L > 0
    return BW

def bound(tuple, low=0, high=2000):
    xs, ys = tuple
    return min(max(xs, low), high), min(max(ys, low), high)


def evaluate_model_on_directory(fcn_model, directory, rotations=[0, 1, 2, 3], window_size=[500, 1000], 
            overlap=0, suffix="_pred.jpg", visualize_thresh=True):

    for img_file in sorted(glob.glob(directory + "/*_image.jpg")):
        img = scipy.misc.imread(img_file)/255.0

        preds = []
        for k in rotations:
            for wsize in window_size:
                img2 = np.rot90(img, k=k)
                preds_tmp = fcn_model.evaluate_tiled(img2, window_size=wsize, overlap=overlap)
                preds_tmp = np.rot90(preds_tmp, k=-k)
                preds.append(preds_tmp)
        preds = np.mean(np.array(preds), axis=0)
        print (preds.mean())


        
        pred_file = img_file[:-10] + suffix
        scipy.misc.imsave(pred_file, preds)

        if visualize_thresh:

            preds_thresh = preds < filters.threshold_otsu(preds)
            thresh_file = img_file[:-10] + "_thresh.jpg"
            scipy.misc.imsave(thresh_file, preds_thresh)









if __name__ == "__main__":
    from models import FCNModel
    import sys
    model = FCNModel.load("results/checkpoint.h5")
    evaluate_model_on_directory(model, sys.argv[1], suffix="_inter.jpg")