import cv2
import numpy as np
import scipy.misc
import scipy.ndimage


def transform_RG_XOR(image, key):
	RG = key  # Random matrix
	fv = cv2.bitwise_xor(image.astype(np.uint8), RG.astype(np.uint8))

	fvs = scipy.ndimage.filters.median_filter(fv, size=(5, 5), footprint=None, output=None, mode='reflect',
	                                          cval=0.0, origin=0)

	# fvs = np.array(fvs).ravel()

	return fvs
