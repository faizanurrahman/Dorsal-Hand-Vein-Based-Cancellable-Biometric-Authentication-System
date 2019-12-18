# import cv2
# import random
# import os
# from PreProcessing_FeatureExtraction.extract_feature import vein_pattern
import numpy as np
import scipy.misc
import scipy.ndimage


def minmax(X, low, high, minX=None, maxX=None, dtype=np.float):
	X = np.asarray(X)
	if minX is None:
		minX = np.min(X)
	if maxX is None:
		maxX = np.max(X)
	# normalize to [0...1].
	X = X - float(minX)
	X = X / float((maxX - minX))
	# scale to [low...high].
	# X = X * (high-low)
	# X = X + low
	return np.asarray(X, dtype=dtype)


def rescale(fvs, a, b):
	m = fvs.min()
	M = fvs.max()
	y = (b - a) * (fvs - m) / (M - m) + a;
	return y


def renormalize(n, range1, range2):
	delta1 = range1[1] - range1[0]
	delta2 = range2[1] - range2[0]
	return (delta2 * (n - range1[0]) / delta1) + range2[0]


def transformMeximumCurvatureRDM(img, Key1, Key2):
	fvs = np.array(img).reshape(10000, 1)
	fvs = rescale(fvs, 1, 100)
	fv = fvs

	# fv = normalize(fvs)

	fvsParts = np.split(fvs, 2)
	X1 = fvsParts[0]
	Y1 = fvsParts[1]

	Key1 = np.array(Key1)
	KeyParts_1 = np.split(Key1, 2)
	X2 = KeyParts_1[0]
	#    Y2=KeyParts[1]
	Key2 = np.array(Key2)
	KeyParts_2 = np.split(Key2, 2)
	Y2 = KeyParts_2[0]

	X2 = np.array(X2)
	X1 = np.array(X1)
	Y1 = np.array(Y1)
	Y2 = np.array(Y2)

	# transform feature random distance method
	Tf = np.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2)

	TfM = scipy.ndimage.filters.median_filter(Tf, size=5, footprint=None, output=None, mode='reflect', cval=0.0,
	                                          origin=0)

	fv = np.array(fv)
	fv = fv.reshape(10000, 1)
	return TfM, fv
