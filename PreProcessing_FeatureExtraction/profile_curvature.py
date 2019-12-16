import math

import numpy as np
import scipy.ndimage as Image


def compute_curvature(image, sigma):
	"""
	compute the curvature of profile in all 4 cross-section

	[Step 1-1] The objective of this step is, for all 4 cross-sections(z)
	of the image (horizontal, vertical, 45 and -45 ,diagonals) is computed then feed it to valley
	detector kappa function.

	kappa function

		kappa(z) = frac(d**2P_f(z)/dz**2,(1 + (dP_f(z)/dz)**2)**frac(3,2))

	To compute kappa function, first we smooth image using 2-dimensional gaussian
	filter to avoid noise from input dorsal data. We use Steerable Filters to smooth and get derivatives in
	higher order of smooth image, for all direction.

	Computing kappa vally detector function:
		1. construct a gaussian filter(h)
		2. take the first (dh/dx) and second (d^2/dh^2) derivatives of the filter
		3. calculate the first and second derivatives of the smoothed signal using
		derivative kernel's.
		:type image: object
		:param image, sigma:
		:return kappa:
	"""

	# 1. constructs the 2D gaussian filter "h" given the window size

	winsize = np.ceil(4 * sigma)  # enough space for the filter
	window = np.arange(-winsize, winsize + 1)
	X, Y = np.meshgrid(window, window)
	G = 1.0 / (2 * math.pi * sigma ** 2)
	G *= np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))

	# 2. calculates first and second derivatives of "G" with respect to "X"

	G1_0 = (-X / (sigma ** 2)) * G
	G2_0 = ((X ** 2 - sigma ** 2) / (sigma ** 4)) * G
	G1_90 = G1_0.T
	G2_90 = G2_0.T
	hxy = ((X * Y) / (sigma ** 8)) * G

	# 3. calculates derivatives w.r.t. to all directions

	image_g1_0 = 0.1 * Image.convolve(image, G1_0, mode='nearest')
	image_g2_0 = 10 * Image.convolve(image, G2_0, mode='nearest')
	image_g1_90 = 0.1 * Image.convolve(image, G1_90, mode='nearest')
	image_g2_90 = 10 * Image.convolve(image, G2_90, mode='nearest')
	fxy = Image.convolve(image, hxy, mode='nearest')
	image_g1_45 = 0.5 * np.sqrt(2) * (image_g1_0 + image_g1_90)
	image_g1_m45 = 0.5 * np.sqrt(2) * (image_g1_0 - image_g1_90)
	image_g2_45 = 0.5 * image_g2_0 + fxy + 0.5 * image_g2_90
	image_g2_m45 = 0.5 * image_g2_0 - fxy + 0.5 * image_g2_90

	# [Step 1-1] Calculation of curvature profiles

	#Hand_mask = mask.astype('float64')

	return np.dstack([
		(image_g2_0 / ((1 + image_g1_0 ** 2) ** (1.5))),
		(image_g2_90 / ((1 + image_g1_90 ** 2) ** (1.5))),
		(image_g2_45 / ((1 + image_g1_45 ** 2) ** (1.5))),
		(image_g2_m45 / ((1 + image_g1_m45 ** 2) ** (1.5))),
	])

# Test

"""
image_path = '../sample dataset/input/s1/2017232_R_0.jpg'
image = cv2.imread(image_path, 0)
processed_image = remove_hair(image, 4)
K = compute_curvature(processed_image, 4)

K[K <= 0] = 0
K /= K.max()

plt.subplot(3,2,1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(3,2,2)
plt.imshow(processed_image, cmap='gray')
plt.title('Processed Image')

plt.subplot(3,2,3)
plt.imshow(K[...,0], cmap='gray')
plt.title('Processed Horizontal')

plt.subplot(3,2,4)
plt.imshow(K[...,1], cmap='gray')
plt.title('Processed Vertical')

plt.subplot(3,2,5)
plt.imshow(K[...,2], cmap='gray')
plt.title('Processed +45 degree')

plt.subplot(3,2,6)
plt.imshow(K[...,3], cmap='gray')
plt.title('Processed -45 degree')
plt.suptitle("compute curvature")
plt.tight_layout()
plt.savefig('curvature_in_all_direction.png')
plt.show()
"""