import numpy as np

from PreProcessing_FeatureExtraction.connect_center import connect_centres
from PreProcessing_FeatureExtraction.detect_vein_center_assign_score import compute_vein_score
from PreProcessing_FeatureExtraction.label import binaries
from PreProcessing_FeatureExtraction.normalize import normalize_data
from PreProcessing_FeatureExtraction.preprocessing import remove_hair
from PreProcessing_FeatureExtraction.profile_curvature import compute_curvature

"""
	In this method, the local maximum curvature is calculated in the cross-sectional 
	profile of all four directions, then selecting the profile that has the maximum 
	depth in the cross-sectional profile. And then to get the full pattern of nerves 
	we add, The result of four directions.

	Miura et al. Proposed a three-step algorithm to solve the above problem.

	Step in Algorithms:
	
	Extraction of the center positions of veins.
	Connection of the center positions.
	Labeling of the image.
"""


def vein_pattern(image, kernel_size, sigma):

	"""
	In this method, the local maximum curvature is calculated in the cross-sectional
	profile of all four directions, then selecting the profile that has the maximum
	depth in the cross-sectional profile. And then to get the full pattern of nerves
	we add, The result of four directions.

	Miura et al. Proposed a three-step algorithm to solve the above problem.

	Step in Algorithms:

	Extraction of the center positions of veins.
	Connection of the center positions.
	Labeling of the image.

	:param image:
	:param kernel_size:
	:param sigma:
	:return: vein_pattern
	"""
	# data conversion to float.
	data = np.asarray(image, dtype=float)
	print("data shape", np.shape(data))
	# data preprocessing with remove hair.
	filter_data = remove_hair(data, kernel_size)

	# converting data to zero mean normalize form.
	preprocessed_data = normalize_data(filter_data, 0, 255)

	# detecting the rough location of vein. it reduce time complexity.
	#vein_mask = LeeMask(preprocessed_data)

	# STEP 1-1st: checking profile is dent or not using kappa value.
	kappa = compute_curvature(preprocessed_data, sigma=sigma)

	# STEP 1-2, 1-3, 1-4: assigning probabilistic score based on kappa values.
	score = compute_vein_score(kappa)

	# STEP 2nd: Connecting the center based on score.
	conect_score = connect_centres(score)

	# STEP 3rd: thresholding pattenrn based on median value.
	threshold = binaries(np.amax(conect_score, axis=2))

	# multiplying original data to binarise data to produce thick vein.
	vein_pattern = np.multiply(image, threshold, dtype=float)

	return vein_pattern

# test
"""
import cv2
import matplotlib.pyplot as plt

image_path = '../sample dataset/input/s1/2017232_R_5.jpg'
image = cv2.imread(image_path, 0)
processed_image = vein_pattern(image, 6, 8)

plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1,2,2)
plt.imshow(processed_image, cmap='gray')
plt.title('Processed Image')

plt.suptitle("Vein Pattern")
plt.tight_layout()
plt.savefig("vein_pattern_extracted.png")
plt.show()

"""
