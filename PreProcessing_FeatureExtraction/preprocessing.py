import os

import cv2
from scipy.signal import convolve2d

from PreProcessing_FeatureExtraction.normalize import normalize_data

dir = os.path.dirname(os.path.realpath(__file__))
# print(dir)
filepath = os.path.join(dir, 'MexicanHatKernalData')


# print(filepath)
def remove_hair(image, mexican_kernel_size, low=1, high=4):

	try:
		read_kernel = cv2.imread(os.path.join(filepath, f'Kernel {mexican_kernel_size}.jpg'), 0)
	#print(read_kernel)
	except FileNotFoundError:
		print('please choose correct size of kernel')

	normalized_kernel = normalize_data(read_kernel, low, high)
	hair_remove = convolve2d(image, normalized_kernel, mode='same', fillvalue=0)
	return hair_remove

# test
"""
image_path = '../sample dataset/input/s1/2017232_R_0.jpg'
image = cv2.imread(image_path, 0)
print("imge", np.shape(image))
processed_image = remove_hair(image, 3)

plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1,2,2)
plt.imshow(processed_image, cmap='gray')
plt.title('Processed Image')

plt.suptitle("Hair Removal")
plt.tight_layout()
plt.show()

"""
