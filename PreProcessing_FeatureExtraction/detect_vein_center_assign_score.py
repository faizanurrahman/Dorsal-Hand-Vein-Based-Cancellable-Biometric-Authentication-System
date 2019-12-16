import numpy


# calculating probabilistic score on 1-D array.
def profile_score_1d(profile_1d):
	"""
	1. we create a binary array by threshold the original array.
	2. We create a new array, which is moved by a pixel in the right
		direction of the threshold array.
	3. Subtract the new array from the threshold array and get another new array.
	4. When the value of the subtracted array is positive, it means that it is the beginning of curvature,
		and when the value of the array is negative it means that it is the end of curvature,
		we store all the starting and ending pairs. And the width of the curvature is measured
		by the length of the start end pair, and the depth of curvature is measured by the
		maximum value present at the location of the start end pairs in the original array.
		Finally, the location of the center is set by the midpoint of width.

	:param profile_1d
	:return: score_1d
	"""

	threshold_1d = (profile_1d > 0).astype(int)  # calculating mask where profile_1d > 0.
	diff = threshold_1d[1:] - threshold_1d[:-1]  # compute 1-shifted difference
	starts = numpy.argwhere(diff > 0)
	starts += 1  # compensates for shifted different
	ends = numpy.argwhere(diff < 0)
	ends += 1  # compensates for shifted different
	if threshold_1d[0]:
		starts = numpy.insert(starts, 0, 0)
	if threshold_1d[-1]:
		ends = numpy.append(ends, len(profile_1d))

	score_1d = numpy.zeros_like(profile_1d)

	if starts.size == 0 and ends.size == 0:
		return score_1d
	# computing and assigning probabilistic score.
	for start, end in zip(starts, ends):
		maximum = numpy.argmax(profile_1d[int(start):int(end)])
		score_1d[start + maximum] = profile_1d[start + maximum] * (end - start)
	return score_1d


def compute_vein_score(k):
	"""
	Evaluates joint vein centre probabilities from cross-sections

	This function take kappa and calculate vein centre probabilistic score
	based on whether kappa is positive or not. function work as follow:
	it consider each dimension of kappa(horizontal, vertical, diagonal etc.)
	then detect the centres of the veins and then based on width and maximum
	value in depth assign a probabilistic score.

	[Step 1-2] Detection of the centres of veins
	[Step 1-3] Assignment of scores to the centre positions
	[Step 1-4] Calculation of all the profiles

	:type k: object
	:param k: kappa function value for all direction
	:return score: probabilistic score for all direction
	"""

	# we have to return this variable correctly.
	score = numpy.zeros(k.shape, dtype='float64')
	# print(score.shape)
	# Horizontal direction
	for index in range(k.shape[0]):
		score[index, :, 0] += profile_score_1d(k[index, :, 0])

	# Vertical direction
	for index in range(k.shape[1]):
		score[:, index, 1] += profile_score_1d(k[:, index, 1])

	# Direction: 45 degrees (\)
	curve = k[:, :, 2]
	i, j = numpy.indices(curve.shape)  # taking indices of mesh.
	for index in range(-curve.shape[0] + 1, curve.shape[1]):
		score[i == (j - index), 2] += profile_score_1d(curve.diagonal(index))  # assigning value to diagonal.

	# Direction: -45 degrees (/)
	curve = numpy.flipud(k[:, :, 3])  # required so we get "/" diagonals correctly
	Vud = numpy.flipud(score)  # match above inversion
	for index in reversed(range(curve.shape[1] - 1, -curve.shape[0], -1)):
		Vud[i == (j - index), 3] += profile_score_1d(curve.diagonal(index))
	# print("Vud shape", Vud.shape)
	return score


"""
# test

image_path = '../sample dataset/input/s1/2017232_R_0.jpg'
image = cv2.imread(image_path, 0)
processed_image = remove_hair(image, 4)
K = compute_curvature(processed_image, 4)
score = compute_vein_score(K)
#K[K <= 0] = 0
#K /= K.max()

plt.subplot(3, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(3, 2, 2)
plt.imshow(processed_image, cmap='gray')
plt.title('Processed Image')

plt.subplot(3, 2, 3)
plt.imshow(score[..., 0], cmap='gray')
plt.title('Score Horizontal')

plt.subplot(3, 2, 4)
plt.imshow(score[..., 1], cmap='gray')
plt.title('Score Vertical')

plt.subplot(3, 2, 5)
plt.imshow(score[..., 2], cmap='gray')
plt.title('Score +45 degree')

plt.subplot(3, 2, 6)
plt.imshow(score[..., 3], cmap='gray')
plt.title('Score -45 degree')
plt.suptitle("compute score")
plt.tight_layout()
plt.savefig('score_in_all_direction.png')
plt.show()

"""
