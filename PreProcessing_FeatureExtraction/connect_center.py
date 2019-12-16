import numpy


def connect_profile_1d(vein_prob_1d):
	"""
	connect a 1d profile probabilistic score
	:param vein_prob_1d:
	:return: connected center
	"""

	return numpy.amin([numpy.amax([vein_prob_1d[3:-1], vein_prob_1d[4:]], axis=0),
	                   numpy.amax([vein_prob_1d[1:-3], vein_prob_1d[:-4]], axis=0)], axis=0)


def connect_centres(vein_score):
	""" 
	Connects vein centres by filtering vein probabilities V

	To connect the center position, to get a continues vein pattern,
	and to remove the noisy location of veins, we perform the following step-

	1.let's consider the horizontal direction, at any center location, say pixel(x, y),
	2. We consider two neighbor pixels, one in the right-hand side and another pixel on
	the left-hand side.
	3. If the value at both neighborhood right and left of the pixel(x,y) is large than
	the pixel value, then a horizontal line is drawn to form a continuous vein pattern,
	and if the values at both neighborhoods is less than the pixel(x,y),
	it is than considered as a noise, in this case, pixel (x,y) value is set to zero.
	4.Similarly, we calculate values in all directions to get continues patterns and remove noise.

	.. math::
		b[w] = min(max(a[w+1], a[w+2]) + max(a[w-1], a[w-2]))
	:type vein_score: object
	:param vein_score: all direction vein score
	:return connected_center: connected center in all direction
	"""
	#print("vein_score: ", vein_score.shape)
	connected_center = numpy.zeros(vein_score.shape, dtype='float64')
	temp = numpy.zeros((400, 400), dtype=numpy.float)
	temp = vein_score[..., 0] + vein_score[..., 1] + vein_score[..., 2] + vein_score[..., 3]
	vein_score = temp
	# Horizontal direction
	for index in range(vein_score.shape[0]):
		connected_center[index, 2:-2, 0] = connect_profile_1d(vein_score[index, :])

	# Vertical direction
	for index in range(vein_score.shape[1]):
		connected_center[2:-2, index, 1] = connect_profile_1d(vein_score[:, index])

	#print(vein_score.shape)
	# Direction: 45 degrees (\)
	i, j = numpy.indices(vein_score.shape)
	border = numpy.zeros((2,), dtype='float64')
	for index in range(-vein_score.shape[0] + 5, vein_score.shape[1] - 4):
		connected_center[:, :, 2][i == (j - index)] = numpy.hstack(
			[border, connect_profile_1d(vein_score.diagonal(index)), border])

	# Direction: -45 degrees (/)
	Vud = numpy.flipud(vein_score)
	Cdud = numpy.flipud(connected_center[:, :, 3])
	for index in reversed(range(vein_score.shape[1] - 5, -vein_score.shape[0] + 4, -1)):
		Cdud[:, :][i == (j - index)] = numpy.hstack([border, connect_profile_1d(Vud.diagonal(index)), border])

	return connected_center


"""
# test

image_path = '../sample dataset/input/s1/2017232_R_0.jpg'
image = cv2.imread(image_path, 0)
processed_image = remove_hair(image, 4)
K = compute_curvature(processed_image, 4)
score = compute_vein_score(K)
connected_score = connect_centres(score)
# K[K <= 0] = 0
# K /= K.max()

plt.subplot(3, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(3, 2, 2)
plt.imshow(processed_image, cmap='gray')
plt.title('Processed Image')

plt.subplot(3, 2, 3)
plt.imshow(connected_score[..., 0], cmap='gray')
plt.title('Connect Horizontal')

plt.subplot(3, 2, 4)
plt.imshow(connected_score[..., 1], cmap='gray')
plt.title('Connect Vertical')

plt.subplot(3, 2, 5)
plt.imshow(connected_score[..., 2], cmap='gray')
plt.title('Connect +45 degree')

plt.subplot(3, 2, 6)
plt.imshow(connected_score[..., 3], cmap='gray')
plt.title('Connect -45 degree')
plt.suptitle("connected score")
plt.tight_layout()
plt.savefig('connected_score_in_all_direction.png')
plt.show()

"""
