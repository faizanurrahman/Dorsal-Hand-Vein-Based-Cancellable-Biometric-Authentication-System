import numpy


def binaries(G):
	"""
	After connecting the veins center, we get a vein pattern in each direction.
	Then it is binaries using a median filter, for each corresponding pixel location,
	we calculate the median value. If the corresponding pixel is smaller than the calculated
	median, then it is the part of the background, and if the value of the pixel is larger or
	equal to the calculated median than it is the part of the vein pattern.
	Finally, we merge all four direction patterns into one by the corresponding pixel is
	replaced by the calculated median value at vein location.
	:param G:
	:return:
	"""
	# take 1-D array and return bool mask based on its median value.
	median = numpy.median(G[G > 0])
	Gbool = G > median
	return Gbool.astype(numpy.float64)
