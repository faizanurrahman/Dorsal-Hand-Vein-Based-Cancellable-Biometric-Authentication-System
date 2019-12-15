import numpy as np


def normalize_data(x, low=0, high=1, data_type=None):
	x = np.asarray(x, dtype=np.float)
	min_x, max_x = np.min(x), np.max(x)
	x = x - float(min_x)
	x = x / float((max_x - min_x))
	x = x * (high - low) + low
	if data_type is None:
		return np.asarray(x, dtype=np.float)
	return np.asarray(x, dtype=data_type)
