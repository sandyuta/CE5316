def predict(weights, x):
	"""Return dot product of `weights` and feature vector `x`.

	Expects `weights` and `x` to be iterables of the same length.
	"""
	return sum(w * xi for w, xi in zip(weights, x))


def mean_squared_error(y_true, y_pred):
	"""Compute mean squared error between two sequences."""
	n = len(y_true)
	if n == 0:
		return 0.0
	return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n


if __name__ == "__main__":
	# tiny self-check
	w = [2.0, 1.0]
	x = (1, 0)
	print("predict:", predict(w, x))
