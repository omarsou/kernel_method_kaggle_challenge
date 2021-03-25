import numpy as np


def lr_solver(K, W, Y, lambda_):
	n = W.shape[0]
	Wsqrt = np.sqrt(np.diag(W))
	return Wsqrt.dot(np.linalg.solve(Wsqrt.dot(K).dot(Wsqrt) \
		   + n * lambda_ * np.identity(n), Wsqrt.dot(Y)))


def sigmoid(x):
	'''
	Numerically stable sigmoid function, prevents overflow in exp
	'''
	positive = x >= 0
	negative = x < 0
	xx = 1 / (1 + np.exp(- x[positive]))
	x[positive] = 1 / (1 + np.exp(- x[positive]))
	z = np.exp(x[negative])
	x[negative] = z / (z + 1)
	return x
