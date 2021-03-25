from classifiers.base_classifier import BaseClassifier
from utils.solvers import sigmoid, lr_solver
import numpy as np


class KernelLogisticRegression(BaseClassifier):
    def __init__(self, kernel, lambda_, save_dir=None, verbose=True, version="Xte.csv", tolerance=1e-5, max_iter=100):
        super().__init__(kernel, save_dir, verbose, version)
        self.lambda_ = lambda_
        self.tolerance = tolerance
        self.max_iter = max_iter

    def _fit(self, y):
        self.alpha = np.random.rand(self.kernel.K_matrix.shape[0])
        previous_alpha = self.alpha
        error = np.inf
        step = 0
        while (error > self.tolerance) and (step < self.max_iter):
            m = self.kernel.K_matrix.dot(self.alpha)
            P = - sigmoid(- y * m)
            W = sigmoid(m) * sigmoid(-m)
            z = m - (P * y) / W
            self.alpha = lr_solver(self.kernel.K_matrix, W, z, self.lambda_)

            error = np.linalg.norm(previous_alpha - self.alpha)
            previous_alpha = self.alpha
            step += 1
        if error < self.tolerance:
            print("Minimum tolerance between two iterations reached")
        if step >= self.max_iter:
            print("Maximum iteration reached")
