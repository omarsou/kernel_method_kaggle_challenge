"""Base class to inherit from."""
import os
import numpy as np
from kernel import KernelIPImplicit, KernelIPExplicit, SumKernelIPExplicit
from utils.solvers import sigmoid


class BaseClassifierError(BaseException):
    pass


class BaseClassifier:
    """Base class"""
    def __init__(self, kernel, save_dir=None, verbose=True, version='Xte1'):
        self.kernel = kernel
        self.save_dir = save_dir
        self.verbose = verbose
        self.version = version
        self.alpha = None

        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

    def fit(self, X, y):
        self.kernel.build_gram_matrix(X)

        if self.verbose:
            print("Gram Matrix Built.")

        if self.save_dir:
            path_2_save = os.path.join(self.save_dir, self.kernel.name + f"_{self.version}" + ".pkl")
            self.kernel.save_kernel(path_2_save)

        self._fit(y)

    def _fit(self, y):
        raise NotImplementedError("Method _fit not implemented")

    def predict_one_sample(self, x_test):
        vector_test = self.kernel.test(x_test)
        return self.alpha.dot(vector_test)

    def predict(self, X):
        if isinstance(self.kernel, KernelIPImplicit):
            return self.predict_implicit(X)
        elif isinstance(self.kernel, KernelIPExplicit) or isinstance(self.kernel, SumKernelIPExplicit):
            return self.predict_explicit(X)
        else:
            raise BaseClassifierError("Kernel Instance not recognized")

    def predict_proba(self, X):
        if isinstance(self.kernel, KernelIPImplicit):
            return self.predict_implicit_proba(X)
        elif isinstance(self.kernel, KernelIPExplicit):
            return self.predict_explicit_proba(X)
        else:
            raise BaseClassifierError("Kernel Instance not recognized")

    def predict_implicit(self, X):
        prediction = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            prediction[i] = self.predict_one_sample(X.loc[i, X.columns[1]])
        output = np.zeros_like(prediction, dtype=int)
        output[prediction < 0] = - 1
        output[prediction >= 0] = 1
        return output

    def predict_implicit_proba(self, X):
        prediction = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            prediction[i] = self.predict_one_sample(X.loc[i, X.columns[1]])
        return sigmoid(np.copy(prediction)), prediction

    def predict_explicit(self, X):
        self.kernel.make_test_phi(X)
        prediction = np.zeros(X.shape[0])
        for idx in range(X.shape[0]):
            prediction[idx] = self.predict_one_sample(idx)
        output = np.zeros_like(prediction, dtype=int)
        output[prediction < 0] = - 1
        output[prediction >= 0] = 1
        return output

    def predict_explicit_proba(self, X):
        self.kernel.make_test_phi(X)
        prediction = np.zeros(X.shape[0])
        for idx in range(X.shape[0]):
            prediction[idx] = self.predict_one_sample(idx)
        return sigmoid(np.copy(prediction)), prediction