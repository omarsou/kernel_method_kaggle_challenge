from classifiers.base_classifier import BaseClassifier
import cvxopt
import numpy as np


class SVMImplementationError(BaseException):
    pass


class SVM(BaseClassifier):
    def __init__(self, kernel, C, save_dir=None, verbose=True, version='Xte'):
        super().__init__(kernel, save_dir, verbose, version)
        self.C = C

    def _fit(self, y):
        cvxopt.solvers.options['show_progress'] = self.verbose
        n = self.kernel.K_matrix.shape[0]
        P = cvxopt.matrix(2 * self.kernel.K_matrix, tc='d')
        q = cvxopt.matrix(- 2 * y, tc='d')
        G = np.concatenate((np.diag(y), np.diag(-y)), axis=0)
        G = cvxopt.matrix(G, tc='d')
        h = np.concatenate((self.C * np.ones((n, 1)), np.zeros((n, 1))), axis=0)
        h = cvxopt.matrix(h, tc='d')

        solution = cvxopt.solvers.qp(P, q, G, h)

        if solution['status'] == "optimal":
            self.alpha = np.array(solution['x']).ravel()
        else:
            raise SVMImplementationError('SVM optimal is not found. Try use a different value of C')


