import numpy as np
import pickle


class Kernel:
    def __init__(self):
        self.train_phi = None
        self.K_matrix = None
        self.test_phi = None
        self.X_train = None
        pass

    def build_gram_matrix(self, X):
        raise NotImplementedError("Method build_gram_matrix not implemented.")

    def test(self, x):
        raise NotImplementedError("Method test not implemented.")

    def save_kernel(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_kernel(path):
        with open(path, "rb") as f:
            kernel_class = pickle.load(f)
        return kernel_class


class KernelIPExplicit(Kernel):
    def __init__(self):
        super().__init__()

    def build_gram_matrix(self, X):
        n = X.shape[0]
        output = np.zeros((n, n))
        self.train_phi = list()
        for i in range(n):
            item = X.loc[i, X.columns[1]]
            self.train_phi.append(self.make_phi(item))

        for i in range(n):
            for j in range(i, n):
                value = self.inner_product_phi(self.train_phi[i], self.train_phi[j])
                output[i, j] = output[j, i] = value

        self.K_matrix = output

    def test(self, indice_x):
        n = len(self.train_phi)
        output = np.zeros(n)
        for i in range(n):
            output[i] = self.inner_product_phi(self.train_phi[i], self.test_phi[indice_x])
        return output

    def make_test_phi(self, X):
        n = X.shape[0]
        self.test_phi = []
        for i in range(n):
            item = X.loc[i, X.columns[1]]
            self.test_phi.append(self.make_phi(item, train=False))
        return

    def make_phi(self, item, train=True):
        raise NotImplementedError("Method make_phi not implemented.")

    def inner_product_phi(self, phi1, phi2):
        raise NotImplementedError("Method inner_product_phi not implemented.")


class KernelIPImplicit(Kernel):
    def __init__(self):
        super().__init__()

    def build_gram_matrix(self, X):
        n = X.shape[0]
        self.X_train = X
        output = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                value1, value2 = X.loc[i, X.columns[1]], X.loc[j, X.columns[1]]
                output[i, j] = output[j, i] = self.K(value1, value2)
        self.K_matrix = output

    def test(self, x):
        X = self.X_train
        n = X.shape[0]
        output = np.zeros(n)
        for i in range(n):
            output[i] = self.K(X.loc[i, X.columns[1]], x)

    def K(self, item1, item2):
        raise NotImplementedError("Method K not implemented")


class SumKernel:
    def __init__(self):
        self.train_phi = list()
        self.K_matrix = None
        self.test_phi = None
        self.X_train = None
        pass

    def build_gram_matrix(self, X):
        raise NotImplementedError("Method build_gram_matrix_sum not implemented.")

    def build_gram_matrix_one(self, X, param):
        raise NotImplementedError("Method build_gram_matrix not implemented.")

    def test(self, x):
        raise NotImplementedError("Method test not implemented.")

    def save_kernel(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_kernel(path):
        with open(path, "rb") as f:
            kernel_class = pickle.load(f)
        return kernel_class


class SumKernelIPExplicitError(BaseException):
    pass


class SumKernelIPExplicit(SumKernel):
    def __init__(self, lst_params):
        super().__init__()
        if not isinstance(lst_params, list):
            raise SumKernelIPExplicitError("If you want to use only one param, you should use the individual param "
                                           "class method.")
        self.lst_params = lst_params

    def build_gram_matrix(self, X):
        n = X.shape[0]
        output = np.zeros((n, n))
        for params in self.lst_params:
            intermediate_output, train_phi = self.build_gram_matrix_one(X, params)
            self.train_phi.append(train_phi)
            output += intermediate_output
        self.K_matrix = output

    def build_gram_matrix_one(self, X, params):
        n = X.shape[0]
        output = np.zeros((n, n))
        train_phi = list()
        for i in range(n):
            item = X.loc[i, X.columns[1]]
            train_phi.append(self.make_phi(item, True, params))

        for i in range(n):
            for j in range(i, n):
                value = self.inner_product_phi(train_phi[i], train_phi[j])
                output[i, j] = output[j, i] = value

        return output, train_phi

    def test(self, indice_x):
        n = len(self.train_phi[0])
        output = np.zeros(n)
        for idx, params in enumerate(self.lst_params):
            current_output = 0
            for i in range(n):
                current_output += self.inner_product_phi(self.train_phi[idx][i], self.test_phi[idx][indice_x])
        return output

    def make_test_phi(self, X):
        n = X.shape[0]
        self.test_phi = []
        for params in self.lst_params:
            current_test_phi = list()
            for i in range(n):
                item = X.loc[i, X.columns[1]]
                current_test_phi.append(self.make_phi(item, train=False, params=params))
            self.test_phi.append(current_test_phi)
        return

    def make_phi(self, item, train=True, params=None):
        raise NotImplementedError("Method make_phi not implemented.")

    def inner_product_phi(self, phi1, phi2):
        raise NotImplementedError("Method inner_product_phi not implemented.")
