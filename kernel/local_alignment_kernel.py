from kernel.base_kernel import KernelIPImplicit
import numpy as np


class LocalAlignmentKernel(KernelIPImplicit):
    def __init__(self, beta=0.5, e=11, d=1, s=np.identity(4), dict_=None):
        super().__init__()
        self.beta = beta
        self.e = e
        self.d = d
        self.s = s
        self.name = '_local_alignement_kernel_'

        if self.s.shape[0] == 4:
            self.dict = {"A": 0, "T": 1, "C": 2, "G": 3}
        else:
            self.dict = dict_

    def K(self, item1, item2):
        m, n = len(item1), len(item2)
        M = np.zeros((m + 1, n + 1))
        X = np.zeros((m + 1, n + 1))
        Y = np.zeros((m + 1, n + 1))
        X2 = np.zeros((m + 1, n + 1))
        Y2 = np.zeros((m + 1, n + 1))

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                M[i, j] = np.exp(self.beta * self.s[self.dict[item1[i - 1]],
                                                    self.dict[item2[j - 1]]]) * (1 + X[i - 1, j - 1] + Y[i - 1, j - 1]
                                                                                 + M[i - 1, j - 1])
                X[i, j] = np.exp(self.beta * self.d) * M[i - 1, j] + \
                          np.exp(self.beta * self.e) * X[i - 1, j]
                Y[i, j] = np.exp(self.beta * self.d) * (M[i, j - 1] + X[i, j - 1]) \
                          + np.exp(self.beta * self.e) * Y[i, j - 1]
                X2[i, j] = M[i - 1, j] + X2[i - 1, j]
                Y2[i, j] = M[i, j - 1] + X2[i, j - 1] + Y2[i, j - 1]
        return 1 + X2[m, n] + Y2[m, n] + M[m, n]
