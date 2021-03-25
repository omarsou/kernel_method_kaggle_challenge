from kernel.base_kernel import KernelIPExplicit
import itertools as it
from utils.utils import reverse_complementaire


class SubstringKernelError(BaseException):
    pass


class SubstringKernel(KernelIPExplicit):
    def __init__(self, k=4, delta=1, m=1):
        super().__init__()
        self.k = k
        self.delta = delta
        self.name = f"_substring_kernel_{k}_{m}_"
        self.m = m

    def make_phi(self, item, train=True):
        if len(item) < self.k:
            raise SubstringKernelError("Sequence is too short, try to reduce k")
        return item

    def inner_product_phi(self, phi1, phi2):
        L = len(phi1)
        s, t = phi1, phi2
        p1 = sum(((self.miss(s[i:i + self.k], t[d + i:d + i + self.k]) <= self.m) 
                  for i, d in it.product(range(L - self.k + 1), range(-self.delta, self.delta + 1)) 
                  if i + d + self.k <= L and i + d >= 0))
        t = reverse_complementaire(t)
        p2 = sum(((self.miss(s[i:i + self.k], t[d + i:d + i + self.k]) <= self.m)
                  for i, d in it.product(range(L - self.k + 1), range(-self.delta, self.delta + 1))
                  if i + d + self.k <= L and i + d >= 0))
        return p1 + p2

    @staticmethod
    def miss(s, t):
        """ Count the number of mismatches between two strings."""
        return sum((si != sj for si, sj in zip(s, t)))