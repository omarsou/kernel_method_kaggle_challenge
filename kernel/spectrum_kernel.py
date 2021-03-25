from kernel.base_kernel import KernelIPExplicit, SumKernelIPExplicit
from utils.utils import reverse_complementaire


class SpectrumKernelError(BaseException):
    pass


class SpectrumKernel(KernelIPExplicit):
    def __init__(self, k, add_inverse=False):
        super().__init__()
        self.k = k
        self.name = f"spectrum_kernel_{k}_"
        self.add_inverse = add_inverse

    def make_phi(self, item, train=True):
        if len(item) < self.k:
            raise SpectrumKernelError("Sequence is too short, try to reduce k")

        dict_ = dict()
        for i in range(len(item) - self.k + 1):
            target = item[i: i + self.k]
            if target not in dict_.keys():
                dict_[target] = 1
            else:
                dict_[target] += 1
            if self.add_inverse and not train:
                target = reverse_complementaire(target)
                if target not in dict_.keys():
                    dict_[target] = 1
                else:
                    dict_[target] += 1

        return dict_

    def inner_product_phi(self, phi1, phi2):
        keys = list(set(phi1.keys()) & set(phi2.keys()))
        output = 0
        for key in keys:
            output += phi1[key] * phi2[key]
        return output


class SumSpectrumKernelError(BaseException):
    pass


class SumSpectrumKernel(SumKernelIPExplicit):
    def __init__(self, lst_params, add_inverse=False):
        super().__init__(lst_params=lst_params)
        self.add_inverse = add_inverse
        self.lst_params = lst_params
        self.name = f"spectrum_kernel_{self.lst_params}_"

    def make_phi(self, item, train=True, params=None):
        if len(item) < params:
            raise SpectrumKernelError("Sequence is too short, try to reduce k")

        dict_ = dict()
        for i in range(len(item) - params + 1):
            target = item[i: i + params]
            if not self.add_inverse:
                if target not in dict_.keys():
                    dict_[target] = 1
                else:
                    dict_[target] += 1
            else:
                target = reverse_complementaire(target)
                if target not in dict_.keys():
                    dict_[target] = 1
                else:
                    dict_[target] += 1
        return dict_

    def inner_product_phi(self, phi1, phi2):
        keys = list(set(phi1.keys()) & set(phi2.keys()))
        output = 0
        for key in keys:
            output += phi1[key] * phi2[key]
        return output
