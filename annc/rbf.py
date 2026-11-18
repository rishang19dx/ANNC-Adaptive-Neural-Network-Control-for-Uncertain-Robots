"""Gaussian radial-basis-function regressor S(Z)."""
import itertools

import numpy as np


class GaussianRBF:
    def __init__(self, centers, width):
        self.centers = np.asarray(centers, dtype=float)   # (N, dim)
        self.inv_two_w2 = 1.0 / (2.0 * width**2)

    @classmethod
    def grid(cls, dim, levels=(-1.0, 1.0), width=2.0):
        centers = np.array(list(itertools.product(levels, repeat=dim)))
        return cls(centers, width)

    @property
    def size(self):
        return self.centers.shape[0]

    def __call__(self, Z):
        d2 = np.sum((self.centers - Z) ** 2, axis=1)
        return np.exp(-d2 * self.inv_two_w2)
