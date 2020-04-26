"""Stochastic optimizers.

"""

import abc

import numpy as np


class Optimizer(abc.ABC):

    @abc.abstractmethod
    def step(self, weights: np.ndarray, gradients: np.ndarray):
        """Use gradients to weights inplace."""


class SGD(Optimizer):
    """Stochastic Gradient Descent (SGD)."""

    def __init__(self, lr: float):
        self.lr = lr

    def step(self, weights, gradients):
        weights -= self.lr * gradients
