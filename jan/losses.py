"""Loss functions.

"""

import abc

import numpy as np


class Loss(abc.ABC):

    @abc.abstractclassmethod
    def loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Apply the loss function."""

    @abc.abstractclassmethod
    def gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute the gradient with respect to y_pred."""


class MSE(Loss):
    """Mean Squared Error (MSE) loss function."""

    @staticmethod
    def loss(y_true, y_pred):
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    def gradient(y_true, y_pred):
        return 2 * (y_pred - y_true)
