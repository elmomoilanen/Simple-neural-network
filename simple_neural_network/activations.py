"""Implements activation function candidates."""
import numpy as np


def tanh(z: np.ndarray) -> np.ndarray:
    exp = np.exp(2.0 * z)
    return (exp - 1.0) / (exp + 1.0)


def dtanh(z: np.ndarray) -> np.ndarray:
    return 1.0 - tanh(z) ** 2


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


def drelu(z: np.ndarray) -> np.ndarray:
    return np.where(z > 0.0, 1.0, 0.0)


def leaky_relu(z: np.ndarray) -> np.ndarray:
    return np.where(z < 0.0, 0.01 * z, z)


def dleaky_relu(z: np.ndarray) -> np.ndarray:
    return np.where(z < 0.0, 0.01, 1.0)


def elu(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0.0, z, np.exp(z) - 1.0)


def delu(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0.0, 1.0, np.exp(z))


def softmax(z: np.ndarray) -> np.ndarray:
    exp = np.exp(z - np.max(z, axis=0))
    return exp / np.sum(exp, axis=0)


def dsoftmax(z: np.ndarray) -> np.ndarray:
    soft = softmax(z)
    return soft * (1.0 - soft)


def identity(z: np.ndarray) -> np.ndarray:
    return z


def didentity(z: np.ndarray) -> np.ndarray:
    return np.ones(z.shape, dtype=z.dtype)
