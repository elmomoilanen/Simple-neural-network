import numpy as np

from simple_neural_network.activations import *

MAX_ATOL = 1e-3


def test_tanh():
    arr = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])

    assert np.allclose(tanh(arr), np.array([-0.999, -0.761, 0.0, 0.761, 0.999]), atol=MAX_ATOL)
    assert np.allclose(
        dtanh(arr), np.array([1.81e-04, 4.19e-01, 1.0, 4.19e-01, 1.81e-04]), atol=MAX_ATOL
    )


def test_relu():
    arr = np.array([-1.0, 1.0, 0.0])

    assert np.allclose(relu(arr), np.array([0.0, 1.0, 0.0]), atol=MAX_ATOL)
    assert np.allclose(drelu(arr), np.array([0.0, 1.0, 0.0]), atol=MAX_ATOL)


def test_leaky_relu():
    arr = np.array([-1.0, 1.0, 0.0])

    assert np.allclose(leaky_relu(arr), np.array([-0.01, 1.0, 0.0]), atol=MAX_ATOL)
    assert np.allclose(dleaky_relu(arr), np.array([0.01, 1.0, 1.0]), atol=MAX_ATOL)


def test_elu():
    arr = np.array([-1.0, 1.0, 0.0])

    assert np.allclose(elu(arr), np.array([-0.632, 1.0, 0.0]), atol=MAX_ATOL)
    assert np.allclose(delu(arr), np.array([0.367, 1.0, 1.0]), atol=MAX_ATOL)


def test_sigmoid():
    arr = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

    assert np.allclose(sigmoid(arr), np.array([0.268, 0.377, 0.5, 0.622, 0.731]), atol=MAX_ATOL)
    assert np.allclose(dsigmoid(arr), np.array([0.196, 0.235, 0.25, 0.235, 0.196]), atol=MAX_ATOL)


def test_swish():
    arr = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

    assert np.allclose(swish(arr), np.array([-0.268, -0.188, 0.0, 0.311, 0.731]), atol=MAX_ATOL)
    assert np.allclose(dswish(arr), np.array([0.072, 0.260, 0.5, 0.739, 0.927]), atol=MAX_ATOL)


def test_softmax():
    arr = np.array([[-1.0, 5.0], [0.0, 2.0], [1.0, -1.0]])

    soft = softmax(arr)
    assert np.all((soft >= 0.0) & (soft <= 1.0))
    assert np.all(soft.argmax(axis=0) == np.array([2, 0]))

    # columns must sum up to ones (column represent a prob dist)
    soft_sum = soft.sum(axis=0)
    corr_sum = np.ones(soft_sum.shape, dtype=soft_sum.dtype)

    assert np.allclose(soft_sum, corr_sum, atol=MAX_ATOL)

    dsoft = dsoftmax(arr)
    assert dsoft.shape == arr.shape
    assert np.all((dsoft >= 0.0) & (dsoft <= 1.0))


def test_identity():
    arr = np.arange(-3, 3)

    assert np.all(identity(arr) == arr)
    assert np.all(didentity(arr) == np.ones(arr.shape, dtype=arr.dtype))
