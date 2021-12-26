import numpy as np

from simple_neural_network.metrics import confusion_matrix


def test_confusion_matrix_binary():
    y_true = np.array(["A", "B", "B", "B"])
    # y_pred is the inverse version
    y_pred = np.array([0, 1, 0, 1])

    corr_cmatrix = np.array([[1, 0], [1, 2]])

    cmatrix = confusion_matrix(y_true, y_pred)

    assert cmatrix.shape == (2, 2)
    assert np.alltrue(corr_cmatrix == cmatrix)


def test_confusion_matrix_multi_category():
    y_true = np.array(["C", "A", "C", "B", "A"])
    # y_pred is the inverse version
    y_pred = np.array([1, 0, 2, 1, 2])

    corr_cmatrix = np.array([[1, 0, 1], [0, 1, 0], [0, 1, 1]])

    cmatrix = confusion_matrix(y_true, y_pred)

    assert cmatrix.shape == (3, 3)
    assert np.alltrue(corr_cmatrix == cmatrix)


def test_confusion_matrix_multi_category_other():
    y_true = np.array(["e", "d", "d", "c", "a", "b", "e", "a", "d", "a", "d"])
    # y_pred is the inverse version
    y_pred = np.array([3, 3, 3, 2, 2, 1, 0, 1, 4, 2, 3])

    corr_cmatrix = np.array(
        [[0, 1, 2, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 3, 1], [1, 0, 0, 1, 0]]
    )

    cmatrix = confusion_matrix(y_true, y_pred)

    assert cmatrix.shape == (5, 5)
    assert np.alltrue(corr_cmatrix == cmatrix)
