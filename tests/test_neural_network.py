import os
import time
from math import isclose

import numpy as np
import pytest

from simple_neural_network.neural_network import ANN

# init random number generator
rg = np.random.default_rng()

FOLDER_PATH = os.path.split(__file__)[0]


def remove_file(full_path):
    if not os.path.isfile(full_path):
        raise ValueError(f"Cannot find or not a file: {full_path}")
    os.unlink(full_path)
    # wait for moment before returning
    time.sleep(0.5)


def test_init_obj_not_accepted_hidden_nodes():
    with pytest.raises(TypeError, match=r"`hidden_nodes` must be a tuple of length two"):
        ANN(hidden_nodes=(15,), method="class")

    with pytest.raises(ValueError, match=r"Node count cannot be under one"):
        ANN(hidden_nodes=(10, 0), method="class")

    with pytest.raises(TypeError, match=r"`hidden_nodes` must be a tuple of length two"):
        ANN(hidden_nodes="10,1", method="class")


def test_init_obj_not_accepted_method():
    allowed_met = ", ".join(ANN.allowed_methods)

    with pytest.raises(ValueError, match=rf"`method` must be one of {allowed_met}"):
        ANN((5, 7), "")

    with pytest.raises(TypeError, match=r"Method must be a string"):
        ANN((3, 2), lambda x: x)


def test_init_obj_hidden_nodes_property():
    ann = ANN((3, 3), "classification")
    ann.hidden_nodes = (5, 5)

    with pytest.raises(ValueError, match=r"Node count cannot be under one"):
        ann.hidden_nodes = (2, 0)

    assert ann.hidden_nodes == (5, 5)


def test_init_obj_method_property():
    ann = ANN((3, 3), "reg")
    ann.method = "classification"

    allowed_met = ", ".join(ANN.allowed_methods)

    with pytest.raises(ValueError, match=rf"`method` must be one of {allowed_met}"):
        ann.method = "classify"

    assert ann.method == "class"


def test_init_obj_not_accepted_attributes():
    allowed_opt = ", ".join(ANN.allowed_optimizers)

    with pytest.raises(ValueError, match=rf"`optimizer` must be one of {allowed_opt}"):
        ann = ANN((1, 1), "classification", optimizer="momentum")

    # test that following doesn't produce errors
    ann = ANN((2, 2), "regression")
    assert ann.hidden_nodes == (2, 2)
    assert ann.method == "reg"

    bad_valid_sizes = (0, 1, -1, 2)

    for valid_size in bad_valid_sizes:
        with pytest.raises(ValueError):
            ann = ANN((2, 2), "regression", validation_size=valid_size)


def test_init_obj_different_hidden_activations():
    activations = ANN.allowed_hidden_activations

    for act in activations:
        ann = ANN((1, 1), "class", activation1=act, activation2=act)

        assert ann._afn1.__name__ == act
        assert ann._afn2.__name__ == act
        assert ann._afn3.__name__ == "softmax"


def test_init_obj_output_activation():
    ann = ANN((5, 1), "class")
    assert ann._afn3.__name__ == "softmax"

    ann = ANN((5, 1), "regression")
    assert ann._afn3.__name__ == "identity"


def test_init_obj_decay_and_learning_rates():
    ann = ANN((5, 5), "class")
    # default decay and learning rate
    assert ann._decay_rate == 0.0
    assert ann._learning_rate == 0.1

    ann = ANN((5, 5), "reg", decay_rate=0.5)
    assert ann._decay_rate == 0.5
    assert ann._learning_rate == 1.0

    ann = ANN((5, 5), "reg", learning_rate=5.0)
    assert ann._decay_rate == 0.0
    assert ann._learning_rate == 5.0


def test_init_obj_object_with_all_attributes():
    ann = ANN(
        (10, 1),
        "regression",
        optimizer="adam",
        decay_rate=0.1,
        learning_rate=0,
        lambda_=50,
        early_stop_threshold=100,
        activation1="tanh",
        activation2="elu",
        validation_size=0.99,
        verbose_level="mid",
        random_seed=12345,
    )

    assert ann.hidden_nodes[1] == 1
    assert ann.method == "reg"

    assert ann._afn3.__name__ == "identity"
    assert ann._afn2.__name__ == "elu"

    assert ann._decay_rate == 0.1
    assert ann._learning_rate == 1.0
    assert int(ann._early_stop_thres) == 100


def test_validation_index_mask():
    valid_size = 0.5
    ann = ANN((5, 5), "classification", validation_size=valid_size)

    x_shape = (10, 3)
    val_index_mask = ann._get_val_indices(x_shape[0])

    assert np.sum(val_index_mask) == int(valid_size * x_shape[0])


def test_generate_index_ranges():
    train_size, batch_size = 1000, 150
    ranges = ANN._generate_index_ranges(train_size, batch_size)

    total_range_count = int(np.ceil(train_size / batch_size))
    correct_ranges = [
        (0, 150),
        (150, 300),
        (300, 450),
        (450, 600),
        (600, 750),
        (750, 900),
        (900, 1000),
    ]

    assert len(ranges) == total_range_count

    assert all(
        range.start == corr_range[0] and range.stop == corr_range[1]
        for range, corr_range in zip(ranges, correct_ranges)
    )


def test_generate_one_index_range():
    train_size, batch_size = 500, 500
    ranges = ANN._generate_index_ranges(train_size, batch_size)

    total_range_count = 1
    correct_ranges = [(0, train_size)]

    assert len(ranges) == total_range_count

    assert all(
        range.start == corr_range[0] and range.stop == corr_range[1]
        for range, corr_range in zip(ranges, correct_ranges)
    )


def test_transforming_y():
    x_type = np.array([1.0]).dtype
    y = np.array(["A", "B", "B", "C", "A"])

    y_ohe, y_inv = ANN._transform_y(y, x_type)

    correct_y_inv = np.array([0, 1, 1, 2, 0])
    correct_y_ohe = np.array([[1, 0, 0, 0, 1], [0, 1, 1, 0, 0], [0, 0, 0, 1, 0]], dtype=x_type)

    comp = y_inv == correct_y_inv
    assert np.sum(comp) == len(correct_y_inv)

    comp = correct_y_ohe == y_ohe
    assert np.sum(comp) == correct_y_ohe.size


def test_weights_resetting():
    ann = ANN((10, 15), "regression", optimizer="sgd")

    nc1, nc2 = ann.hidden_nodes

    attr_count, out_node_count = 3, 7
    ann._reset_weights(attr_count, out_node_count)

    assert ann._w["w1"].shape == (nc1, attr_count)
    assert ann._w["w2"].shape == (nc2, nc1)
    assert ann._w["w3"].shape == (out_node_count, nc2)

    assert ann._b["b1"].shape == (nc1, 1)
    assert ann._b["b2"].shape == (nc2, 1)
    assert ann._b["b3"].shape == (out_node_count, 1)


def test_momentum_and_adapter_weights():
    ann = ANN((4, 3), "class", optimizer="adam")

    attr_count, out_node_count = 2, 3
    ann._reset_weights(attr_count, out_node_count)

    for i in range(1, 4):
        assert ann._mom_w[f"w{i}"].shape == ann._s_w[f"w{i}"].shape
        assert ann._mom_b[f"b{i}"].shape == ann._s_b[f"b{i}"].shape


def test_cross_entropy_consistency():
    ANN((5, 5), "class")

    y_inv = np.array([0, 2, 1])
    # y_pred has largest probability for 0 and 1 of y_inv (first and third col)
    y_pred = np.array([[0.6, 0.3, 0.15], [0.2, 0.4, 0.7], [0.2, 0.3, 0.15]])
    # y_pred_other has largest probability for all y_inv values (it's giving correct prediction)
    y_pred_other = np.array([[0.6, 0.3, 0.15], [0.2, 0.3, 0.7], [0.2, 0.4, 0.15]])

    assert np.all(y_pred.sum(axis=0) == 1)
    assert np.all(y_pred_other.sum(axis=0) == 1)

    cost1 = ANN._cross_entropy(y_inv, y_pred)
    cost2 = ANN._cross_entropy(y_inv, y_pred_other)

    assert cost1 >= 0 and cost2 >= 0
    # y_pred_other predictions are "better" than y_pred, it should have smaller cost value
    assert cost1 > cost2


def test_cross_entropy():
    ann = ANN((5, 5), "class")

    y = np.array(["A", "B", "B", "C", "A"])
    y_ohe, y_inv = ANN._transform_y(y, np.float64)

    y_inv_uniq = len(np.unique(y_inv))
    assert y_inv_uniq == 3
    # y_pred must be 3 x 5 (y_inv_uniq x len(y))
    data = rg.normal(size=(y_inv_uniq, len(y)))
    y_pred = ann._afn3(data)
    assert y_pred.shape == y_ohe.shape

    cross_entr = ANN._cross_entropy(y_inv, y_pred)
    assert cross_entr >= 0

    # other way to compute cross-entropy (extra term 1e-9 in ANN cross entropy shouldn't matter)
    mtable = y_ohe * y_pred
    with np.errstate(divide="ignore"):
        m_log = np.where(mtable > 0, np.log(mtable), 0)

    cross_entr_other = -np.sum(m_log) / len(y)

    assert isclose(cross_entr, cross_entr_other, abs_tol=0.01)


def test_cross_entropy_derivative():
    x_type = np.array([1.0]).dtype
    y = np.array(["A", "B", "B", "C"])
    _, y_inv = ANN._transform_y(y, x_type)

    # y_pred must be 3 x 4
    y_pred = np.array([[0.25, 0.0, 0.7, 0.5], [0.5, 0.0, 0.15, 0.5], [0.25, 1.0, 0.15, 0.0]])

    dcross_entr = ANN._dcross_entropy(y_inv, y_pred)
    assert dcross_entr.shape == y_pred.shape

    assert np.all(
        np.isclose(
            dcross_entr,
            np.array(
                [
                    [-0.1875, 0.000, 0.175, 0.125],
                    [0.125, -0.250, -0.2125, 0.125],
                    [0.0625, 0.250, 0.0375, -0.250],
                ]
            ),
            atol=0.001,
        )
    )


def test_mse():
    y = np.array([1, -1, 0, 0])
    y_pred = np.array([-1, 1, 0, -2])

    assert isclose(ANN._mse(y, y_pred), 1.5, abs_tol=0.001)


def test_mse_derivative():
    y = np.array([1, -1, 0, 0])
    y_pred = np.array([-1, 1, 0, -2])

    dmse = ANN._dmse(y, y_pred)
    assert dmse.shape == y.shape

    assert np.all(np.isclose(dmse, np.array([-0.5, 0.5, 0.0, -0.5]), atol=0.001))


def test_eval_cost_with_regularization():
    ann = ANN((2, 2), "class", lambda_=4)

    y_inv = np.array([1, 0])
    y_pred = np.array([[0, 1], [1, 0]])

    ann._w["w1"] = np.ones((2, 1))
    ann._w["w2"] = np.ones((2, 2))
    ann._w["w3"] = np.ones((2, 2))

    # regu_prefix = 1.0 (lambda / 2 * 2)
    regu_penalty = ann._w["w1"].size + ann._w["w2"].size + ann._w["w3"].size

    cost = ann._eval_cost(y_inv, y_pred)
    assert cost >= 0

    cost_without_pen = cost - regu_penalty

    assert isclose(ANN._cross_entropy(y_inv, y_pred), cost_without_pen, abs_tol=0.01)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_fitting_basic_class_model():
    ann = ANN((3, 3), "class", verbose_level="low")

    x = rg.normal(size=(5, 2)) + 0.5
    y = np.array(["red", "blue", "green", "red", "green"])

    w_save_path = os.path.join(FOLDER_PATH, "weights.h5")
    epochs = 2
    batch_size = 2
    use_valid = False
    iters = int(np.ceil(x.shape[0] / batch_size))

    ann.fit(
        x,
        y.reshape(-1, 1),
        epochs=epochs,
        batch_size=batch_size,
        use_validation=use_valid,
        weights_save_path=w_save_path,
    )

    assert ann._use_valid is use_valid
    assert ann._batch_size == batch_size
    assert ann._epochs == epochs
    assert ann._iters == iters

    assert os.path.exists(w_save_path)
    remove_file(w_save_path)
    # file should have been removed by now
    assert not os.path.exists(w_save_path)

    len(ann._train_stats["cost"]) == epochs
    len(ann._train_stats["acc"]) == epochs


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_fitting_custom_class_model():
    ann = ANN(
        (3, 3),
        "class",
        optimizer="adam",
        learning_rate=0.5,
        lambda_=2.5,
        activation1="tanh",
        activation2="elu",
        verbose_level="low",
    )

    x = rg.normal(size=(4, 2)) + 0.5
    y = np.array(["red", "blue", "red", "green"])

    w_save_path = os.path.join(FOLDER_PATH, "weights.h5")
    epochs = 2
    batch_size = 1
    use_valid = False
    iters = int(np.ceil(x.shape[0] / batch_size))

    ann.fit(
        x,
        y.reshape(-1, 1),
        epochs=epochs,
        batch_size=batch_size,
        use_validation=use_valid,
        weights_save_path=w_save_path,
    )

    assert ann._use_valid is use_valid
    assert ann._batch_size == batch_size
    assert ann._epochs == epochs
    assert ann._iters == iters

    assert os.path.exists(w_save_path)
    remove_file(w_save_path)
    # file should have been removed by now
    assert not os.path.exists(w_save_path)

    len(ann._train_stats["cost"]) == epochs
    len(ann._train_stats["acc"]) == epochs


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_fitting_reg_model():
    ann = ANN((3, 3), "reg", verbose_level="low")

    x = rg.normal(size=(5, 2)) + 0.5
    y = np.array([-1.0, 0.5, -0.5, 1.0, 0.5])

    w_save_path = os.path.join(FOLDER_PATH, "weights.h5")
    epochs = 2
    batch_size = 2
    use_valid = True
    val_size = int(ann._val_stats["size"] * x.shape[0])
    iters = int(np.ceil((x.shape[0] - val_size) / batch_size))

    ann.fit(
        x,
        y.reshape(-1, 1),
        epochs=epochs,
        batch_size=batch_size,
        use_validation=use_valid,
        weights_save_path=w_save_path,
    )

    # regression has only one out dimension
    assert ann._w["w3"].shape[0] == 1

    assert ann._use_valid is use_valid
    assert ann._batch_size == batch_size
    assert ann._epochs == epochs
    assert ann._iters == iters

    assert os.path.exists(w_save_path)
    remove_file(w_save_path)
    # file should have been removed by now
    assert not os.path.exists(w_save_path)

    len(ann._train_stats["cost"]) == epochs
    len(ann._train_stats["acc"]) == epochs


def test_predict_reg():
    ann = ANN((3, 2), "reg")

    x = rg.normal(size=(5, 2)) + 0.5
    ann._reset_weights(x.shape[1], out_node_count=1)

    y_pred = ann.predict(x)

    # must have x.shape[0] predictions
    assert y_pred.shape[0] == x.shape[0]


def test_predict_class():
    ann = ANN((3, 2), "class")

    x = rg.normal(size=(5, 2)) + 0.5
    y_categories = 3
    ann._reset_weights(x.shape[1], out_node_count=y_categories)

    y_pred = ann.predict(x)

    # must have x.shape[0] predictions
    assert y_pred.shape[0] == x.shape[0]
    # prediction is in inverse format
    assert np.all((y_pred >= 0) & (y_pred < y_categories))


def test_predict_file_not_found():
    ann = ANN((5, 3), "class")

    x = rg.normal(size=(5, 2))
    file_path = "file_certainly_not_existing.test"

    with pytest.raises(FileNotFoundError, match=rf"Cannot find file {file_path}"):
        ann.predict(x, weights_path=file_path)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_get_fit_results():
    ann = ANN((3, 3), "reg", verbose_level="low")

    x = rg.normal(size=(5, 2)) + 0.5
    y = np.array([-1.0, 0.5, -0.5, 1.0, 0.5])

    w_save_path = os.path.join(FOLDER_PATH, "weights.h5")
    epochs = 3

    ann.fit(
        x,
        y.reshape(-1, 1),
        epochs=epochs,
        use_validation=False,
        weights_save_path=w_save_path,
    )

    assert os.path.exists(w_save_path)
    remove_file(w_save_path)

    results = ann.get_fit_results()
    # validation wasn't used
    mandatory_keys = ("epochs", "train_data", "weights_last_saved_epoch")
    nested_keys = ("smallest_cost", "smallest_cost_epoch", "best_acc", "best_acc_epoch")

    assert isinstance(results, dict)
    results_keys = set(results.keys())
    # no other keys than mandatory
    assert len(results_keys.difference(mandatory_keys)) == 0

    assert results["epochs"] == epochs

    results_nested_keys = set(results["train_data"].keys())
    assert len(results_nested_keys.difference(nested_keys)) == 0
