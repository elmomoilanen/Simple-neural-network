"""Implements the neural network class `ANN`.

public methods:
- fit
    Fit the neural network for data X and y.
- predict
    Predict output for test data X.
- plot_fit_results
    Plot cost and accuracy measures from the fitting step.
"""
import os
import math
import time
import logging

from typing import Tuple, Optional

import numpy as np
import h5py
import matplotlib.pyplot as plt

from .activations import (
    tanh,
    dtanh,
    relu,
    drelu,
    leaky_relu,
    dleaky_relu,
    elu,
    delu,
    softmax,
    dsoftmax,
    identity,
    didentity,
)

logger = logging.getLogger(__name__)


class ANN:
    """Two hidden layer fully-connected artificial neural network.

    Params
    ------
    hidden_nodes: tuple, of two int values
        Node counts in the 1st and 2nd hidden layers.

    method: str
        Type of learning, either `class` for classification or `reg` for regression.

    Kwargs
    ------
    optimizer: str, default `sgd`
        Optimization algorithm for learning, either `sgd` or `adam`.

    decay_rate: non-negative float, default 0.0
        Adaptive learning rate can be used during learning and it's determined
        by a formula exp(-decay_rate * epoch). By default, decay rate equals 0
        meaning that this learning rate will be constant one and determined instead
        by the `learning_rate` attribute.

    learning_rate: non-negative float, default 0.1
        If `decay_rate` is zero and hence exp(-decay_rate * epoch) equals one,
        this constant rate will be used with 0.1 as default value. Otherwise,
        when positive decay rate is in use, this rate will not have any significance.

    lambda_: non-negative float, default 0.0
        Strength of L2 regularization, controls the squared l2-norm that
        is added to the cost function. By default equals zero, in which case
        regularization is not applied. The complete regularization term is
        lambda / 2n * squared l2-norm of weights, where n is the observation count.

    early_stop_threshold: int, default unbounded
        Restrict the number of total training passes (epochs) through the network
        by setting this number. Given stop threshold T, if the value of the cost
        function doesn't decrease in T contiguous total passes, training is stopped.
        By default, this threshold is not applied.

    activation1: str, default `relu`
        Name of the activation function between inputs and first hidden layer.
        Options are limited to `tanh`, `relu`, `leaky_relu` or `elu`.

    activation2: str, default `relu`
        Name of the activation function between first and second hidden layers.
        Same options allowed as in the first activation.

    validation_size: float, default 0.2 (20 % of data)
        Defines size of the validation data set, separate from the training data.
        During training the cost function will be evaluated with validation data
        if this attribute value is larger than zero. `fit` method has a parameter
        `use_validation` that determines whether validation is used at all. Thus,
        setting it to False overrides this validation size.

    verbose_level: str, default `high`
        Amount of logging entries. Possible options `high`, `mid` and `low`.
        Default value is `high` which logs data for each training epoch.

    seed: int, default None
        Initial random seed value.

    Examples
    --------
    Example for classification task with synthetic data. 20 % of the data will be used
    for validation and thus the batch size value 50 means that there will be 8
    iterations for every epoch.

    >>> import numpy as np
    >>> from simple_neural_network.neural_network import ANN
    >>> rg = np.random.default_rng()
    >>> X = rg.normal(size=(500, 2))
    >>> y = np.argmax(0.4 * np.sin(X) + 0.6 * np.cos(X) + 0.1, axis=1)
    >>> ann = ANN(hidden_nodes=(40, 15), learning_rate=1.0, activation1="tanh", early_stop_threshold=50)
    >>> ann.fit(X, y.reshape(-1, 1), epochs=500, batch_size=50)
    >>> ann.plot_fit_results()
    """

    allowed_methods = ("class", "reg")
    allowed_optimizers = ("sgd", "adam")
    allowed_hidden_activations = ("tanh", "relu", "leaky_relu", "elu")

    def __init__(
        self, hidden_nodes: Tuple[int, int] = (50, 50), method: str = "class", **kwargs
    ) -> None:
        self.hidden_nodes = hidden_nodes
        self.method = method

        self._optimizer = str(kwargs.get("optimizer", "sgd"))
        if self._optimizer not in self.allowed_optimizers:
            raise ValueError(f"`optimizer` must be one of {', '.join(self.allowed_optimizers)}")

        self._decay_rate = max(float(kwargs.get("decay_rate", 0)), 0.0)
        self._learning_rate = max(float(kwargs.get("learning_rate", 0.1)), 0.0)
        if self._decay_rate > 0:
            self._learning_rate = 1.0

        self._lambda = max(float(kwargs.get("lambda_", 0)), 0.0)
        self._early_stop_thres = float(kwargs.get("early_stop_threshold", math.inf))
        self._stopping_epoch = None

        self._afunc1 = kwargs.get("activation1", "relu")
        self._afunc2 = kwargs.get("activation2", "relu")
        self._afunc3 = "softmax" if self.method == "class" else "identity"

        self._afn1, self._dafn1 = self._set_afunc("hidden", self._afunc1)
        self._afn2, self._dafn2 = self._set_afunc("hidden", self._afunc2)
        self._afn3, self._dafn3 = self._set_afunc("output", self._afunc3)

        self._w = {"w1": np.nan, "w2": np.nan, "w3": np.nan}
        self._b = {"b1": np.nan, "b2": np.nan, "b3": np.nan}

        self._z = {"z1": np.nan, "z2": np.nan, "z3": np.nan}
        self._h = {"h1": np.nan, "h2": np.nan}

        self._train_stats = {"cost": np.nan, "acc": np.nan}

        val_size = float(kwargs.get("validation_size", 0.2))
        if not (val_size > 0 and val_size < 1):
            raise ValueError("`validation_size` must be within (0,1)")

        self._val_stats = {"cost": np.nan, "acc": np.nan, "size": val_size}

        self._batch_size = np.nan
        self._epochs = np.nan
        self._iters = np.nan

        self._weights_save_path = None
        self._use_valid = None

        self._mom_w = {"w1": np.nan, "w2": np.nan, "w3": np.nan}
        self._mom_b = {"b1": np.nan, "b2": np.nan, "b3": np.nan}

        self._s_w = {"w1": np.nan, "w2": np.nan, "w3": np.nan}
        self._s_b = {"b1": np.nan, "b2": np.nan, "b3": np.nan}

        self._mom_beta1 = 0.9
        self._mom_beta2 = 0.999

        self._verbose_level = str(kwargs.get("verbose_level", "high"))

        seed = kwargs.get("seed", None)
        self._rng = np.random.default_rng(seed=seed)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(hidden_nodes={self.hidden_nodes!r}, method={self.method!r})"
        )

    @property
    def hidden_nodes(self):
        return self._hidden_nodes

    @hidden_nodes.setter
    def hidden_nodes(self, nodes):
        if not isinstance(nodes, tuple) or len(nodes) != 2:
            raise TypeError("`hidden_nodes` must be a tuple of length two")

        if any(node <= 0 for node in nodes):
            raise ValueError("node count cannot be under one")

        self._hidden_nodes = nodes

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, method_type):
        if not isinstance(method_type, str):
            raise TypeError("method must be a string")

        method_type = method_type.lower()

        if method_type == "classification":
            method_type = "class"
        elif method_type == "regression":
            method_type = "reg"

        if method_type not in self.allowed_methods:
            raise ValueError(f"`method` must be one of {', '.join(self.allowed_methods)}")

        self._method = method_type

    def _set_afunc(self, layer, func):
        func_name = func.lower()

        if layer == "hidden":
            if func_name not in self.allowed_hidden_activations:
                allowed_activ = ", ".join(self.allowed_hidden_activations)

                raise TypeError(f"Allowed activation functions: {allowed_activ}")

        func_pairs = {
            "tanh": (tanh, dtanh),
            "relu": (relu, drelu),
            "leaky_relu": (leaky_relu, dleaky_relu),
            "elu": (elu, delu),
            "softmax": (softmax, dsoftmax),
            "identity": (identity, didentity),
        }
        return func_pairs[func_name]

    def _get_val_indices(self, x_rows):
        val_size = int(self._val_stats["size"] * x_rows)

        if val_size == 0:
            raise ValueError("Validation data size cannot be zero")

        all_idx = np.arange(x_rows)
        val_idx = self._rng.choice(all_idx, size=val_size, replace=False)

        return np.isin(all_idx, val_idx)

    @staticmethod
    def _transform_y(y, x_type):
        uniq_vals, y_inv = np.unique(y, return_inverse=True)

        y_ohe = np.zeros((uniq_vals.shape[0], y.shape[0]), dtype=x_type)
        y_ohe[y_inv, np.arange(y_ohe.shape[1])] = 1

        # e.g. y has K classes and observations, then y_ohe dim is K x N
        # e.g. y_inv[i]: for column `i` in y_ohe, its y_inv[i] row equals 1, rest rows are zero

        return y_ohe, y_inv

    def _reset_weights(self, attr_count, out_node_count):
        n_count1, n_count2 = self.hidden_nodes

        self._w["w1"] = self._rng.normal(scale=1 / np.sqrt(attr_count), size=(n_count1, attr_count))
        self._w["w2"] = self._rng.normal(scale=1 / np.sqrt(n_count1), size=(n_count2, n_count1))
        self._w["w3"] = self._rng.normal(
            scale=1 / np.sqrt(n_count2), size=(out_node_count, n_count2)
        )

        self._b["b1"] = np.zeros((n_count1, 1))
        self._b["b2"] = np.zeros((n_count2, 1))
        self._b["b3"] = np.zeros((out_node_count, 1))

        if self._optimizer == "adam":
            self._reset_momentum_weights(attr_count, out_node_count)
            self._reset_adapter_weights()

    def _reset_momentum_weights(self, attr_count, out_node_count):
        n_count1, n_count2 = self.hidden_nodes

        self._mom_w["w1"] = np.zeros((n_count1, attr_count))
        self._mom_w["w2"] = np.zeros((n_count2, n_count1))
        self._mom_w["w3"] = np.zeros((out_node_count, n_count2))

        self._mom_b["b1"] = np.zeros((n_count1, 1))
        self._mom_b["b2"] = np.zeros((n_count2, 1))
        self._mom_b["b3"] = np.zeros((out_node_count, 1))

    def _reset_adapter_weights(self):
        self._s_w["w1"] = np.zeros(self._mom_w["w1"].shape)
        self._s_w["w2"] = np.zeros(self._mom_w["w2"].shape)
        self._s_w["w3"] = np.zeros(self._mom_w["w3"].shape)

        self._s_b["b1"] = np.zeros(self._mom_b["b1"].shape)
        self._s_b["b2"] = np.zeros(self._mom_b["b2"].shape)
        self._s_b["b3"] = np.zeros(self._mom_b["b3"].shape)

    def _feedforward(self, X):
        self._z["z1"] = np.matmul(self._w["w1"], X.T) + self._b["b1"]
        self._h["h1"] = self._afn1(self._z["z1"])

        self._z["z2"] = np.matmul(self._w["w2"], self._h["h1"]) + self._b["b2"]
        self._h["h2"] = self._afn2(self._z["z2"])

        self._z["z3"] = np.matmul(self._w["w3"], self._h["h2"]) + self._b["b3"]
        return self._afn3(self._z["z3"])

    @staticmethod
    def _cross_entropy(y_inv, y_pred):
        # y_inv[i] is the correct y inv value for column i (and only value equal to one)
        # filter out probabilities (y_pred values) for which y is zero
        non_zero_y = y_pred[y_inv, np.arange(y_pred.shape[1])]

        # add small term to avoid taking logarithm from zero
        log_y = np.log(non_zero_y + 1e-9)

        return -np.mean(log_y)

    @staticmethod
    def _dcross_entropy(y_inv, y_pred):
        cols = np.arange(y_pred.shape[1])
        non_zero_y = y_pred[y_inv, cols]

        non_zero_y_inv = np.zeros(non_zero_y.shape)

        pos_mask = non_zero_y > 0
        non_zero_y_inv[pos_mask] = -1.0 / non_zero_y[pos_mask]
        non_zero_y_inv[~pos_mask] = 0.0

        non_zero_y_inv /= y_inv.shape[0]

        dmatrix = np.zeros_like(y_pred)
        dmatrix[y_inv, cols] = non_zero_y_inv

        return dmatrix

    @staticmethod
    def _mse(y, y_pred):
        square_of_err = (y.T - y_pred) ** 2
        return np.mean(square_of_err) * 0.5

    @staticmethod
    def _dmse(y, y_pred):
        return (y_pred - y.T) / y.shape[0]

    def _eval_cost(self, y_inv, y_pred):
        if self._lambda > 0:
            # apply L2 regularization
            regu_prefix = self._lambda / (2.0 * y_inv.shape[0])

            w1_sofs = np.sum(np.reshape(self._w["w1"] ** 2, -1))
            w2_sofs = np.sum(np.reshape(self._w["w2"] ** 2, -1))
            w3_sofs = np.sum(np.reshape(self._w["w3"] ** 2, -1))

            regu_penalty = regu_prefix * (w1_sofs + w2_sofs + w3_sofs)
        else:
            regu_penalty = 0

        if self.method == "class":
            return self._cross_entropy(y_inv, y_pred) + regu_penalty

        # for mse, y and y_inv are the same
        return self._mse(y_inv, y_pred) + regu_penalty

    def _eval_dcost(self, y_inv, y_pred):
        if self.method == "class":
            return self._dcross_entropy(y_inv, y_pred)

        return self._dmse(y_inv, y_pred)

    def _eval_acc(self, y_inv, y_pred):
        if self.method == "class":
            return np.sum(np.argmax(y_pred, axis=0) == y_inv) / y_inv.shape[0]

        ss_total = np.sum((y_inv - y_inv.mean()) ** 2)
        ss_resid = np.sum((y_inv.T - y_pred) ** 2)

        if ss_total < 1e-6:
            return 0.0

        return max(1.0 - (ss_resid / ss_total), 0.0)

    def _compute_predict(self, X):
        h1_pred = self._afn1(np.matmul(self._w["w1"], X.T) + self._b["b1"])
        h2_pred = self._afn2(np.matmul(self._w["w2"], h1_pred) + self._b["b2"])

        return self._afn3(np.matmul(self._w["w3"], h2_pred) + self._b["b3"])

    def _save_weights(self):
        if os.path.isdir(self._weights_save_path):
            save_file = os.path.join(self._weights_save_path, "weights.h5")
        else:
            save_file = self._weights_save_path

        with h5py.File(save_file, "w") as file:
            file.create_dataset("w3", data=self._w["w3"])
            file.create_dataset("w2", data=self._w["w2"])
            file.create_dataset("w1", data=self._w["w1"])
            file.create_dataset("b3", data=self._b["b3"])
            file.create_dataset("b2", data=self._b["b2"])
            file.create_dataset("b1", data=self._b["b1"])

    def _load_weights(self, weights_path):
        with h5py.File(weights_path, "r") as file:
            self._w["w3"] = file["w3"][:]
            self._w["w2"] = file["w2"][:]
            self._w["w1"] = file["w1"][:]
            self._b["b3"] = file["b3"][:]
            self._b["b2"] = file["b2"][:]
            self._b["b1"] = file["b1"][:]

    def _backpropagate(self, X, y_inv, y_pred, epoch):
        """Backpropagation algorithm to train the network.

        Recall that the layer L values of the network are computed
        in the following manner

        h_L = afunc_L(matmul(weights_L, h_(L-1)) + bias_L)

        where the h_0 term must be X', where ' denotes the tranpose
        operation.

        For every layer L (hidden and output layers), compute
        the error (or delta) at that level and then the gradient
        of the weights recursively, starting from the output level,
        as follows

        error_L = Dafunc_L * matmul(weights_(L+1)', error_(L+1))
        ∇ weights_L = matmul(error_L, h_(L-1)')

        where Dafunc_L is the derivative of the layer L activation function and
        * is the Hadamard product. Note that the output level error is given as

        error = Dafunc * ∇ costfunc

        With these formulas, the errors are propagated backwards and avoid
        duplicate multiplications. Hence the algorithm name backpropagation.

        At the end, weights must be updated and this really depends on the algorithm
        which can be e.g. the standard stochastic gradient descent or Adam. See the
        _update_w methods for further information.
        """
        delta3 = self._eval_dcost(y_inv, y_pred) * self._dafn3(self._z["z3"])
        self._update_w3(delta3, epoch)

        delta2 = np.matmul(self._w["w3"].T, delta3) * self._dafn2(self._z["z2"])
        self._update_w2(delta2, epoch)

        delta1 = np.matmul(self._w["w2"].T, delta2) * self._dafn1(self._z["z1"])
        self._update_w1(delta1, X, epoch)

    def _update_w3(self, delta3, epoch):
        eps = self._learning_rate * np.exp(-self._decay_rate * epoch) / self._batch_size

        if self._lambda > 0:
            l2_reg = self._w["w3"] * (self._lambda / delta3.shape[0])
        else:
            l2_reg = 0.0

        w3_grad = np.matmul(delta3, self._h["h2"].T) + l2_reg
        b3_grad = np.sum(delta3, axis=1).reshape(-1, 1)

        if self._optimizer == "sgd":
            self._w["w3"] = self._w["w3"] - eps * w3_grad
            self._b["b3"] = self._b["b3"] - eps * b3_grad

        else:  # adam
            self._mom_w["w3"] = (
                self._mom_w["w3"] * self._mom_beta1 + (1.0 - self._mom_beta1) * w3_grad
            )
            self._mom_b["b3"] = (
                self._mom_b["b3"] * self._mom_beta1 + (1.0 - self._mom_beta1) * b3_grad
            )

            self._s_w["w3"] = (
                self._s_w["w3"] * self._mom_beta2 + (1.0 - self._mom_beta2) * w3_grad * w3_grad
            )
            self._s_b["b3"] = (
                self._s_b["b3"] * self._mom_beta2 + (1.0 - self._mom_beta2) * b3_grad * b3_grad
            )

            exp = epoch if epoch > 0 else 1

            mom_w3_new = self._mom_w["w3"] / (1.0 - self._mom_beta1 ** exp)
            mom_b3_new = self._mom_b["b3"] / (1.0 - self._mom_beta1 ** exp)

            s_w3_new = self._s_w["w3"] / (1.0 - self._mom_beta2 ** exp)
            s_b3_new = self._s_b["b3"] / (1.0 - self._mom_beta2 ** exp)

            self._w["w3"] = self._w["w3"] - eps * (
                mom_w3_new / (np.where(s_w3_new > 0, np.sqrt(s_w3_new), 0.0) + 1.0e-9)
            )
            self._b["b3"] = self._b["b3"] - eps * (
                mom_b3_new / (np.where(s_b3_new > 0, np.sqrt(s_b3_new), 0.0) + 1.0e-9)
            )

    def _update_w2(self, delta2, epoch):
        eps = self._learning_rate * np.exp(-self._decay_rate * epoch) / self._batch_size

        if self._lambda > 0:
            l2_reg = self._w["w2"] * (self._lambda / delta2.shape[0])
        else:
            l2_reg = 0.0

        w2_grad = np.matmul(delta2, self._h["h1"].T) + l2_reg
        b2_grad = np.sum(delta2, axis=1).reshape(-1, 1)

        if self._optimizer == "sgd":
            self._w["w2"] = self._w["w2"] - eps * w2_grad
            self._b["b2"] = self._b["b2"] - eps * b2_grad

        else:  # adam
            self._mom_w["w2"] = (
                self._mom_w["w2"] * self._mom_beta1 + (1.0 - self._mom_beta1) * w2_grad
            )
            self._mom_b["b2"] = (
                self._mom_b["b2"] * self._mom_beta1 + (1.0 - self._mom_beta1) * b2_grad
            )

            self._s_w["w2"] = (
                self._s_w["w2"] * self._mom_beta2 + (1.0 - self._mom_beta2) * w2_grad * w2_grad
            )
            self._s_b["b2"] = (
                self._s_b["b2"] * self._mom_beta2 + (1.0 - self._mom_beta2) * b2_grad * b2_grad
            )

            exp = epoch if epoch > 0 else 1

            mom_w2_new = self._mom_w["w2"] / (1.0 - self._mom_beta1 ** exp)
            mom_b2_new = self._mom_b["b2"] / (1.0 - self._mom_beta1 ** exp)

            s_w2_new = self._s_w["w2"] / (1.0 - self._mom_beta2 ** exp)
            s_b2_new = self._s_b["b2"] / (1.0 - self._mom_beta2 ** exp)

            self._w["w2"] = self._w["w2"] - eps * (
                mom_w2_new / (np.where(s_w2_new > 0, np.sqrt(s_w2_new), 0.0) + 1.0e-9)
            )
            self._b["b2"] = self._b["b2"] - eps * (
                mom_b2_new / (np.where(s_b2_new > 0, np.sqrt(s_b2_new), 0.0) + 1.0e-9)
            )

    def _update_w1(self, delta1, X, epoch):
        eps = self._learning_rate * np.exp(-self._decay_rate * epoch) / self._batch_size

        if self._lambda > 0:
            l2_reg = self._w["w1"] * (self._lambda / delta1.shape[0])
        else:
            l2_reg = 0.0

        w1_grad = np.matmul(delta1, X) + l2_reg
        b1_grad = np.sum(delta1, axis=1).reshape(-1, 1)

        if self._optimizer == "sgd":
            self._w["w1"] = self._w["w1"] - eps * w1_grad
            self._b["b1"] = self._b["b1"] - eps * b1_grad

        else:  # adam
            self._mom_w["w1"] = (
                self._mom_w["w1"] * self._mom_beta1 + (1.0 - self._mom_beta1) * w1_grad
            )
            self._mom_b["b1"] = (
                self._mom_b["b1"] * self._mom_beta1 + (1.0 - self._mom_beta1) * b1_grad
            )

            self._s_w["w1"] = (
                self._s_w["w1"] * self._mom_beta2 + (1.0 - self._mom_beta2) * w1_grad * w1_grad
            )
            self._s_b["b1"] = (
                self._s_b["b1"] * self._mom_beta2 + (1.0 - self._mom_beta2) * b1_grad * b1_grad
            )

            exp = epoch if epoch > 0 else 1

            mom_w1_new = self._mom_w["w1"] / (1.0 - self._mom_beta1 ** exp)
            mom_b1_new = self._mom_b["b1"] / (1.0 - self._mom_beta1 ** exp)

            s_w1_new = self._s_w["w1"] / (1.0 - self._mom_beta2 ** exp)
            s_b1_new = self._s_b["b1"] / (1.0 - self._mom_beta2 ** exp)

            self._w["w1"] = self._w["w1"] - eps * (
                mom_w1_new / (np.where(s_w1_new > 0, np.sqrt(s_w1_new), 0.0) + 1.0e-9)
            )
            self._b["b1"] = self._b["b1"] - eps * (
                mom_b1_new / (np.where(s_b1_new > 0, np.sqrt(s_b1_new), 0.0) + 1.0e-9)
            )

    @staticmethod
    def _generate_index_ranges(train_size, batch_size):
        if batch_size > train_size:
            raise ValueError("Batch size cannot be larger than train size")

        if batch_size <= 0:
            raise ValueError("Batch size must be at least one")

        return [
            range(start, min(start + batch_size, train_size))
            for start in range(0, train_size, batch_size)
        ]

    def _log_status(self, epoch, elapsed_time, log_type="train"):
        print_log = False

        if epoch == self._epochs:
            print_log = True
        elif self._verbose_level == "high":
            print_log = True
        elif self._verbose_level == "mid" and epoch % 10 == 0:
            print_log = True
        elif epoch % 50 == 0:
            print_log = True

        if log_type == "train" and print_log:
            elapsed_m, elapsed_s = elapsed_time // 60, elapsed_time % 60

            logger.info("####################")
            logger.info(
                f"epoch: {epoch}/{self._epochs}, elapsed time: {elapsed_m:.0f} m. {elapsed_s:.2f} s."
            )
            logger.info(f"train cost: {self._train_stats['cost'][epoch - 1]:.3f}")
            logger.info(f"train acc: {self._train_stats['acc'][epoch - 1]:.3f}")

        elif log_type == "valid" and print_log:
            logger.info(f"valid cost: {self._val_stats['cost'][epoch - 1]:.3f}")
            logger.info(f"valid acc: {self._val_stats['acc'][epoch - 1]:.3f}")

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, **kwargs) -> None:
        """Fit the neural network.

        Epoch is one total pass of all the training data, batch size determines the count of
        training data in one pass. For n observations (X.shape[0]), there are ceil(n / batch_size)
        iterations in one total pass.

        Params
        ------
        X: NumPy array
            Numerical data matrix of shape n x p, n observations and p attributes.

        y: NumPy array
            Dependent variable with numerical or categorical values of shape n x 1.

        epochs: int
            Count of total feedforward/backpropagation passes through the network.

        Kwargs
        ------
        batch_size: int
            Count of training data in one pass, by default X.shape[0]. For batch size k (k <= X.shape[0]),
            there will be ceil(X.shape[0] / k) iterations in one epoch. Smallest allowed batch size is one.

        use_validation: bool
            True if a separate validation data should be created, False otherwise. True by default.

        weights_save_path: str
            Save path for best weights/biases in terms of minimizing the cost function, by default the current
            working directory (cwd) and with file name `weights.h5`.
        """
        if len(X.shape) != 2:
            raise ValueError("Give array `X` as n x p, NumPy's .reshape(1, -1) might be helpful")
        if not (len(y.shape) == 2 and y.shape[1] == 1):
            raise ValueError("Give array `y` as n x 1, NumPy's .reshape(-1, 1) might be helpful")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Dimension mismatch for arrays, must be X.shape[0] == y.shape[0]")

        self._use_valid = bool(kwargs.get("use_validation", True))
        self._epochs = epochs
        self._weights_save_path = str(kwargs.get("weights_save_path", os.getcwd()))

        self._train_stats["cost"], self._train_stats["acc"] = np.zeros(epochs), np.zeros(epochs)
        train_cost_min = np.inf

        if self._use_valid:
            val_mask = self._get_val_indices(X.shape[0])
            X, X_val = X[~val_mask], X[val_mask]
            y, y_val = y[~val_mask], y[val_mask]

            self._val_stats["cost"], self._val_stats["acc"] = np.zeros(epochs), np.zeros(epochs)
            val_cost_min = np.inf

        self._batch_size = max(min(int(kwargs.get("batch_size", X.shape[0])), X.shape[0]), 1)
        x_indices = np.arange(X.shape[0])

        idx_ranges = self._generate_index_ranges(X.shape[0], self._batch_size)
        self._iters = len(idx_ranges)

        if self.method == "class":
            y, y_inv = self._transform_y(y, x_type=X.dtype)
            out_node_count = y.shape[0]

            if self._use_valid:
                y_val, y_val_inv = self._transform_y(y_val, x_type=X_val.dtype)
        else:
            out_node_count = 1
            y_inv = y

            if self._use_valid:
                y_val_inv = y_val

        self._reset_weights(X.shape[1], out_node_count)

        no_upgrade_counter = 0
        start_timestamp = time.perf_counter()

        logger.info(f"begin training, doing {self._iters} iterations for {self._epochs} epochs")

        for epoch in range(1, self._epochs + 1):
            epoch_rows = self._rng.choice(x_indices, size=len(x_indices), replace=False)
            X, y_inv = X[epoch_rows], y_inv[epoch_rows]

            for iter_ in range(self._iters):
                batch_rows = idx_ranges[iter_]

                X_batch, y_inv_batch = X[batch_rows], y_inv[batch_rows]
                y_pred_batch = self._feedforward(X_batch)

                self._backpropagate(X_batch, y_inv_batch, y_pred_batch, epoch)

            y_pred_train = self._compute_predict(X)
            self._train_stats["cost"][epoch - 1] = self._eval_cost(y_inv, y_pred_train)
            self._train_stats["acc"][epoch - 1] = self._eval_acc(y_inv, y_pred_train)

            self._log_status(epoch, elapsed_time=time.perf_counter() - start_timestamp)

            if self._use_valid:
                y_val_pred = self._compute_predict(X_val)

                self._val_stats["cost"][epoch - 1] = self._eval_cost(y_val_inv, y_val_pred)
                self._val_stats["acc"][epoch - 1] = self._eval_acc(y_val_inv, y_val_pred)

                self._log_status(epoch, None, log_type="valid")

                if self._val_stats["cost"][epoch - 1] < val_cost_min:
                    val_cost_min = self._val_stats["cost"][epoch - 1]
                    self._save_weights()
                    no_upgrade_counter = 0
                else:
                    no_upgrade_counter += 1
                    if no_upgrade_counter >= self._early_stop_thres:
                        logger.info(
                            f"early stop threshold {self._early_stop_thres} reached, stop training"
                        )
                        break
            else:
                if self._train_stats["cost"][epoch - 1] < train_cost_min:
                    train_cost_min = self._train_stats["cost"][epoch - 1]
                    self._save_weights()
                    no_upgrade_counter = 0
                else:
                    no_upgrade_counter += 1
                    if no_upgrade_counter >= self._early_stop_thres:
                        logger.info(
                            f"early stop threshold {self._early_stop_thres} reached, stop training"
                        )
                        break

        self._stopping_epoch = epoch

    def predict(self, X: np.ndarray, weights_path: Optional[str] = None) -> np.ndarray:
        """Predict output for test data X.

        For classification tasks, it might be necessary to reformat
        returned predictions before evaluating precision with test data y.
        This means that either both test y and pred y are in inverse format
        or in their original format (whatever it is, numbers, chars etc.).

        Params
        ------
        X: NumPy array
            Numerical data matrix of shape n x p, n observations and p attributes.

        weights_path: str
            File path for pre-trained weights, otherwise currently available weights.

        Returns
        -------
        NumPy array with shape (n,)
            Predicted output, for classification tasks as indices of unique values
            of original y (inverse format, see e.g. NumPy's unique). For regression
            returned data is not in any specific format, but just the predicted values.
        """
        if len(X.shape) != 2:
            raise ValueError("Give array `X` as n x p, NumPy's .reshape(1, -1) might be helpful")

        if weights_path is not None:
            if not os.path.isfile(weights_path):
                raise ValueError(f"Cannot find file {weights_path}")

            self._load_weights(weights_path)
        else:
            if not isinstance(self._w["w1"], np.ndarray):
                raise ValueError(
                    "Current weights are empty, fit the model before running prediction"
                )

        y_pred = self._compute_predict(X)

        if self.method == "class":
            return np.argmax(y_pred, axis=0)

        return y_pred.reshape(-1)

    def plot_fit_results(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """Plot fit results, cost and accuracy.

        Cost and accuracy are plotted for training data, and
        also for validation if possible (validation was used).

        Params
        ------
        figsize: tuple, two int values
            Width, height in inches.
        """
        if not isinstance(self._train_stats["cost"], np.ndarray):
            raise ValueError("Nothing to plot yet, train the model first")

        train_cost, train_acc = self._train_stats["cost"], self._train_stats["acc"]
        # filter out zero values (no results for these epochs)
        train_filt_cost = train_cost[train_cost > 0]
        train_filt_acc = train_acc[train_acc > 0]

        fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=120)

        ax[0].plot(train_filt_cost, color="blue", label="train")
        ax[1].plot(train_filt_acc, color="blue", label="train")

        if isinstance(self._val_stats["cost"], np.ndarray) and len(self._val_stats["cost"]) > 0:
            val_cost, val_acc = self._val_stats["cost"], self._val_stats["acc"]
            val_filt_cost = val_cost[val_cost > 0]
            val_filt_acc = val_acc[val_acc > 0]

            ax[0].plot(val_filt_cost, color="orange", label="valid")
            ax[1].plot(val_filt_acc, color="orange", label="valid")

        ax[0].set_ylabel("cost")
        ax[1].set_ylabel("accuracy")

        for j in range(2):
            ax[j].set_xlabel("epoch")
            ax[j].legend(loc="best")
            ax[j].grid()

        fig.tight_layout()

        plt.show()
        plt.close(fig)
