"""Implements an evolution algorithm to search an optimal hyperparameter combination.

The goal is to find a globally optimal combination but obviously it's not guaranteed
that such solution can be found with this algorithm or the initial setup of parameters.

Strategy is the follows:

1) Initialize a population of size N
2) Compute fitness score for each member of the population
3) Order members by fitness and select the top M (< N) performers and also few P (< M) of the worst
4) For random members of the population mutate some of the parameters
5) Reproduce N-(M+P) new members from the M+P survived members

Repeat the process K times from step 2.
"""
import os
import time
import random
import logging

from typing import Tuple, Dict, List

import numpy as np

from .neural_network import ANN
from .metrics import eval_cost

logger = logging.getLogger(__name__)


class Evolution:
    """Algorithm to search optimal set of hyperparameters.

    From the hyperparameters, optimizer and activation functions are restricted
    by the `ANN` class and thus cannot be passed here from the user. On the contrary,
    neurons, learning_rates and lambdas (L2 regularization) can be passed by
    giving them via kwargs. They must be given as tuples.

    Parameters
    ----------
    generations : int >= 1
        Number of generations of the evolution.

    population_size : int >= 2
        Size of the population, set of hyperparameters.

    Other parameters
    ----------------
    **kwargs : Iterable[Union[float, int]]
        Keyword arguments `neurons`, `learning_rates` and `lambdas` are accepted and used.
        Their values must be iterables with either float or int items. If any of the
        mentioned arguments is not given, then default values will be used instead.

    Examples
    --------
    Consider a regression type learning task with synthetic data. There will
    be two generations and for every generation three fitted neural networks
    (with population size three) that each have 10 epochs.

    >>> import numpy as np
    >>> rg = np.random.default_rng()
    >>> X = rg.normal(size=(500, 2))
    >>> y = 0.7 * np.sin(X[:, 0]) + 0.3 * np.cos(X[:, 1])
    >>> from simple_neural_network.evolution import Evolution
    >>> evo = Evolution(generations=2, population_size=3)
    >>> results = evo.fit(X, y.reshape(-1, 1), "regression", epochs=10)
    >>> len(results)
    3
    """

    top_percentage = 0.6
    poor_percentage = 0.2
    mutate_threshold = 0.75

    allowed_hyperparameters = (
        "hidden_nodes",
        "optimizer",
        "learning_rate",
        "lambda_",
        "activation1",
        "activation2",
    )

    def __init__(self, generations: int, population_size: int, **kwargs) -> None:
        self.generations = generations

        self._top_popu = None
        self._poor_popu = None
        self._other_popu = None

        self.pop_size = population_size

        self._fitness_scores = None
        self._lowest_col, self._highest_col, self._median_col = range(3)

        default_neurons = (
            list(range(5, 50, 5)) + list(range(50, 225, 25)) + list(range(250, 1250, 250))
        )

        self._activations = ANN.allowed_hidden_activations
        self._optimizers = ANN.allowed_optimizers
        self._neurons = tuple(kwargs.get("neurons", default_neurons))
        self._learning_rates = tuple(kwargs.get("learning_rates", (1e-3, 1e-2, 0.1, 0.5, 1.0)))
        self._lambdas = tuple(kwargs.get("lambdas", (0.0, 5.0, 25.0, 50.0)))

        self._param_set = {
            "hidden_nodes": self._neurons,
            "activation": self._activations,
            "optimizer": self._optimizers,
            "learning_rate": self._learning_rates,
            "lambda_": self._lambdas,
        }

        self._rng = np.random.default_rng()

    def __repr__(self):
        return f"{self.__class__.__name__}(generations={self.generations!r}, population_size={self.pop_size!r})"

    @property
    def generations(self):
        return self._generations

    @generations.setter
    def generations(self, generations):
        self._generations = max(int(generations), 1)

    @property
    def pop_size(self):
        return self._pop_size

    @pop_size.setter
    def pop_size(self, population_size):
        self._pop_size = max(int(population_size), 2)

        self._init_popu_partition()

    def _init_popu_partition(self):
        self._top_popu = int(self.top_percentage * self.pop_size)
        self._poor_popu = int(self.poor_percentage * self.pop_size)
        self._other_popu = self.pop_size - self._top_popu - self._poor_popu

        if self._other_popu < 0:
            pop_sum = self._top_popu + self._poor_popu
            raise ValueError(
                f"Size {pop_sum} of top and poor performers cannot be larger than total population size {self.pop_size}"
            )

    def _get_random_param(self, param, count=1):
        return tuple(random.choices(self._param_set[param], k=count))

    def _init_population(self):
        population = []

        for _ in range(self.pop_size):
            popu_elem = {
                "hidden_nodes": self._get_random_param("hidden_nodes", count=2),
                "optimizer": self._get_random_param("optimizer")[0],
                "learning_rate": self._get_random_param("learning_rate")[0],
                "lambda_": self._get_random_param("lambda_")[0],
            }
            popu_elem["activation1"], popu_elem["activation2"] = self._get_random_param(
                "activation", count=2
            )

            population.append(popu_elem)

        return population

    def _save_generation_fitness_score(self, generation, population):
        fitness_scores = np.array([member["fitness"] for member in population])

        self._fitness_scores[generation, self._lowest_col] = np.min(fitness_scores)
        self._fitness_scores[generation, self._highest_col] = np.max(fitness_scores)
        self._fitness_scores[generation, self._median_col] = np.median(fitness_scores)

    def _select_by_fitness(self, population):
        members_sorted = sorted(population, key=lambda x: x["fitness"])

        # cost function value represents fitness, the lowest cost is the best
        fittest_members = members_sorted[: self._top_popu]
        poor_members = random.choices(members_sorted[self._top_popu :], k=self._poor_popu)

        new_population = fittest_members + poor_members
        random.shuffle(new_population)

        return new_population

    def _mutate(self, population):
        param_keys = self._param_set.keys()

        for member in population:
            if random.random() > self.mutate_threshold:
                # some of the params for this member are mutated
                mutate_keys = tuple(key for key in param_keys if random.random() > 0.5)

                for key in mutate_keys:
                    if key == "hidden_nodes":
                        member[key] = self._get_random_param(key, count=2)
                    elif key == "activation":
                        member["activation1"], member["activation2"] = self._get_random_param(
                            "activation", count=2
                        )
                    else:
                        member[key] = self._get_random_param(key)[0]

    def _reproduce(self, population):
        offsprings = []
        param_keys = set(population[0].keys()).difference(("fitness",))

        for _ in range(self._other_popu):
            mother, father = random.choices(population, k=2)
            offspring = {}

            for param in param_keys:
                if random.random() > 0.5:
                    offspring[param] = mother[param]
                else:
                    offspring[param] = father[param]

            offsprings.append(offspring)

        for offspring in offsprings:
            population.append(offspring)

    @staticmethod
    def _remove_weights_file(file_path):
        if not os.path.isfile(file_path):
            logger.warning(f"Not a file or cannot find `{file_path}`")
            return

        os.unlink(file_path)
        time.sleep(0.5)

    def split_data_to_train_and_test(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Split data randomly to train and test parts.

        Test part will contain maximum of 10 % or 1 observation of the original data.

        Parameters
        ----------
        X : NumPy array
        y : NumPy array

        Returns
        -------
        Tuple : Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
            First of the tuples contains x_train and y_train arrays (90 % or n-1 of the obs).
            Second and last of the tuples contains x_test and y_test (10 % or 1 obs of the data).
        """
        test_prop = 0.1
        test_size = max(int(X.shape[0] * test_prop), 1)

        all_idx = np.arange(X.shape[0])
        test_idx = self._rng.choice(all_idx, size=test_size, replace=False)
        test_mask = np.isin(all_idx, test_idx)

        x_train, x_test = X[~test_mask], X[test_mask]
        y_train, y_test = y[~test_mask], y[test_mask]

        return (x_train, y_train), (x_test, y_test)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method_type: str,
        early_stop_threshold: int = 25,
        epochs: int = 50,
        use_validation: bool = True,
    ) -> List[Dict]:
        """Run evolution based hyperparameter optimization.

        Algorithm seeks the optimal parameter combination from the parameter space.
        At the end of last generation, a list of parameter combinations is returned
        such that the first is the most fittest compared to others.

        Parameters
        ----------
        X : NumPy array
            Numerical data matrix of shape n x p, n observations and p attributes.

        y : NumPy array
            Dependent variable with numerical or categorical values of shape n x 1.

        method_type : str
            Type of learning, either `class` for classification or `reg` for regression.

        early_stop_threshold : int
            Restricts the number of total training passes (epochs) through the network
            to this value T when the value of the cost function doesn't decrease in T
            contiguous total passes. Defaults to 25.

        epochs : int
            Count of total feedforward/backpropagation passes through the network.
            Default value is 50.

        use_validation : bool
            Whether to use separate validation data in neural network fitting.
            True as default.

        Returns
        -------
        list
            Of dicts containing hyperparameter combinations. Here the whole hyperparameter
            population is sorted in ascending order in terms of the fitness which is the
            cost function value. Thus the first element of this list contains the "best"
            hyperparameter combination.
        """
        if len(X.shape) != 2:
            raise ValueError("Give array `X` as n x p, NumPy's .reshape(1, -1) might be helpful")
        if not (len(y.shape) == 2 and y.shape[1] == 1):
            raise ValueError("Give array `y` as n x 1, NumPy's .reshape(-1, 1) might be helpful")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Dimension mismatch for arrays, must be X.shape[0] == y.shape[0]")

        early_stop_thres = int(early_stop_threshold)
        epochs = int(epochs)
        use_validation = bool(use_validation)

        weights_save_path = os.path.join(os.path.split(__file__)[0], "_evo_weights.h5")

        (x_train, y_train), (x_test, y_test) = self.split_data_to_train_and_test(X, y)

        population = self._init_population()

        self._fitness_scores = np.zeros((self.generations, 3))
        start_timestamp = time.perf_counter()

        for gener in range(1, self.generations + 1):
            logger.info("####################")
            logger.info(f"evolution generation: {gener}/{self.generations}")

            for iter, param_set in enumerate(population):
                logger.info(f"hyperparameter set {iter+1}/{len(population)}")
                # ANN cannot handle argument "fitness"
                param_set.pop("fitness", None)

                ann = ANN(
                    **param_set,
                    method=method_type,
                    early_stop_threshold=early_stop_thres,
                    verbose_level="mid",
                )
                ann.fit(
                    x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=32,
                    weights_save_path=weights_save_path,
                    use_validation=use_validation,
                )
                try:
                    y_pred = ann.predict(x_test, weights_path=weights_save_path)
                except FileNotFoundError as err:
                    logger.warning(err)
                    # assuming cost is used as fitness, +inf is the worst then
                    param_set["fitness"] = np.inf
                    continue

                param_set["fitness"] = eval_cost(
                    y_test.reshape(-1), y_pred.reshape(-1), method=method_type
                )
                logger.info(f"fitness value (cost) for test data: {param_set['fitness']:.2f}")

            self._remove_weights_file(weights_save_path)
            self._save_generation_fitness_score(gener - 1, population)

            if gener == self.generations:
                return sorted(population, key=lambda x: x["fitness"])

            new_population = self._select_by_fitness(population)
            self._mutate(new_population)
            self._reproduce(new_population)

            population = new_population

            elapsed_time = time.perf_counter() - start_timestamp
            logger.info(f"elapsed time: {elapsed_time//60:.0f} m. {elapsed_time%60:.1f} s.")

        # should never land here
        return []
