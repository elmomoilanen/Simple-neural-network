# Simple neural network #

[![main](https://github.com/elmomoilanen/Simple-neural-network/actions/workflows/main.yml/badge.svg)](https://github.com/elmomoilanen/Simple-neural-network/actions/workflows/main.yml)

This library implements a simple two hidden layer artificial neural network with NumPy.

## Install ## 

Poetry is the recommended tool for installation.

Run the following command to install default dependencies within an in-project virtual environment

```bash
poetry install
```

The setup for the in-project virtual environment is controlled by *poetry.toml*. In order to run e.g. unit tests, optional dependency group `dev` must be added to installation by appending `--with dev` to the command above.

For the plotting to work correctly, it might be required to set the backend for Matplotlib. One way to do this is to set the MPLBACKEND environment variable (overrides any matplotlibrc configuration) for the current shell.

## Use ##

Module *neural_network* contains a class *ANN* which implements the two hidden layer neural network model. Its primary public methods are *fit* and *predict*, the other two methods *get_fit_results* and *plot_fit_results* can be used to inspect results of the fitting step after it has been run. Selecting appropriate hyperparameters is an important part of the neural network design. To assist with this, the *evolution* module contains a class *Evolution*. This class has a public method *fit* that implements an evolution-based algorithm to search for an optimal hyperparameter combination.

The following example illustrates the usage of this library. To see the API docs, you can render the [documentation as HTML](#docs) or read the docstrings directly.

Start a new Python shell within the virtual environment e.g. as follows

```bash
MPLBACKEND= poetry run python
```

with a proper backend (e.g. macosx or qt5agg) after the equal sign. If the backend has been set correctly earlier, just drop this setting.

Consider now a typical supervised learning task where the goal is to learn a function between provided example input-output (X-y) pairs such that the learned function would also generalize well for unseen data. Assume that X is a numerical data matrix of shape n x p (n observations, p attributes) and y is an array of labels of size n. As the dependent variable y contains labels, the function classifies each x_i from the input space to the output space. Notice that this example and the synthetic data generation (skipped here) are also presented in the ANN's docstring.

Following code imports the ANN class from the *simple_neural_network* package and fits a model (i.e., learns a function) for the example X-y pairs.

```python
from simple_neural_network import ANN

ann = ANN(
    hidden_nodes=(40, 15),
    learning_rate=1.0,
    activation1="tanh",
    early_stop_threshold=50,
)

# By default, validation data constitutes 20 % of the provided dataset
# Batch size b means ceil(X_train.shape[0]/b) iterations per epoch
ann.fit(X, y.reshape(-1, 1), epochs=500, batch_size=50)

# Fit done, get a summary of the fitting results
ann.get_fit_results()

# Plot cost and accuracy
ann.plot_fit_results()
```

Notice that in order to run the model fitting using the ANN's *fit* method, a certain set of hyperparameters must be defined in advance. This can be done manually or automated by an additional hyperparameter optimization step, but more on this latter option later. For the complete set of hyperparameter options, please check the ANN's docstring as there are a myriad of choices.

Speaking of neural networks in general, learning is said to happen when the weights between neurons adjust during fitting process. The quality or strength of this learning doesn't necessarily increase continuously from the beginning to the end. Thus, it might be a good strategy to halt the fitting process if results don't improve over some T contiguous number of epochs (total passes through the network). To make this work during the learning process, the optimal model weights are saved to a file, by default to current working directory with name *weights.h5*. This way the best model is kept available for later use irrespective of how the fitting process goes up to the end.

After the model has been fitted we might encounter new input data X_new for which we would like to get the predicted labels y_new. This can simply be done by calling the ANN's *predict* method, and either assuming that a previously created and fitted object is still in memory or in other case passing a file path for the pre-trained neural network weights. For now, we continue from the previous code snippet and reuse the best model weights saved in the file *weights.h5*.

If for some reason we got also y_new (true labels for X_new), we can evaluate performance of the prediction by using *confusion_matrix* function from the *metrics* module.

```python
y_new_pred = ann.predict(X_new, "weights.h5")

# Import confusion_matrix directly from the package
from simple_neural_network import confusion_matrix

conf_matrix = confusion_matrix(y_true=y_new, y_pred=y_new_pred)
```

Rows of the confusion matrix represent the true labels with zero-based indexing and columns corresponding predicted labels. For example, in case of binary labels the confusion matrix is of size 2 x 2, and entry (0, 1) would represent the total count of cases where the true label is zero but the predicted label one. Similarly, entry (1, 0) represent the total count of cases where the prediction is zero but the true label is one. Then, of course, diagonal entries show the correctly predicted label counts.

Let's now get back to the hyperparameter problem. Finding optimal or even good hyperparameters is a difficult task. One option is to use an evolutionary algorithm that can be seen as a bit enhanced version of the basic cross-validation approach that would likely take much more time to complete.

The evolution algorithm used here needs for the start two parameters N and K that describe the population size (sets of hyperparameters) and number of generations (how long the algorithm will be run). At the beginning of the first generation, a population of size N is initialized. After that a fitness score (cost) is computed for every member of the population, members are ordered by the score and top M (< N) and few P (< M) of the worst performers are selected to survive, for random members of the survivors some of their parameters are mutated and finally N-(M+P) new members are reproduced from the survived members making the population size again N. This process, starting from the fitness score computations, is repeated K times.

This evolution algorithm can be used as follows

```python
from simple_neural_network import Evolution

# Search the optimal set of hyperparameters with N=20 and K=10
evo = Evolution(generations=10, population_size=20)

evo.fit(X, y.reshape(-1, 1), "classification")
```

where the *fit* method call returns a list of 20 parameter combinations where the first combination is the most fittest (has the lowest cost function value). This combination can be passed for a new ANN object or used to further narrow down search region of the hyperparameter space.

## Docs ##

Render the documentation as HTML files with the following command

```bash
poetry run sphinx-build -b html docs/source/ docs/build/html
```

and open the starting page `docs/build/html/index.html` in browser.

## Examples ##

Folder *examples* contain a classification task example for the MNIST data (the task is to classify handwritten digits). In order to run the code in a notebook, the data must be downloaded separately, and the Jupyter package must be installed and included in the current virtual environment.

Following command installs the Jupyter package (`examples` is now the required dependency group in pyproject.toml)

```bash
poetry install --with examples
```

and then fire up a notebook by running

```bash
poetry run jupyter notebook
```
