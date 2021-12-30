# Simple neural network #

[![main](https://github.com/elmomoilanen/Simple-neural-network/actions/workflows/main.yml/badge.svg)](https://github.com/elmomoilanen/Simple-neural-network/actions/workflows/main.yml)

Library that implements a simple two hidden layer artificial neural network with NumPy. Purpose of this library is to provide an easy learning and testing environment for essentials of supervised learning and neural network models in particular. 

## Install ## 

File *pyproject.toml* lists the dependencies. The recommended way to do the installation is via Poetry (a package and dependency manager for Python) but of course other options are possible as long as the libraries listed in *tool.poetry.dependencies* are installed.

With Poetry, after cloning and navigating to the target folder, run the following command in a shell

```bash
poetry install --no-dev
```

which creates a virtual environment for the library and installs required non-development dependencies (NumPy etc.) inside it. Virtual environment setup is controlled by the *poetry.toml* file. As the *--no-dev* option skips installation of the dev dependencies, don't include it in command above if e.g. you want to be able to run the unit tests (pytest is needed for that).

## Use ##

Module *neural_network* contains the *ANN* class which implements the two hidden layer neural network model. Its two main public methods are *fit* and *predict*, other methods *get_fit_results* and *plot_fit_results* can be used to check results of the fitting step. Module *evolution* contains the *Evolution* class which implements an evolution based algorithm to find the optimal hyperparameter combination. This is also used via public method *fit*.

The following example illustrates the usage of this library.

Consider a typical supervised learning task where the aim is to learn a function **f** between provided example input-output (X-y) pairs such that the learned function would also generalize well for unseen data. Assume that X is a numerical data matrix of shape n x p (n observations, p attributes) and y is an array of labels of size n. As the dependent variable y contains labels, the function **f** classifies each *x_i* from the input space to the output space.

Following code imports the ANN class from neural_network module and fits the model (or learns the function) for the example X-y pairs. We notice that in order to run the model fitting (ann.fit), a certain set of hyperparameters must be set in advance (when instantiating the *ann* object). This can be done manually or automated by an additional hyperparameter optimization step. More on this latter option later.

```python
from simple_neural_network.neural_network import ANN

ann = ANN(hidden_nodes=(40, 15), learning_rate=1.0, activation1="tanh", early_stop_threshold=50)
ann.fit(X, y.reshape(-1, 1), epochs=500, batch_size=50)

# fit done, get a summary of the fitting
ann.get_fit_results()

# plot cost and accuracy data of the fitting
ann.plot_fit_results()
```

In case of neural networks, learning is said to happen when the weights between neurons update during fitting. Quality or strength of this learning doesn't necessarily increase all along from the beginning to the end and thus it might be a good strategy to halt the fitting if the results doesn't get better in some T contiguous number of steps (more precisely, total passes through the network aka epochs). In this case of an early stop and actually in all other cases too, the optimal weights are saved to a file, by default to the current working directory with name *weights.h5*. This way the best model is kept available for later use irrespective of how the fitting procedure goes to the end.

After the model has been fitted we might encounter new data X_new for which we would like to get the predicted labels y_new. This can simply be done by calling the predict method of the ANN class, and either assuming that a previously created and fitted object is still in memory or in other case passing a file path for the pre-trained neural network weights. For now, we continue the previous code snippet and assume that the weights of the best model found during fitting are still available.

If for some reason we got also y_new (true labels for X_new), we can evaluate the performance of the prediction by using the *confusion_matrix* function from the metrics module.

```python
y_new_pred = ann.predict(X_new)

# assume we got also y_new, so import the confusion matrix to evaluate prediction performance
from simple_neural_network.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_true=y_new, y_pred=y_new_pred)
```

Rows of the confusion matrix represent the true labels with zero-based indexing and columns corresponding predicted labels. For example, entry (0, 1) of the table represents a case where the true label is zero and predicted one. Thus, diagonal entries indicate the correctly predicted counts.
