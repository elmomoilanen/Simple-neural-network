# Simple neural network #

[![main](https://github.com/elmomoilanen/Simple-neural-network/actions/workflows/main.yml/badge.svg)](https://github.com/elmomoilanen/Simple-neural-network/actions/workflows/main.yml)

Library that implements a simple two hidden layer artificial neural network with NumPy. Purpose of this library is to provide an easy learning and testing environment for essentials of supervised learning and neural network models in particular. 

## Install ## 

File *pyproject.toml* lists the dependencies. Installation is recommended to do via Poetry (a package and dependency manager for Python) but of course other options are possible as long as the libraries listed in *tool.poetry.dependencies* are installed.

With Poetry, after cloning and navigating to the target folder, run the following command in a shell

```bash
poetry install --no-dev
```

which creates a virtual environment for the library and installs required non-development dependencies (NumPy etc.) inside it. Virtual environment setup is controlled by the *poetry.toml* file. As the *--no-dev* option skips installation of the dev dependencies, don't include it in the command above if e.g. want to be able to run the unit tests (pytest is needed for that).

## Use ##

Module *neural_network* contains class *ANN* which implements the two hidden layer neural network model. Its two main public methods are *fit* and *predict*, other two methods *get_fit_results* and *plot_fit_results* can be used to inspect results of the fitting step after it has been run. Selecting appropriate hyperparameters is an important part of the neural network design and for this respect module *evolution* contains class *Evolution* with a public method *fit* implementing an evolution based algorithm to search an optimal hyperparameter combination.

The following example illustrates the usage of this library.

Consider a typical supervised learning task where the aim is to learn a function **f** between provided example input-output (X-y) pairs such that the learned function would also generalize well for unseen data. Assume that X is a numerical data matrix of shape n x p (n observations, p attributes) and y is an array of labels of size n. As the dependent variable y contains labels, the function **f** classifies each *x_i* from the input space to the output space.

Following code imports the ANN class from the neural_network module and fits a model (i.e., learns a function **f**) for the example X-y pairs. We notice that in order to run the model fitting (ann.fit), a certain set of hyperparameters must be set in advance. This can be done manually or automated by an additional hyperparameter optimization step. More on this latter option later.

```python
from simple_neural_network.neural_network import ANN

ann = ANN(hidden_nodes=(40, 15), learning_rate=1.0, activation1="tanh", early_stop_threshold=50)
ann.fit(X, y.reshape(-1, 1), epochs=500, batch_size=50)

# fit done, get a summary of the process
ann.get_fit_results()

# plot cost and accuracy data of the fitting
ann.plot_fit_results()
```

Speaking of neural networks, learning is said to happen when the weights between neurons adjust during fitting process. Quality or strength of this learning doesn't necessarily increase all along from the beginning to the end and thus it might be a good strategy to halt the fitting process if results don't get better in some T contiguous number of epochs (total passes through the network). In the case of an early stop and actually in all other cases too, the optimal weights are saved to a file, by default to the current working directory with name *weights.h5*. This way the best model is kept available for later use irrespective of how the fitting process goes to the end.

After the model has been fitted we might encounter new input data X_new for which we would like to get the predicted labels y_new. This can simply be done by calling the predict method of the ANN class, and either assuming that a previously created and fitted object is still in memory or in other case passing a file path for the pre-trained neural network weights. For now, we continue the previous code snippet and assume that the weights of the best model found during fitting are still available. Notice however that this assumption doesn't usually hold in practise and in this case one needs to pass the path to the weights file.

If for some reason we got also y_new (true labels for X_new), we can evaluate the performance of the prediction by using the *confusion_matrix* function from the *metrics* module.

```python
y_new_pred = ann.predict(X_new)

# import the confusion matrix to evaluate prediction performance
from simple_neural_network.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_true=y_new, y_pred=y_new_pred)
```

Rows of the confusion matrix represent the true labels with zero-based indexing and columns corresponding predicted labels. For example, entry (0, 1) of the table represents a case where the true label is zero and predicted one. Thus, diagonal entries indicate the correctly predicted counts.

At the end of this example let's get back to the hyperparameter aspect. Finding optimal or even good hyperparameters is a difficult task, one and maybe the most common possibility being just trying different combinations manually. As mentioned above, other option is to use an evolutionary algorithm and for the case of this library, the algorithm found in *evolution* module. This evolution algorithm can be seen as a bit enhanced version of the basic cross-validation approach that would likely take much more time to complete.

Evolution algorithm can be used as follows (assume the same input-output data X-y that were used above)

```python
from simple_neural_network.evolution import Evolution

evo = Evolution(generations=10, population_size=20)

evo.fit(X, y.reshape(-1, 1), "classification")
```

where the result of *fit* method call will be a list of parameter combinations of size 20 where the first combination is the most fittest (had lowest cost function value). This combination can be passed for a new ANN object or further narrow down search region of the hyperparameter space.
