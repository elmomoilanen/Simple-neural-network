"""Quick guide to simple neural network package.

Package contains four modules in total.

1) activations
    Contains implementations for activation functions, e.g. tanh and relu.
    
2) evolution
    Contains implementation for Evolution class.

3) metrics
    Contains metrics to evaluate performance of neural networks.

4) neural_network
    Contain implementation for ANN class which has the core functionality
    of this package.
"""
__version__ = "1.0.0"

import logging.config
import os
import json

from pathlib import Path

with open(os.path.join(Path(__file__).parent.parent, "logging.json"), "r") as conf_file:
    logging.config.dictConfig(json.load(conf_file))
