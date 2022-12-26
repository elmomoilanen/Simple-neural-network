"""Quick guide to simple neural network package.

Package contains the following modules
- activations
- evolution
- metrics
- neural_network

`ANN` class from the `neural_network` module implements the main functionality
of this package and can be interacted via its primary public interfaces `fit`
and `predict`.
"""
__version__ = "1.0.0"

import logging.config
import os
import json

from pathlib import Path

with open(os.path.join(Path(__file__).parent.parent, "logging.json"), "r") as conf_file:
    logging.config.dictConfig(json.load(conf_file))
