"""Guide to simple_neural_network package.

Package contains following modules:
- neural_network
- evolution

`ANN` class from the neural_network module implements
the main functionality of this package.
"""
__version__ = "1.0.0"

import logging.config
import os
import json

from pathlib import Path

with open(os.path.join(Path(__file__).parent.parent, "logging.json"), "r") as conf_file:
    logging.config.dictConfig(json.load(conf_file))
