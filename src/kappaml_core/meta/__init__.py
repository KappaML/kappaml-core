"""
The :mod:`kappaml_core.meta` module contains meta-learning algorithms
"""

from .meta_classifier import MetaClassifier
from .meta_regressor import MetaRegressor

__all__ = [
    "MetaRegressor",
    "MetaClassifier",
]
