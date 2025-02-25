"""
The :mod:`kappaml_core.meta` module contains meta-learning algorithms
"""

from .meta_forecaster import MetaForecaster
from .meta_regressor import MetaRegressor

__all__ = [
    "MetaForecaster",
    "MetaRegressor",
]
