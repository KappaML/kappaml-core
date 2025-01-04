from typing import List

from river.base import Regressor
from river.metrics import MAE
from river.model_selection import GreedyRegressor


class MetaRegressor(GreedyRegressor):
    """Meta-regressor.

    This is a wrapper around a list of regressor models.

    Parameters
    ----------
    models
    metric

    """

    def __init__(self, models: List[Regressor], metric=MAE()):
        super().__init__(models, metric)
