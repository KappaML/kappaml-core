from copy import deepcopy
from typing import List

from pymfe.mfe import MFE
from river.base import Regressor
from river.linear_model import LinearRegression
from river.metrics import MAE
from river.model_selection.base import ModelSelectionRegressor


class MetaRegressor(ModelSelectionRegressor):
    """Meta-regressor.

    This implements a meta-regressor that uses a list of base regressor models
    and a meta learner. The meta learner uses meta features from stream characteristics
    to select the best base regressor at a given point in time.

    Parameters
    ----------
    models: list of Regressor
        A list of base regressor models.
    meta_learner: Regressor
        default=LinearRegression
        Meta learner used to predict the best base estimator.
    metric: Metric
        default=MAE
        Metric used to evaluate the performance of the base regressors.
    mfe_groups: list (default=['general', 'statistical', 'info-theory'])
        Groups of meta-features to use from PyMFE
    window_size: int (default=100)
        The size of the window used for extracting meta-features.

    Notes
    -----
    The meta-regressor uses the PyMFE library to extract meta-features from the stream.
    """

    def __init__(
        self,
        models: List[Regressor],
        meta_learner: Regressor = LinearRegression(),
        metric=MAE(),
        mfe_groups: list = ["general", "statistical", "info-theory"],
        window_size: int = 100,
    ):
        super().__init__(models, metric)  # type: ignore
        self.mfe_groups = mfe_groups
        self.window_size = window_size
        self.meta_learner = meta_learner

        self.metrics = [deepcopy(metric) for _ in range(len(self))]

        self._best_model = models[0]
        self._best_metric = self.metrics[0]

        self.mfe = MFE(groups=self.mfe_groups)

    def learn_one(self, x, y):
        for model, metric in zip(self.models, self.metrics):
            y_pred = model.predict_one(x)
            metric.update(y, y_pred)
            model.learn_one(x, y)

            if metric.is_better_than(self._best_metric):
                self._best_model = model
                self._best_metric = metric

    @property
    def best_model(self):
        return self._best_model
