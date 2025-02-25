from typing import List, Optional

from river import metrics
from river.base import Forecaster
from river.metrics import MAE


class MetaForecaster(Forecaster):
    """Meta-forecaster that combines multiple forecasting models.

    This forecaster implements an ensemble approach that dynamically selects
    and combines predictions from multiple base forecasters. It extends River's
    GreedyRegressor with specific forecasting capabilities.

    Parameters
    ----------
    models : List[Forecaster]
        List of forecasting models to ensemble
    metric : metrics.Metric, optional (default=MAE())
        Metric used to evaluate and select models
    window_size : int, optional (default=10)
        Size of the rolling window used for model evaluation
    combination_method : str, optional (default='dynamic')
        Method to combine predictions:
        - 'dynamic': Dynamically select best performing model
        - 'average': Simple average of all model predictions
        - 'weighted': Weighted average based on recent performance
    """

    def __init__(
        self,
        models: List[Forecaster],
        metric: Optional[metrics.Metric] = None,
        window_size: int = 10,
        combination_method: str = "dynamic",
    ):
        # Initialize with default MAE metric if none provided
        if metric is None:
            metric = MAE()

        super().__init__(models=models, metric=metric)

        self.window_size = window_size
        self.combination_method = combination_method
        self.model_weights = {i: 1.0 for i in range(len(models))}
        self.prediction_history = []

    def learn_one(self, x, y):
        """Update all models with the new observation.

        Parameters
        ----------
        x : dict
            Features
        y : float
            Target value
        """
        # Update base models
        for model in self.models:
            model.learn_one(x, y)

        # Update model weights based on recent performance
        if len(self.prediction_history) >= self.window_size:
            self._update_weights()

        return self

    def predict_one(self, x):
        """Make a prediction using the ensemble.

        Parameters
        ----------
        x : dict
            Features

        Returns
        -------
        float
            Combined prediction from the ensemble
        """
        predictions = [model.predict_one(x) for model in self.models]

        if self.combination_method == "dynamic":
            # Use the best performing model
            best_model_idx = max(self.model_weights.items(), key=lambda x: x[1])[0]
            return predictions[best_model_idx]

        elif self.combination_method == "weighted":
            # Weighted average based on model performance
            total_weight = sum(self.model_weights.values())
            weighted_sum = sum(
                pred * (self.model_weights[i] / total_weight)
                for i, pred in enumerate(predictions)
            )
            return weighted_sum

        else:  # 'average'
            # Simple average of all predictions
            return sum(predictions) / len(predictions)

    def _update_weights(self):
        """Update model weights based on recent performance."""
        recent_errors = []

        # Calculate recent errors for each model
        for model_idx, model in enumerate(self.models):
            model_errors = []
            for hist in self.prediction_history[-self.window_size :]:
                x, y = hist["x"], hist["y"]
                pred = model.predict_one(x)
                error = self.metric.get()(y, pred)
                model_errors.append(error)

            avg_error = sum(model_errors) / len(model_errors)
            recent_errors.append(avg_error)

        # Update weights inversely proportional to errors
        total_error = sum(recent_errors)
        if total_error > 0:
            for i, error in enumerate(recent_errors):
                self.model_weights[i] = 1.0 - (error / total_error)
