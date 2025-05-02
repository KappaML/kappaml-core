from collections import deque
from copy import deepcopy
from typing import List

import numpy as np
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

        # Window of (x, y) pairs for meta-feature extraction
        self.window_data_x = deque(maxlen=window_size)
        self.window_data_y = deque(maxlen=window_size)

        # Track performance of each model on the current window
        self.window_metrics = [deepcopy(metric) for _ in range(len(self))]

        # Store meta-features for each window
        self.meta_features = None

        # To track the index of the best model for each window
        self.best_model_indices = []

    def _extract_meta_features(self):
        """Extract meta-features from the current window."""
        if len(self.window_data_x) < self.window_size:
            return None

        # Convert deque to appropriate format for PyMFE
        X = np.array(
            [
                list(x.values()) if isinstance(x, dict) else list(x)
                for x in self.window_data_x
            ]
        )
        y = np.array(self.window_data_y)

        try:
            self.mfe.fit(X, y)
            meta_features = self.mfe.extract()
            # Convert to dict for easier use with River
            features_dict = {
                name: value for name, value in zip(meta_features[0], meta_features[1])
            }
            return features_dict
        except Exception as e:
            print(f"Error extracting meta-features: {e}")
            return None

    def _get_best_model_index(self):
        """Get the index of the best performing model on the current window."""
        best_score = float("-inf")
        best_index = 0

        for i, metric in enumerate(self.window_metrics):
            if metric.get() > best_score:
                best_score = metric.get()
                best_index = i

        return best_index

    def learn_one(self, x, y):
        # Store data in window
        self.window_data_x.append(x)
        self.window_data_y.append(y)

        # Update all models and their metrics
        for i, (model, metric) in enumerate(zip(self.models, self.metrics)):
            y_pred = model.predict_one(x)
            metric.update(y, y_pred)
            model.learn_one(x, y)

            # Update window metrics
            self.window_metrics[i].update(y, y_pred)

            # Update global best model if needed
            if metric.is_better_than(self._best_metric):
                self._best_model = model
                self._best_metric = metric

        # When the window is full, extract meta-features and train meta-learner
        if len(self.window_data_x) == self.window_size:
            meta_features = self._extract_meta_features()

            if meta_features:
                # Get the best model index for this window
                best_model_idx = self._get_best_model_index()
                self.best_model_indices.append(best_model_idx)

                # Train meta-learner to predict the best model index
                self.meta_learner.learn_one(meta_features, best_model_idx)

                # Store current meta-features for use in predict_one
                self.meta_features = meta_features

                # Reset window metrics for next window
                self.window_metrics = [deepcopy(self.metric) for _ in range(len(self))]

    def predict_one(self, x):
        # First add the data point to the window (without the label yet)
        # This ensures we're always using recent data for meta-feature extraction
        self.window_data_x.append(x)

        # If window has enough data, extract meta-features for prediction
        if len(self.window_data_x) == self.window_size:
            current_meta_features = self._extract_meta_features()

            # If we have extracted meta-features and a trained meta-learner,
            # use it to predict best model
            if current_meta_features is not None:
                try:
                    predicted_model_idx = int(
                        round(self.meta_learner.predict_one(current_meta_features))
                    )
                    # Ensure index is valid
                    predicted_model_idx = max(
                        0, min(predicted_model_idx, len(self.models) - 1)
                    )

                    # Remove the last x since we haven't called learn_one yet
                    self.window_data_x.pop()

                    return self.models[predicted_model_idx].predict_one(x)
                except Exception as e:
                    print(f"Error predicting with meta-learner: {e}")
                    # Remove the last x
                    self.window_data_x.pop()
            else:
                # Remove the last x
                self.window_data_x.pop()
        else:
            # Remove the last x if we didn't extract meta-features
            self.window_data_x.pop()

        # Fall back to the best model so far if meta-learner prediction fails
        if self._best_model is not None:
            return self._best_model.predict_one(x)

        # Default to first model
        return self.models[0].predict_one(x)

    @property
    def best_model(self):
        # Extract current meta-features if window is full
        if len(self.window_data_x) == self.window_size:
            current_meta_features = self._extract_meta_features()

            # If meta-learner is trained and we have current meta-features,
            # use them for prediction
            if current_meta_features is not None:
                try:
                    predicted_model_idx = int(
                        round(self.meta_learner.predict_one(current_meta_features))
                    )
                    predicted_model_idx = max(
                        0, min(predicted_model_idx, len(self.models) - 1)
                    )
                    return self.models[predicted_model_idx]
                except Exception:
                    pass

        return self._best_model
