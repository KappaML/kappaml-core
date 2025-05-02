from collections import deque
from copy import deepcopy
from typing import List

import numpy as np
from pymfe.mfe import MFE
from river.base import Classifier, Regressor
from river.metrics import MAE
from river.model_selection.base import ModelSelectionRegressor
from river.tree import HoeffdingTreeClassifier


class MetaRegressor(ModelSelectionRegressor):
    """Meta-regressor.

    This implements a meta-regressor that uses a list of base regressor models
    and a meta learner. The meta learner uses meta features from stream characteristics
    to select the best base regressor at a given point in time.

    Parameters
    ----------
    models: list of Regressor
        A list of base regressor models.
    meta_learner: Classifier
        default=HoeffdingTreeClassifier
        Meta learner used to predict the best base estimator.
    metric: Metric
        default=MAE
        Metric used to evaluate the performance of the base regressors.
    mfe_groups: list (default=['general'])
        Groups of meta-features to use from PyMFE
    window_size: int (default=200)
        The size of the window used for extracting meta-features.
    meta_update_frequency: int (default=50)
        How frequently to extract meta-features and update the meta-learner.
        Higher values mean less frequent updates but more stable meta-model.
    """

    def __init__(
        self,
        models: List[Regressor],
        meta_learner: Classifier = HoeffdingTreeClassifier(),
        metric=MAE(),
        mfe_groups: list = ["general"],
        window_size: int = 200,
        meta_update_frequency: int = 50,
    ):
        super().__init__(models, metric)  # type: ignore

        self.meta_learner = meta_learner

        self.mfe_groups = mfe_groups

        self.window_size = window_size
        self.meta_update_frequency = meta_update_frequency

        # Track performance of each model globally
        self.metrics = [deepcopy(metric) for _ in range(len(self))]

        self.mfe = MFE(groups=self.mfe_groups, suppress_warnings=True)

        # Window of (x, y) pairs for meta-feature extraction
        self.window_data_x = deque(maxlen=window_size)
        self.window_data_y = deque(maxlen=window_size)

        # Track performance of each model on the current window
        self.window_metrics = [deepcopy(metric) for _ in range(len(self))]

        # Store most recent meta-features
        self.meta_features = None

        # Counter to track samples for meta-update frequency
        self.sample_counter = 0

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
            self.mfe.fit(X, y, suppress_warnings=True)
            meta_features = self.mfe.extract(suppress_warnings=True)
            # Convert to dict for easier use with River
            features_dict = {
                name: value for name, value in zip(meta_features[0], meta_features[1])
            }
            # Remove nan values
            features_dict = {k: v for k, v in features_dict.items() if not np.isnan(v)}
            return features_dict
        except Exception as e:
            print(f"Error extracting meta-features: {e}")
            return None

    def _get_best_window_model_index(self):
        """Get the index of the best performing model on the current window."""
        best_metric = self.window_metrics[0]
        best_index = 0

        for i, metric in enumerate(self.window_metrics):
            if metric.is_better_than(best_metric):
                best_metric = metric
                best_index = i

        return best_index, best_metric.get()

    def _get_best_global_model_index(self):
        """Get the best global model."""
        best_metric = self.metrics[0]
        best_index = 0

        for i, metric in enumerate(self.metrics):
            if metric.is_better_than(best_metric):
                best_metric = metric
                best_index = i

        return best_index, best_metric.get()

    def learn_one(self, x, y):
        # Store data in window
        self.window_data_x.append(x)
        self.window_data_y.append(y)
        self.sample_counter += 1

        # Update all models and their metrics
        for i, (model, metric) in enumerate(zip(self, self.metrics)):
            y_pred = model.predict_one(x)
            metric.update(y, y_pred)
            model.learn_one(x, y)

            # Update window metrics
            self.window_metrics[i].update(y, y_pred)

        # Only extract meta-features and update meta-learner periodically
        if (
            len(self.window_data_x) >= self.window_size
            and self.sample_counter >= self.meta_update_frequency
        ):
            meta_features = self._extract_meta_features()

            if meta_features:
                # Get the best model index for this window
                best_model_idx, _ = self._get_best_window_model_index()

                # Train meta-learner to predict the best model index
                self.meta_learner.learn_one(meta_features, best_model_idx)

                # Store current meta-features for use in predict_one
                self.meta_features = meta_features

                # Reset window metrics for next window
                self.window_metrics = [deepcopy(self.metric) for _ in range(len(self))]

                # Reset sample counter
                self.sample_counter = 0

        return self

    def predict_one(self, x):
        # If we have meta-features and a trained meta-learner, use them
        if self.meta_features is not None:
            try:
                predicted_model_idx = int(
                    round(self.meta_learner.predict_one(self.meta_features))
                )
                # Ensure index is valid
                predicted_model_idx = max(
                    0, min(predicted_model_idx, len(self.models) - 1)
                )
                return self.models[predicted_model_idx].predict_one(x)
            except Exception as e:
                print(f"Error predicting with meta-learner: {e}")

        # Fall back to the best model so far if meta-learner prediction fails
        return self.best_model.predict_one(x)

    @property
    def best_model(self):
        return self.models[self._get_best_global_model_index()[0]]
