from typing import List

from river.base import Classifier
from river.metrics import Accuracy
from river.model_selection.base import ModelSelectionClassifier
from river.tree import HoeffdingTreeClassifier

from kappaml_core.meta.base import MetaEstimator


class MetaClassifier(MetaEstimator, ModelSelectionClassifier):
    """Meta-classifier.

    This implements a meta-classifier that uses a list of base classifier models
    and a meta learner. The meta learner uses meta features from stream characteristics
    to select the best base classifier at a given point in time.

    Parameters
    ----------
    models: list of Classifier
        A list of base classifier models.
    meta_learner: Classifier
        default=HoeffdingTreeClassifier
        Meta learner used to predict the best base estimator.
    metric: Metric
        default=Accuracy
        Metric used to evaluate the performance of the base classifiers.
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
        models: List[Classifier],
        meta_learner: Classifier = HoeffdingTreeClassifier(),
        metric=Accuracy(),
        mfe_groups: list = ["general"],
        window_size: int = 200,
        meta_update_frequency: int = 50,
    ):
        super().__init__(
            models, meta_learner, metric, mfe_groups, window_size, meta_update_frequency
        )
