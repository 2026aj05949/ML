import numpy as np
from CustomDecisionTreeClassifier import CustomDecisionTreeClassifier


class CustomRandomForestClassifier:

    def __init__(
        self,
        n_trees=10,
        max_depth=10,
        min_samples_split=2,
        n_features=None,
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        pass

    def _bootstrap_samples(self, X, y):
        pass

    def predict(self, X):
        pass