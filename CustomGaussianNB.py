import numpy as np


class CustomGaussianNB:

    def __init__(self):
        self._classes = None
        self._mean = None
        self._var = None
        self._priors = None

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def _predict_single(self, x):
        pass

    def _pdf(self, class_idx, x):
        pass