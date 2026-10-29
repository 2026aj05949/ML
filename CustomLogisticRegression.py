import numpy as np


class CustomLogisticRegression:

    def __init__(self, lr=0.01, num_iter=1000):
        self.lr = lr
        self.num_iter = num_iter
        self.weights = None
        self.bias = None
        self.classes_ = None
        self.models = []

    def _sigmoid(self, z):
        pass

    def fit(self, X, y):
        pass

    def _fit_binary(self, X, y):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass