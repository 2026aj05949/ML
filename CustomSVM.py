import numpy as np


class CustomSVM:

    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.classes_ = None
        self.w = None
        self.b = None
        self.models = []

    def fit(self, X, y):
        pass

    def _fit_binary(self, X, y):
        pass

    def _decision_function(self, X):
        pass

    def predict(self, X):
        pass