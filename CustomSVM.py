import numpy as np

class CustomSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.classes_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes_ = np.unique(y)

        # Multi-class via One-vs-Rest strategy
        if len(self.classes_) > 2:
            self.models = []
            for cls in self.classes_:
                binary_y = np.where(y == cls, 1, -1)
                model = CustomSVM(lr=self.lr, lambda_param=self.lambda_param, n_iters=self.n_iters)
                model._fit_binary(X, binary_y)
                self.models.append(model)
        else:
            binary_y = np.where(y == self.classes_[1], 1, -1)
            self._fit_binary(X, binary_y)

    def _fit_binary(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]

    def _decision_function(self, X):
        return np.dot(X, self.w) - self.b

    def predict(self, X):
        X = np.array(X)
        if hasattr(self, 'models'):
            scores = np.column_stack([model._decision_function(X) for model in self.models])
            return self.classes_[np.argmax(scores, axis=1)]
        else:
            approx = self._decision_function(X)
            return np.where(approx >= 0, self.classes_[1], self.classes_[0])
