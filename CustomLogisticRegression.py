import numpy as np

class CustomLogisticRegression:
    def __init__(self, lr=0.01, num_iter=1000):
        self.lr = lr
        self.num_iter = num_iter
        self.weights = None
        self.bias = None
        self.classes_ = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes_ = np.unique(y)

        # Multi-class via One-vs-Rest strategy
        if len(self.classes_) > 2:
            self.models = []
            for cls in self.classes_:
                binary_y = np.where(y == cls, 1, 0)
                model = CustomLogisticRegression(lr=self.lr, num_iter=self.num_iter)
                model._fit_binary(X, binary_y)
                self.models.append(model)
        else:
            self._fit_binary(X, y)

    def _fit_binary(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        X = np.array(X)
        if hasattr(self, 'models'):
            probs = np.column_stack([model.predict_proba(X)[:, 1] for model in self.models])
            return probs / np.sum(probs, axis=1, keepdims=True)
        else:
            linear_model = np.dot(X, self.weights) + self.bias
            prob_1 = self._sigmoid(linear_model)
            return np.column_stack([1 - prob_1, prob_1])

    def predict(self, X):
        X = np.array(X)
        if hasattr(self, 'models'):
            probs = self.predict_proba(X)
            return self.classes_[np.argmax(probs, axis=1)]
        else:
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            return np.array([1 if i > 0.5 else 0 for i in y_predicted])