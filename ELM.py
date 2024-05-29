import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error

class ELM(BaseEstimator, RegressorMixin):
    def __init__(self, n_hidden_units=1000, activation='sigmoid', random_state=None):
        self.n_hidden_units = n_hidden_units
        self.activation = activation
        self.random_state = random_state

    def _initialize_weights(self, n_features):
        rng = np.random.default_rng(self.random_state)
        self.W = rng.uniform(-1, 1, (n_features, self.n_hidden_units))
        self.b = rng.uniform(-1, 1, self.n_hidden_units)

    def _hidden_layer(self, X):
        G = np.dot(X, self.W) + self.b
        if self.activation == 'sigmoid':
            return expit(G)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

    def fit(self, X, y, C=10):
        self._initialize_weights(X.shape[1])
        H = self._hidden_layer(X)
        eye = np.eye(self.n_hidden_units)
        self.beta = np.linalg.inv(H.T @ H + 1 / C * eye) @ H.T @ y

    def predict(self, X):
        H = self._hidden_layer(X)
        T_pred = np.dot(H, self.beta)
        return T_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred)
