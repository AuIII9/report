"""Bernoulli naive Bayes implemented with NumPy, without model libraries."""
import numpy as np


def _check_X(X):
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 784:
        raise ValueError("X must have shape (n_samples, 784)")
    if not np.issubdtype(X.dtype, np.number) or np.iscomplexobj(X):
        raise ValueError("X must contain real numeric pixels")
    if not np.all(np.isfinite(X)) or np.any(X < 0) or np.any(X > 255):
        raise ValueError("Pixels must be finite and in [0, 255]")
    return X


class NaiveBayesClassifier:
    """Course protocol: raw pixels enter fit/predict; threshold is fixed at 128."""

    def __init__(self, alpha: float = 1.0):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.alpha = float(alpha)

    def fit(self, X, y):
        X = _check_X(X)
        y = np.asarray(y)
        if y.shape != (len(X),) or len(y) == 0:
            raise ValueError("y must have shape (n_samples,) and be nonempty")
        if not np.all(np.isin(y, np.arange(10))):
            raise ValueError("Labels must be integers in [0, 9]")
        y = y.astype(np.int64)
        binary = X > 128
        self.classes_ = np.arange(10)
        counts = np.bincount(y, minlength=10)
        self.class_log_prior_ = np.log((counts + self.alpha) / (len(y) + 10 * self.alpha))
        ones = np.stack([binary[y == c].sum(axis=0) for c in self.classes_])
        self.feature_prob_ = (ones + self.alpha) / (counts[:, None] + 2 * self.alpha)
        self.feature_log_prob_ = np.log(self.feature_prob_)
        self.feature_log_neg_prob_ = np.log1p(-self.feature_prob_)
        self._log_odds = self.feature_log_prob_ - self.feature_log_neg_prob_
        self._bias = self.class_log_prior_ + self.feature_log_neg_prob_.sum(axis=1)
        return self

    def _joint_log_likelihood(self, X):
        if not hasattr(self, "_bias"):
            raise RuntimeError("Call fit before prediction")
        X = _check_X(X)
        result = np.empty((len(X), 10))
        for start in range(0, len(X), 2048):
            binary = (X[start:start + 2048] > 128).astype(np.float64)
            result[start:start + len(binary)] = binary @ self._log_odds.T + self._bias
        return result

    def predict(self, X):
        scores = self._joint_log_likelihood(X)
        return self.classes_[scores.argmax(axis=1)]

    def predict_proba(self, X):
        scores = self._joint_log_likelihood(X)
        scores -= scores.max(axis=1, keepdims=True)
        p = np.exp(scores)
        return p / p.sum(axis=1, keepdims=True)
