"""A fixed, manually specified Bayesian network with twelve observed features."""
import numpy as np

FEATURE_NAMES = (
    "top_left", "top_middle", "top_right",
    "middle_left", "middle_middle", "middle_right",
    "bottom_left", "bottom_middle", "bottom_right",
    "aspect_ratio", "left_right_asymmetry", "up_down_asymmetry",
)
CARDINALITIES = (5,) * 12
# Every node also has Y as a parent. These additional edges are hand-chosen.
FEATURE_PARENTS = (None, 0, 1, None, 3, 4, None, 6, 7, None, None, None)
DENSITY_THRESHOLDS = (0.08, 0.20, 0.40, 0.60)
ASPECT_THRESHOLDS = (0.35, 0.55, 0.75, 0.95)
ASYMMETRY_THRESHOLDS = (0.04, 0.08, 0.12, 0.18)


def _check_X(X):
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 784:
        raise ValueError("X must have shape (n_samples, 784)")
    if not np.issubdtype(X.dtype, np.number) or np.iscomplexobj(X):
        raise ValueError("X must contain real numeric pixels")
    if not np.all(np.isfinite(X)) or np.any(X < 0) or np.any(X > 255):
        raise ValueError("Pixels must be finite and in [0, 255]")
    return X


def extract_features(X):
    """Return twelve integer features in [0, 4] from raw (n, 784) images.

    Thresholds are fixed before evaluation; no statistics are fitted to test data.
    The nine densities use a 3x3 partition of the central 24x24 pixel region.
    Asymmetry is the fraction of differing pixels after reflection of the full image.
    """
    X = _check_X(X)
    features = np.empty((len(X), 12), dtype=np.int64)
    for start in range(0, len(X), 2048):
        images = X[start:start + 2048].reshape(-1, 28, 28) > 128
        n = len(images)
        density = images[:, 2:26, 2:26].reshape(n, 3, 8, 3, 8).mean(axis=(2, 4)).reshape(n, 9)
        features[start:start + n, :9] = np.digitize(density, DENSITY_THRESHOLDS)
        rows = images.any(axis=2)
        cols = images.any(axis=1)
        height = 28 - rows.argmax(axis=1) - rows[:, ::-1].argmax(axis=1)
        width = 28 - cols.argmax(axis=1) - cols[:, ::-1].argmax(axis=1)
        nonempty = rows.any(axis=1)
        ratio = np.where(nonempty, width / np.maximum(height, 1), 0.0)
        features[start:start + n, 9] = np.digitize(ratio, ASPECT_THRESHOLDS)
        lr = np.mean(images != images[:, :, ::-1], axis=(1, 2))
        ud = np.mean(images != images[:, ::-1, :], axis=(1, 2))
        features[start:start + n, 10] = np.digitize(lr, ASYMMETRY_THRESHOLDS)
        features[start:start + n, 11] = np.digitize(ud, ASYMMETRY_THRESHOLDS)
    return features


class BayesNetClassifier:
    """Y is the root; each feature has Y and at most one feature parent."""

    def __init__(self, alpha: float = 1.0):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.alpha = float(alpha)

    def fit(self, X, y):
        features = extract_features(X)
        y = np.asarray(y)
        if y.shape != (len(features),) or len(y) == 0:
            raise ValueError("y must have shape (n_samples,) and be nonempty")
        if not np.all(np.isin(y, np.arange(10))):
            raise ValueError("Labels must be integers in [0, 9]")
        y = y.astype(np.int64)
        self.classes_ = np.arange(10)
        counts = np.bincount(y, minlength=10)
        self.class_log_prior_ = np.log((counts + self.alpha) / (len(y) + 10 * self.alpha))
        self.cpts_ = []
        self.log_cpts_ = []
        self.parent_counts_ = []
        for i, parent in enumerate(FEATURE_PARENTS):
            r = CARDINALITIES[i]
            q = 1 if parent is None else CARDINALITIES[parent]
            p = np.zeros(len(y), dtype=np.int64) if parent is None else features[:, parent]
            code = (y * q + p) * r + features[:, i]
            table = np.bincount(code, minlength=10 * q * r).reshape(10, q, r)
            support = table.sum(axis=2, keepdims=True)
            cpt = (table + self.alpha) / (support + r * self.alpha)
            self.cpts_.append(cpt)
            self.log_cpts_.append(np.log(cpt))
            self.parent_counts_.append(support[..., 0])
        return self

    def _joint_log_likelihood(self, X):
        if not hasattr(self, "log_cpts_"):
            raise RuntimeError("Call fit before prediction")
        features = extract_features(X)
        scores = np.broadcast_to(self.class_log_prior_, (len(features), 10)).copy()
        for i, parent in enumerate(FEATURE_PARENTS):
            p = np.zeros(len(features), dtype=np.int64) if parent is None else features[:, parent]
            scores += self.log_cpts_[i][self.classes_[None, :], p[:, None], features[:, i, None]]
        return scores

    def predict(self, X):
        scores = self._joint_log_likelihood(X)
        return self.classes_[scores.argmax(axis=1)]

    def predict_proba(self, X):
        scores = self._joint_log_likelihood(X)
        scores -= scores.max(axis=1, keepdims=True)
        p = np.exp(scores)
        return p / p.sum(axis=1, keepdims=True)
