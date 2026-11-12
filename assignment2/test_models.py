"""Small independent checks for probabilities, counts, and the course protocol."""
import unittest
import numpy as np
from naive_bayes import NaiveBayesClassifier
from bayes_net import BayesNetClassifier, extract_features, FEATURE_PARENTS
from main import evaluate


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.X = np.zeros((4, 784), dtype=np.uint8)
        self.X[0, 0] = 255
        self.X[2:, :] = 255
        self.y = np.array([0, 0, 1, 1])

    def test_bernoulli_matches_direct_probability_product(self):
        model = NaiveBayesClassifier().fit(self.X, self.y)
        np.testing.assert_allclose(model.feature_prob_[0, :2], [.5, .25])
        np.testing.assert_allclose(model.feature_prob_[1, :2], [.75, .75])
        # Long double avoids underflow in this independent, non-log calculation.
        direct = []
        for row in self.X:
            values = []
            for c in range(10):
                prior = (np.count_nonzero(self.y == c) + 1) / 14
                probs = model.feature_prob_[c].astype(np.longdouble)
                values.append(prior * np.prod(np.where(row > 128, probs, 1 - probs)))
            values = np.asarray(values)
            direct.append((values / values.sum()).astype(float))
        np.testing.assert_allclose(model.predict_proba(self.X), direct, atol=1e-12)

    def test_bn_matches_direct_counting_and_joint(self):
        model = BayesNetClassifier().fit(self.X, self.y)
        F = extract_features(self.X)
        for i, parent in enumerate(FEATURE_PARENTS):
            for c in range(10):
                for p in range(model.cpts_[i].shape[1]):
                    mask = self.y == c
                    if parent is not None:
                        mask = mask & (F[:, parent] == p)
                    for value in range(5):
                        expected = (np.count_nonzero(mask & (F[:, i] == value)) + 1) / (mask.sum() + 5)
                        self.assertAlmostEqual(model.cpts_[i][c, p, value], expected)
            np.testing.assert_allclose(model.cpts_[i].sum(axis=2), 1)
        joint = np.empty((4, 10))
        for n in range(4):
            for c in range(10):
                prob = (np.count_nonzero(self.y == c) + 1) / 14
                for i, parent in enumerate(FEATURE_PARENTS):
                    p = 0 if parent is None else F[n, parent]
                    prob *= model.cpts_[i][c, p, F[n, i]]
                joint[n, c] = prob
        np.testing.assert_allclose(model.predict_proba(self.X), joint / joint.sum(axis=1, keepdims=True))

    def test_features_and_structure(self):
        F = extract_features(self.X)
        self.assertEqual(F.shape, (4, 12))
        self.assertTrue(((F >= 0) & (F <= 4)).all())
        np.testing.assert_array_equal(F[1], np.zeros(12))
        np.testing.assert_array_equal(F[2], [4] * 10 + [0, 0])
        for child, parent in enumerate(FEATURE_PARENTS):
            self.assertTrue(parent is None or parent < child)  # topological order
        self.assertGreaterEqual(sum(p is not None for p in FEATURE_PARENTS), 2)
        self.assertEqual(extract_features(np.zeros((0, 784))).shape, (0, 12))

    def test_protocol_empty_batches_and_input_validation(self):
        for cls in (NaiveBayesClassifier, BayesNetClassifier):
            with self.assertRaises(RuntimeError):
                cls().predict(self.X)
            model = cls()
            self.assertIs(model.fit(self.X, self.y), model)
            self.assertEqual(model.predict(self.X).shape, (4,))
            self.assertEqual(model.predict(np.zeros((0, 784))).shape, (0,))
            self.assertEqual(model.predict_proba(np.zeros((0, 784))).shape, (0, 10))
            with self.assertRaises(ValueError):
                model.predict(np.zeros((2, 28, 28)))
            with self.assertRaises(ValueError):
                model.fit(self.X, [0, 1, 2, 10])
            with self.assertRaises(ValueError):
                model.predict(np.full((1, 784), np.nan))
            with self.assertRaises(ValueError):
                cls(alpha=0)

    def test_macro_metrics_and_matrix_orientation(self):
        scores = evaluate(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
        self.assertEqual(scores["confusion_matrix"][0][:2], [1, 1])
        self.assertAlmostEqual(scores["accuracy"], .75)
        self.assertAlmostEqual(scores["macro_precision"], (1 + 2 / 3) / 10)
        self.assertAlmostEqual(scores["macro_recall"], (.5 + 1) / 10)
        self.assertAlmostEqual(scores["macro_f1"], (2 / 3 + .8) / 10)


if __name__ == "__main__":
    unittest.main()
