import unittest

import numpy as np

from explanation_evaluation import (
    delete_frames,
    frame_dropout,
    metric_drops,
    top_fraction_overlap,
)


class ExplanationEvaluationTest(unittest.TestCase):
    def test_top_fraction_overlap(self):
        reference = np.arange(10, dtype=float)
        perturbed = reference.copy()
        self.assertEqual(top_fraction_overlap(reference, perturbed, 0.2), 1.0)

    def test_deletion_and_dropout_shapes(self):
        features = np.arange(40).reshape(10, 4)
        scores = np.arange(10, dtype=float)
        retained, indices = delete_frames(features, scores, 0.2, "high")
        self.assertEqual(retained.shape, (8, 4))
        self.assertEqual(indices.tolist(), list(range(8)))
        dropped, kept = frame_dropout(
            features, 0.05, rng=np.random.default_rng(1)
        )
        self.assertEqual(dropped.shape, (9, 4))
        self.assertEqual(len(kept), 9)

    def test_metric_drops(self):
        drops = metric_drops(
            {"f1": 0.6, "spearman": 0.1},
            {"f1": 0.55, "spearman": 0.08},
        )
        self.assertAlmostEqual(drops["f1"], 0.05)
        self.assertAlmostEqual(drops["spearman"], 0.02)


if __name__ == "__main__":
    unittest.main()
