"""Small CPU tests for the manuscript-defined ISCRL contracts."""

import math
import unittest

import torch

from iscrl_components import AIMController, SimCLRObjective, SimCLRProjector
from main import build_parser, scheduled_learning_rate
from models import ISCRLPolicy
from rewards import compute_reward, single_space_reward


class ManuscriptDefaultsTest(unittest.TestCase):
    def test_training_and_aim_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["--split", "split.json", "--metric", "summe"])
        self.assertEqual(args.warmup_epochs, 50)
        self.assertEqual(args.rl_epochs, 300)
        self.assertEqual(args.hidden_dim, 512)
        self.assertEqual(args.reward_beta, 0.10)
        self.assertEqual(args.warmup_lr, 1e-5)
        self.assertEqual(args.temporal_threshold, 20)
        self.assertEqual(args.intervention_min, 0.10)
        self.assertEqual(args.intervention_max, 0.50)
        self.assertEqual(args.lr_gamma, 0.5)
        self.assertAlmostEqual(scheduled_learning_rate(args, 29), 1e-5)
        self.assertAlmostEqual(scheduled_learning_rate(args, 30), 5e-6)


class SimCLRTest(unittest.TestCase):
    def test_projection_and_infonce(self):
        torch.manual_seed(3)
        projector = SimCLRProjector(1024, 512, 128)
        objective = SimCLRObjective(projector)
        features = torch.randn(1, 6, 1024)
        loss = objective(features)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in projector.parameters()))


class PolicyTest(unittest.TestCase):
    def test_dual_branch_shapes(self):
        torch.manual_seed(4)
        model = ISCRLPolicy(
            input_dim=16,
            state_dim=8,
            invariant_dim=4,
            projector_hidden_dim=8,
            attention_dropout=0.0,
        )
        features = torch.randn(2, 5, 16)
        probabilities, state, attention, invariant, branches = model(features)
        self.assertEqual(probabilities.shape, (2, 5, 1))
        self.assertEqual(state.shape, (2, 5, 8))
        self.assertEqual(attention.shape, (2, 5, 5))
        self.assertEqual(invariant.shape, (2, 5, 4))
        self.assertEqual(set(branches), {"original", "invariant"})
        self.assertTrue(torch.all((probabilities >= 0) & (probabilities <= 1)))


class RewardTest(unittest.TestCase):
    def test_dual_reward_weighting(self):
        original = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]]
        )
        invariant = torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]]
        )
        actions = torch.tensor([[[1.0], [0.0], [1.0], [0.0]]])
        indices = torch.tensor([0, 2])
        original_reward = single_space_reward(original, indices)[2]
        invariant_reward = single_space_reward(invariant, indices)[2]
        actual = compute_reward(original, invariant, actions, beta=0.10)
        expected = 0.90 * original_reward + 0.10 * invariant_reward
        self.assertTrue(torch.allclose(actual, expected))


class AIMTest(unittest.TestCase):
    def test_per_video_update_equations(self):
        aim = AIMController()
        clip, lr_scale, state = aim.update("video_1", 1.0)
        self.assertAlmostEqual(state.intervention, 0.11)
        self.assertAlmostEqual(clip, 5.0 / 1.11)
        self.assertAlmostEqual(lr_scale, 1.0 / 1.11)
        aim.update_baseline("video_1", 1.0)
        self.assertAlmostEqual(aim.state_for("video_1").baseline, 0.1)
        self.assertAlmostEqual(aim.state_for("video_2").intervention, 0.10)


if __name__ == "__main__":
    unittest.main()
