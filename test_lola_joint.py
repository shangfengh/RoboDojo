import copy
import unittest

import torch

from lola_joint import JointNormalizer, LoLAJointCore, joint_config


def example_batch():
    state = torch.linspace(0, 1, 28).reshape(2, 14)
    history = torch.zeros(2, 32, 14)
    history[:, -1] = state
    mask = torch.zeros(2, 32, dtype=torch.bool)
    mask[:, -1] = True
    valid = torch.ones(2, 16, dtype=torch.bool)
    valid[1, 3:] = False
    return {
        "observation.state": state, "history.state": history, "history.valid_mask": mask,
        "action": torch.full((2, 16, 14), .3), "action.valid_mask": valid,
        "frame_index": torch.zeros(2, dtype=torch.int64),
    }


def example_stats():
    return {"partition": "train", "features": {
        key: {"mean": [.2] * 14, "std": [.5] * 14}
        for key in ("observation.state", "action")
    }}


class LoLAJointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.model = LoLAJointCore(joint_config(small_test=True), example_stats())
        self.batch = example_batch()
        self.visual = torch.randn(2, 3, 64)
        self.global_condition = torch.randn(2, 64)
        self.mask = torch.ones(2, 3, dtype=torch.bool)
        self.time = torch.tensor([.4, .7])
        self.noise = (torch.randn(2, 2, 32), torch.randn(2, 2, 16))

    def forward_model(self, batch=None):
        return self.model(self.batch if batch is None else batch, self.visual,
                          self.global_condition, self.mask, self.time, self.noise)

    def test_normalization_roundtrip_preserves_gripper_values(self):
        normalizer = JointNormalizer(example_stats())
        raw = torch.rand(3, 16, 14)
        normalized = normalizer.action(raw)
        torch.testing.assert_close(normalized[..., [6, 13]], raw[..., [6, 13]], rtol=0, atol=0)
        torch.testing.assert_close(normalizer.raw_action(normalized), raw)
        torch.testing.assert_close(normalizer.state(raw)[..., [0, 7]], (raw[..., [0, 7]] - .2) / .5)

    def test_continuous_loss_and_backward(self):
        result = self.forward_model()
        self.assertEqual(result["prediction"].shape, (2, 16, 14))
        self.assertEqual(result["full_flow_chunks"].item(), 2)
        prediction = result["prediction"][..., [6, 13]]
        valid = self.batch["action.valid_mask"]
        expected = ((prediction - .3).square() * valid[..., None]).sum() / (valid.sum() * 2)
        torch.testing.assert_close(result["gripper_loss"], expected)
        result["loss"].backward()
        for module in (self.model.action_encoder, self.model.state_encoder,
                       self.model.dit.gripper_out_proj, self.model.dit.arm_out_proj):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)

    def test_padding_content_cannot_change_loss(self):
        expected = self.forward_model()
        batch = copy.deepcopy(self.batch)
        batch["action"][~batch["action.valid_mask"]] = 12345
        batch["history.state"][~batch["history.valid_mask"]] = 9876
        actual = self.forward_model(batch)
        for key in ("loss", "flow_loss", "arm_loss", "gripper_loss", "prediction"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_sampling_is_continuous_bounded_and_deterministic(self):
        self.model.eval()
        first = self.model.sample(self.batch, self.visual, self.global_condition, self.mask, self.noise)
        second = self.model.sample(self.batch, self.visual, self.global_condition, self.mask, self.noise)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        grip = first[..., [6, 13]]
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(((grip >= 0) & (grip <= 1)).all())
        self.assertTrue(((grip > .01) & (grip < .99)).any())

    def test_partial_chunks_have_no_flow_loss(self):
        self.batch["action.valid_mask"][:, 3:] = False
        result = self.forward_model()
        self.assertEqual(result["flow_loss"].item(), 0)
        self.assertTrue(torch.isfinite(result["loss"]))

    def test_invalid_inputs_rejected(self):
        bad = copy.deepcopy(self.batch)
        bad["history.valid_mask"][0, 0] = True
        with self.assertRaisesRegex(ValueError, "left-padded"):
            self.forward_model(bad)
        bad = copy.deepcopy(self.batch)
        bad["action"][0, 0, 6] = 2
        with self.assertRaisesRegex(ValueError, "gripper targets"):
            self.forward_model(bad)
        self.time[0] = 0
        with self.assertRaisesRegex(ValueError, "Flow time"):
            self.forward_model()

    def test_saved_weights_preserve_sampler(self):
        replacement = LoLAJointCore(joint_config(small_test=True), example_stats())
        replacement.load_state_dict(self.model.state_dict(), strict=True)
        args = (self.batch, self.visual, self.global_condition, self.mask, self.noise)
        torch.testing.assert_close(self.model.sample(*args), replacement.sample(*args), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
