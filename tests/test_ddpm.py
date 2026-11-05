import contextlib
import io
import math
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ddpm import (get_alpha_bar, forward_diffusion, reverse_step,
                  linear_beta_schedule, compute_ddpm_loss, ddpm_sample)
from model import NoisePredictor
from train import train


class DDPMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_forward_and_reverse_match_analytical_results(self):
        betas = linear_beta_schedule(2, 0.1, 0.2)
        torch.testing.assert_close(betas, torch.tensor([0.1, 0.2]))
        torch.testing.assert_close(get_alpha_bar(betas), torch.tensor([0.9, 0.72]))
        x0 = torch.tensor([[[[0.2, -0.4]]], [[[0.1, 0.5]]]])
        epsilon = torch.full_like(x0, 0.7)
        xt = forward_diffusion(x0, torch.tensor([1, 2]), betas, epsilon)
        torch.testing.assert_close(xt[0], math.sqrt(0.9) * x0[0] + math.sqrt(0.1) * epsilon[0])
        torch.testing.assert_close(xt[1], math.sqrt(0.72) * x0[1] + math.sqrt(0.28) * epsilon[1])
        xt = forward_diffusion(x0, torch.tensor([2, 2]), betas, epsilon)
        # Independent posterior-mean expression from the paper's Eq. 7.
        posterior = (0.2 * math.sqrt(0.9) * x0 + 0.1 * math.sqrt(0.8) * xt) / 0.28
        torch.testing.assert_close(reverse_step(xt, 2, epsilon, betas, torch.zeros_like(xt)), posterior)
        noisy = reverse_step(xt, 2, epsilon, betas, torch.ones_like(xt))
        torch.testing.assert_close(noisy - posterior, torch.full_like(xt, math.sqrt(0.2)))

    def test_loss_remains_differentiable_without_rounding(self):
        prediction = torch.tensor([0.0001], requires_grad=True)
        loss = compute_ddpm_loss(torch.zeros_like(prediction), prediction)
        self.assertGreater(loss.item(), 0)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([0.0002]))

    def test_invalid_timesteps_and_shapes_fail(self):
        betas = linear_beta_schedule(3)
        x = torch.zeros(2, 1, 28, 28)
        for t in (torch.tensor([0, 1]), torch.tensor([1, 4]), torch.tensor([1.0, 2.0])):
            with self.subTest(t=t), self.assertRaises(AssertionError):
                forward_diffusion(x, t, betas, x)
        with self.assertRaises(AssertionError):
            compute_ddpm_loss(x, x[:1])
        with self.assertRaises(AssertionError):
            reverse_step(x, 1, x, betas, x[:1])
        with self.assertRaises(AssertionError):
            compute_ddpm_loss(x, x + float("nan"))

    def test_final_reverse_step_adds_no_noise_or_clipping(self):
        betas = linear_beta_schedule(3)
        xt = torch.full((1, 1, 1, 1), 2.0)
        prediction = torch.zeros_like(xt)
        result = reverse_step(xt, 1, prediction, betas, torch.ones_like(xt) * 100)
        torch.testing.assert_close(result, xt / (1 - betas[0]).sqrt())
        self.assertGreater(result.item(), 1)

    def test_sampling_uses_live_predictions_and_is_repeatable(self):
        class Recorder(nn.Module):
            def __init__(self):
                super().__init__()
                self.timesteps, self.inputs = [], []

            def forward(self, x, t):
                self.timesteps.append(int(t[0]))
                self.inputs.append(x.clone())
                return x * 0.1

        model = Recorder().eval()
        betas = linear_beta_schedule(3, 0.01, 0.03)
        noise = torch.full((2, 1, 2, 2), 0.25)
        torch.manual_seed(7)
        result = ddpm_sample(noise, betas, model)
        torch.manual_seed(7)
        repeated = ddpm_sample(noise, betas, model)
        torch.testing.assert_close(result, repeated, atol=0, rtol=0)
        self.assertEqual(model.timesteps, [3, 2, 1, 3, 2, 1])
        self.assertFalse(torch.equal(model.inputs[0], model.inputs[1]))
        self.assertFalse(result.requires_grad)

    def test_real_training_loop_updates_timestep_conditioned_model(self):
        torch.manual_seed(3)
        model = NoisePredictor(T=1000)
        images = torch.rand(4, 1, 28, 28) * 2 - 1
        loader = DataLoader(TensorDataset(images, torch.zeros(4)), batch_size=2)
        before = model.output[-1].weight.detach().clone()
        with contextlib.redirect_stdout(io.StringIO()):
            train(model, loader, linear_beta_schedule(1000), epochs=1)
        self.assertFalse(torch.equal(before, model.output[-1].weight))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        early = model(images, torch.ones(4, dtype=torch.long))
        late = model(images, torch.full((4,), 1000, dtype=torch.long))
        self.assertEqual(early.shape, images.shape)
        self.assertFalse(torch.allclose(early, late))


if __name__ == "__main__":
    unittest.main()
