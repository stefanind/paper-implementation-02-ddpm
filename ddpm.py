import torch
from torch import Tensor, nn


def get_alpha_bar(betas: Tensor) -> Tensor:
    assert betas.ndim == 1 and betas.numel() > 0
    assert ((0 < betas) & (betas < 1)).all(), "Betas must be between 0 and 1"
    return torch.cumprod(1 - betas, dim=0)


def forward_diffusion(x_0: Tensor, t: Tensor, betas: Tensor, epsilon: Tensor) -> Tensor:
    assert x_0.ndim == 4 and x_0.shape == epsilon.shape, "Expected matching BCHW tensors"
    assert t.shape == (x_0.shape[0],) and t.dtype == torch.long
    assert x_0.device == epsilon.device == t.device == betas.device
    assert ((1 <= t) & (t <= len(betas))).all(), "Timesteps must be in 1..T"
    alpha_bar = get_alpha_bar(betas)[t - 1].view(-1, 1, 1, 1)
    x_t = alpha_bar.sqrt() * x_0 + (1 - alpha_bar).sqrt() * epsilon
    assert torch.isfinite(x_t).all(), "Nonfinite forward diffusion"
    return x_t


def reverse_step(x_t: Tensor, t: int, epsilon_pred: Tensor, betas: Tensor, z: Tensor) -> Tensor:
    assert 1 <= t <= len(betas), "Timesteps must be in 1..T"
    assert x_t.shape == epsilon_pred.shape == z.shape, "Noise tensors must match x_t"
    assert x_t.device == epsilon_pred.device == z.device == betas.device
    alphas = 1 - betas
    alpha_bar = get_alpha_bar(betas)
    mu_t = (1 / alphas[t - 1].sqrt()) * (
        x_t - (betas[t - 1] / (1 - alpha_bar[t - 1]).sqrt()) * epsilon_pred
    )
    x_previous = mu_t if t == 1 else mu_t + betas[t - 1].sqrt() * z
    assert torch.isfinite(x_previous).all(), "Nonfinite reverse diffusion"
    return x_previous


def linear_beta_schedule(T: int, beta_1: float = 0.0001, beta_T: float = 0.02) -> Tensor:
    assert T > 0 and 0 < beta_1 <= beta_T < 1, "Invalid beta schedule"
    return torch.linspace(beta_1, beta_T, T, dtype=torch.float32)


def compute_ddpm_loss(epsilon: Tensor, epsilon_pred: Tensor) -> Tensor:
    assert epsilon.shape == epsilon_pred.shape, "Noise prediction must match the target"
    loss = ((epsilon - epsilon_pred) ** 2).mean()
    assert torch.isfinite(loss), "Nonfinite loss"
    return loss


@torch.no_grad()
def ddpm_sample(x_T: Tensor, betas: Tensor, model: nn.Module) -> Tensor:
    assert not model.training, "Call model.eval() before sampling"
    x_t = x_T
    for t in range(len(betas), 0, -1):
        timesteps = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        epsilon_pred = model(x_t, timesteps)
        z = torch.randn_like(x_t)
        x_t = reverse_step(x_t, t, epsilon_pred, betas, z)
    return x_t
