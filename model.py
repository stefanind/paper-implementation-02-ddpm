"""A small U-Net that predicts the noise in a 28x28 MNIST image."""

import torch
from torch import Tensor, nn


class NoisePredictor(nn.Module):
    def __init__(self, T: int):
        super().__init__()
        assert T > 0
        self.T = T
        self.time = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
        self.input = nn.Conv2d(1, 32, 3, padding=1)
        self.down = nn.Sequential(
            nn.GroupNorm(4, 32), nn.SiLU(), nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.SiLU(), nn.Conv2d(64, 64, 3, padding=1), nn.SiLU(),
        )
        self.up = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.output = nn.Sequential(
            nn.GroupNorm(4, 64), nn.SiLU(), nn.Conv2d(64, 32, 3, padding=1),
            nn.SiLU(), nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        assert x_t.ndim == 4 and x_t.shape[1:] == (1, 28, 28), "Expected BCHW MNIST images"
        assert t.shape == (x_t.shape[0],) and t.dtype == torch.long
        assert ((1 <= t) & (t <= self.T)).all(), "Timesteps must be in 1..T"
        time = self.time(t[:, None].float() / self.T)[:, :, None, None]
        h = self.input(x_t) + time
        return self.output(torch.cat((h, self.up(self.down(h))), dim=1))
