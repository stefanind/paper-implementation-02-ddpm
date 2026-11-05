import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from ddpm import linear_beta_schedule, forward_diffusion, compute_ddpm_loss, ddpm_sample
from model import NoisePredictor


def train(model: NoisePredictor, loader: DataLoader, betas: torch.Tensor, epochs: int) -> None:
    assert epochs > 0 and len(loader) > 0
    assert model.T == len(betas), "Model and diffusion must use the same T"
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x_0, _ in loader:
            x_0 = x_0.to(betas.device)
            assert torch.isfinite(x_0).all() and x_0.abs().max() <= 1, "Normalize images to [-1, 1]"
            t = torch.randint(1, len(betas) + 1, (x_0.shape[0],), device=betas.device)
            epsilon = torch.randn_like(x_0)
            x_t = forward_diffusion(x_0, t, betas, epsilon)
            epsilon_pred = model(x_t, t)
            loss = compute_ddpm_loss(epsilon, epsilon_pred)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch={epoch + 1} loss={total_loss / len(loader):.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cpu", help="cpu or cuda; no automatic selection")
    args = parser.parse_args()
    assert args.epochs > 0
    torch.manual_seed(0)
    betas = linear_beta_schedule(1000).to(args.device)
    model = NoisePredictor(T=len(betas)).to(args.device)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    train(model, loader, betas, args.epochs)

    model.eval()
    x_T = torch.randn(16, 1, 28, 28, device=args.device)
    samples = ddpm_sample(x_T, betas, model)
    output = Path("runs/learning")
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "betas": betas.cpu()}, output / "model.pt")
    # Clamp for PNG display only; reverse diffusion uses unbounded tensors.
    save_image((samples.clamp(-1, 1) + 1) / 2, output / "samples.png", nrow=4)


if __name__ == "__main__":
    main()
