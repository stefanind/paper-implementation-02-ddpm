# Denoising Diffusion Probabilistic Models (DDPM)

A minimal PyTorch implementation of
[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
by Jonathan Ho, Ajay Jain, and Pieter Abbeel (2020). This project implements the
paper's forward diffusion, noise-prediction objective, and reverse sampling loop
with a small U-Net trained on MNIST. It is written for learning, keeping the
equations and their use in a complete training loop easy to follow.

## Paper to code

1. **`ddpm.py`** implements the diffusion equations as six functions:
   - `linear_beta_schedule` and `get_alpha_bar` define beta, alpha, and cumulative alpha.
   - `forward_diffusion` samples noisy images directly from clean images (Eq. 4).
   - `compute_ddpm_loss` computes the noise-prediction MSE (Eq. 14).
   - `reverse_step` computes the reverse mean (Eq. 11), using fixed variance `beta_t`.
   - `ddpm_sample` runs the reverse chain from Gaussian noise (Algorithm 2).
2. **`model.py`** defines a small U-Net for MNIST. It takes a noisy image and its
   timestep and predicts the noise. A small MLP embeds `t / T`; one downsampling
   layer, one upsampling layer, and a skip connection handle the image. This is a
   simplified architecture for studying the algorithm.
3. **`train.py`** implements the training procedure (Algorithm 1):

   ```python
   t = torch.randint(1, T + 1, (batch_size,), device=device)
   epsilon = torch.randn_like(x_0)
   x_t = forward_diffusion(x_0, t, betas, epsilon)
   epsilon_pred = model(x_t, t)
   loss = compute_ddpm_loss(epsilon, epsilon_pred)
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()
   ```

After training, `ddpm_sample(x_T, betas, model)` starts with Gaussian noise and
predicts the noise in the current image at each reverse step. It calls
`reverse_step` so the reverse equation is defined in one place. No additional
noise is added at `t=1`. Clipping happens only when converting generated values
to a PNG.

Image and noise tensors have shape `[batch, 1, 28, 28]`; training timesteps have
shape `[batch]`. The `.view(-1, 1, 1, 1)` in `forward_diffusion` broadcasts each
image's alpha-bar over its pixels. `reverse_step` takes one integer timestep and
an explicit `z` tensor, as every image in the sampling batch follows the same step.

## Run

Python 3.10+, with PyTorch and torchvision:

```powershell
python -m pip install -r requirements.txt
python train.py --epochs 10 --device cpu
```

MNIST downloads to `data/` on the first run. Images stay at `28x28` and are
normalized to `[-1, 1]`. The example uses 1,000 diffusion steps, a linear beta
schedule from `0.0001` to `0.02`, batch size 128, and Adam at `1e-3`.
Edit these directly in `train.py`.
`--device cuda` requires a CUDA-enabled PyTorch installation; the script does
not select a different device if your requested device is unavailable.

The script trains, generates 16 images from Gaussian noise, and writes
`runs/learning/samples.png` and `runs/learning/model.pt`. Running it again
replaces those outputs. The weights file contains `model` and `betas`.
Ten epochs is a starting experiment, not a promise of sample quality.

Timesteps are **1 through T**. Assertions catch invalid
timesteps, mismatched shapes, nonfinite loss, and nonfinite gradients.
Run without `python -O` so assertions remain enabled.

## Check

```powershell
python -m unittest discover -s tests -v
```

The tests check the diffusion equations against analytical results, validate
the sampling loop, and train the model on a small batch to verify gradients
and weight updates.
