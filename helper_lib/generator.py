# helper_lib/generator.py
from __future__ import annotations
import torch
import matplotlib.pyplot as plt

def _denorm(x: torch.Tensor) -> torch.Tensor:
    # x in [-1,1] -> [0,1] for display
    return (x + 1.0) * 0.5

def generate_samples(gan, device: str = "cpu", num_samples: int = 16, nrow: int = 4):
    """
    Sample 'num_samples' images from GAN.generator and show a grid.
    """
    gan.to(device)
    with torch.no_grad():
        imgs = gan.sample(num_samples, device=device).cpu()
        imgs = _denorm(imgs).clamp(0, 1)
    # make a simple grid
    fig, axes = plt.subplots(nrow, nrow, figsize=(nrow*2, nrow*2))
    idx = 0
    for r in range(nrow):
        for c in range(nrow):
            ax = axes[r, c]
            ax.imshow(imgs[idx, 0].numpy(), cmap="gray")
            ax.axis("off")
            idx += 1
    plt.tight_layout()
    plt.show()


def _get_diffusion_schedule(T: int = 1000, device: str = "cpu"):
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1.0 - betas
    alpha_hats = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_hats


@torch.no_grad()
def generate_samples(model,
                     device: str = "cpu",
                     num_samples: int = 16,
                     diffusion_steps: int = 1000):
    """
    Generate images using the trained diffusion model.

    model : trained DiffusionMLP
    device: "cpu" or "cuda"
    """
    model.to(device)
    model.eval()

    betas, alphas, alpha_hats = _get_diffusion_schedule(diffusion_steps, device=device)

    # start from pure Gaussian noise
    x_t = torch.randn(num_samples, 1, 28, 28, device=device)

    for t_step in reversed(range(diffusion_steps)):
        t = torch.full((num_samples,), t_step, device=device, dtype=torch.long)

        alpha_t = alphas[t_step]
        alpha_hat_t = alpha_hats[t_step]

        # predict noise
        pred_noise = model(x_t, t)

        # reverse diffusion step (DDPM mean)
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)
        x_prev = coef1 * (x_t - coef2 * pred_noise)

        if t_step > 0:
            z = torch.randn_like(x_t)
            sigma_t = torch.sqrt(betas[t_step])
            x_t = x_prev + sigma_t * z
        else:
            x_t = x_prev

    # x_t is now x_0 in [-1,1]; rescale to [0,1] for plotting
    imgs = (x_t.clamp(-1, 1) + 1) * 0.5          # (B,1,28,28)

    # plot on a grid
    n = num_samples
    cols = int(n**0.5)
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(imgs[i, 0].cpu().numpy(), cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def generate_ebm_samples(model,
                         device: str = "cpu",
                         num_samples: int = 16,
                         steps: int = 60,
                         step_size: float = 0.1,
                         noise_scale: float = 0.01):
    """
    Generate images from a trained EBM by starting from noise and doing
    gradient descent on the input to lower energy.

    model : trained EnergyCNN
    """
    model.to(device)
    model.eval()

    c, h, w = 1, 28, 28
    x = torch.randn(num_samples, c, h, w, device=device).clamp(-1.0, 1.0)

    for _ in range(steps):
        x.requires_grad_(True)
        energy = model(x).sum()
        grad_x, = torch.autograd.grad(
            energy, x, create_graph=False, retain_graph=False
        )
        x = x - step_size * grad_x + noise_scale * torch.randn_like(x)
        x = x.clamp(-1.0, 1.0).detach()

    # Rescale from [-1,1] to [0,1] for plotting
    imgs = (x.clamp(-1.0, 1.0) + 1.0) * 0.5

    n = num_samples
    cols = int(n ** 0.5)
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(imgs[i, 0].cpu().numpy(), cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
