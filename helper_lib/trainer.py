# helper_lib/trainer.py
from __future__ import annotations
import torch
import torch.nn as nn
from torch.optim import Optimizer
from tqdm import tqdm

def train_model(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: str = "cpu",
    epochs: int = 2
) -> nn.Module:
    model.to(device)
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {ep}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        avg = running / len(train_loader.dataset)
        print(f"[train] epoch={ep} loss={avg:.4f}")
    return model

# ==== train_gan ==============================================================
import torch, torch.nn as nn
from tqdm import tqdm

def train_gan(
    gan, data_loader, device: str = "cpu",
    epochs: int = 5, latent_dim: int = 100,
    lr_g: float = 2e-4, lr_d: float = 2e-4, beta1: float = 0.5
):
    gan.to(device)
    G, D = gan.generator, gan.discriminator
    bce = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(beta1, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(beta1, 0.999))

    for ep in range(1, epochs+1):
        g_loss_sum = d_loss_sum = n_sum = 0
        for real,_ in tqdm(data_loader, desc=f"GAN Epoch {ep}/{epochs}"):
            real = real.to(device)                    # (B,1,28,28) in [-1,1]
            B = real.size(0)
            ones  = torch.ones(B, device=device)
            zeros = torch.zeros(B, device=device)

            # ---- update D ----
            D.train(); G.train()
            # real
            d_real = D(real)
            loss_d_real = bce(d_real, ones)
            # fake
            z = torch.randn(B, latent_dim, device=device)
            with torch.no_grad():
                fake = G(z)
            d_fake = D(fake.detach())
            loss_d_fake = bce(d_fake, zeros)

            loss_d = loss_d_real + loss_d_fake
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            # ---- update G ---- (non-saturating)
            z = torch.randn(B, latent_dim, device=device)
            fake = G(z)
            d_fake_for_g = D(fake)
            loss_g = bce(d_fake_for_g, ones)
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()

            g_loss_sum += loss_g.item()*B; d_loss_sum += loss_d.item()*B; n_sum += B

        print(f"[GAN] epoch={ep}  loss_G={g_loss_sum/n_sum:.4f}  loss_D={d_loss_sum/n_sum:.4f}")
    return gan

def _get_diffusion_schedule(T: int = 1000, device: str = "cpu"):
    """
    Create a simple linear beta schedule for DDPM.
    Returns betas, alphas, alpha_hats tensors on the given device.
    """
    betas = torch.linspace(1e-4, 0.02, T, device=device)           # (T,)
    alphas = 1.0 - betas                                           # (T,)
    alpha_hats = torch.cumprod(alphas, dim=0)                      # (T,)
    return betas, alphas, alpha_hats


def train_diffusion(model,
                    data_loader,
                    criterion,
                    optimizer,
                    device: str = "cpu",
                    epochs: int = 5,
                    num_steps: int = 1000):
    """
    Train a DDPM-like diffusion model.

    model      : DiffusionMLP (noise predictor)
    data_loader: MNIST-style loader (images in [-1,1])
    criterion  : e.g. nn.MSELoss()
    optimizer  : e.g. torch.optim.Adam(model.parameters(), lr=1e-3)
    """
    model.to(device)
    model.train()

    betas, alphas, alpha_hats = _get_diffusion_schedule(num_steps, device=device)

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for x, _ in tqdm(data_loader, desc=f"[diffusion] epoch {epoch}", leave=False):
            x = x.to(device)        # assume already scaled to [-1,1]
            b = x.size(0)

            # 1) sample random timesteps t for each sample
            t = torch.randint(0, num_steps, (b,), device=device)   # [0, T-1]

            # 2) compute q(x_t | x_0)
            alpha_hat_t = alpha_hats[t].view(-1, 1, 1, 1)          # (B,1,1,1)
            noise = torch.randn_like(x)
            x_t = torch.sqrt(alpha_hat_t) * x + torch.sqrt(1 - alpha_hat_t) * noise

            # 3) predict noise epsilon_theta(x_t, t)
            optimizer.zero_grad()
            pred_noise = model(x_t, t)

            # 4) loss = MSE(pred, true_noise)
            loss = criterion(pred_noise, noise)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * b

        avg_loss = running_loss / len(data_loader.dataset)
        print(f"[diffusion] epoch={epoch} loss={avg_loss:.4f}")

    return model

def sample_ebm_negatives(model,
                         batch_size: int,
                         img_shape: tuple[int, int, int] = (1, 28, 28),
                         device: str = "cpu",
                         steps: int = 20,
                         step_size: float = 0.1,
                         noise_scale: float = 0.01):
    """
    Use Langevin-like dynamics to sample low-energy images from the EBM.

    We start from Gaussian noise x_0 ~ N(0, I) and perform gradient descent
    on the input image to reduce the energy E(x), with some noise added.
    """
    c, h, w = img_shape
    x = torch.randn(batch_size, c, h, w, device=device).clamp(-1.0, 1.0)

    for _ in range(steps):
        x.requires_grad_(True)
        energy = model(x).sum()                    # scalar
        grad_x, = torch.autograd.grad(
            energy, x, create_graph=False, retain_graph=False
        )
        # gradient descent on x to lower energy + Gaussian noise
        x = x - step_size * grad_x + noise_scale * torch.randn_like(x)
        x = x.clamp(-1.0, 1.0).detach()

    return x

def train_ebm(model,
              data_loader,
              optimizer,
              device: str = "cpu",
              epochs: int = 5,
              steps_per_batch: int = 20,
              step_size: float = 0.1,
              noise_scale: float = 0.01):
    """
    Train an energy-based model with contrastive loss:
    minimize E(x_pos) for real data, maximize E(x_neg) for negative samples.

    model      : EnergyCNN
    data_loader: MNIST loader, images scaled to [-1,1]
    optimizer  : e.g. Adam(model.parameters(), lr=1e-4)
    """
    model.to(device)
    model.train()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        n_samples = 0

        for x_pos, _ in tqdm(data_loader, desc=f"[ebm] epoch {epoch}", leave=False):
            x_pos = x_pos.to(device)              # (B,1,28,28)
            b = x_pos.size(0)

            # 1) Sample negative images from the current model
            img_shape = tuple(x_pos.shape[1:])
            x_neg = sample_ebm_negatives(
                model,
                batch_size=b,
                img_shape=img_shape,
                device=device,
                steps=steps_per_batch,
                step_size=step_size,
                noise_scale=noise_scale,
            )

            # 2) Compute contrastive loss: E(x_pos) - E(x_neg)
            optimizer.zero_grad()
            energy_pos = model(x_pos)            # (B,)
            energy_neg = model(x_neg)            # (B,)
            loss = energy_pos.mean() - energy_neg.mean()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * b
            n_samples += b

        avg_loss = running_loss / max(1, n_samples)
        print(f"[ebm] epoch={epoch} loss={avg_loss:.4f}")

    return model
