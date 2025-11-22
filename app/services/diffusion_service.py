# app/services/diffusion_service.py
import io
import base64
import time
from pathlib import Path

import torch
from PIL import Image

from helper_lib.model import get_model


class DiffusionSampler:
    """
    Wraps the Diffusion model (DiffusionMLP) and provides convenient
    sampling methods for the API (base64 or saved PNGs).
    """

    def __init__(
        self,
        weights_path: str = "weights/diffusion_mnist.pt",
        device: str = "cpu",
        timesteps: int = 200,
    ):
        self.device = device
        self.timesteps = timesteps

        # get diffusion model from helper_lib
        self.model = get_model("Diffusion", dataset="mnist").to(self.device)
        try:
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"[INFO] Loaded diffusion weights: {weights_path}")
        except Exception as e:
            print(
                f"[WARN] Could not load diffusion weights ({weights_path}): "
                f"{e}. Using untrained model."
            )

        self.model.eval()

    def _schedule(self, T: int):
        """
        Linear beta schedule, must be consistent with training.
        """
        betas = torch.linspace(1e-4, 0.02, T, device=self.device)
        alphas = 1.0 - betas
        alpha_hats = torch.cumprod(alphas, dim=0)
        return betas, alphas, alpha_hats

    @torch.no_grad()
    def sample_base64_pngs(
        self,
        n: int = 4,
        steps: int | None = None,
        seed: int | None = None,
    ) -> list[str]:
        """
        Return n images as base64 PNG data URLs.
        """
        if steps is None:
            steps = self.timesteps

        if seed is not None:
            torch.manual_seed(seed)

        betas, alphas, alpha_hats = self._schedule(steps)

        # start from Gaussian noise
        x_t = torch.randn(n, 1, 28, 28, device=self.device)

        for t_step in reversed(range(steps)):
            t = torch.full((n,), t_step, device=self.device, dtype=torch.long)
            alpha_t = alphas[t_step]
            alpha_hat_t = alpha_hats[t_step]

            pred_noise = self.model(x_t, t)

            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)
            x_prev = coef1 * (x_t - coef2 * pred_noise)

            if t_step > 0:
                z = torch.randn_like(x_t)
                sigma_t = torch.sqrt(betas[t_step])
                x_t = x_prev + sigma_t * z
            else:
                x_t = x_prev

        imgs = (x_t.clamp(-1, 1) + 1.0) * 0.5  # [0,1]

        out: list[str] = []
        for i in range(n):
            arr = (imgs[i, 0].cpu().numpy() * 255).astype("uint8")
            img = Image.fromarray(arr, mode="L")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            out.append(
                "data:image/png;base64,"
                + base64.b64encode(buf.getvalue()).decode("ascii")
            )
        return out

    @torch.no_grad()
    def save_pngs(
        self,
        n: int = 16,
        steps: int | None = None,
        seed: int | None = None,
        out_dir: str = "weights/diffusion_samples",
        prefix: str = "diff",
    ) -> list[str]:
        """
        Save generated images as PNG files on disk and return absolute paths.
        """
        if steps is None:
            steps = self.timesteps

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        if seed is not None:
            torch.manual_seed(seed)

        betas, alphas, alpha_hats = self._schedule(steps)
        x_t = torch.randn(n, 1, 28, 28, device=self.device)

        for t_step in reversed(range(steps)):
            t = torch.full((n,), t_step, device=self.device, dtype=torch.long)
            alpha_t = alphas[t_step]
            alpha_hat_t = alpha_hats[t_step]

            pred_noise = self.model(x_t, t)

            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)
            x_prev = coef1 * (x_t - coef2 * pred_noise)

            if t_step > 0:
                z = torch.randn_like(x_t)
                sigma_t = torch.sqrt(betas[t_step])
                x_t = x_prev + sigma_t * z
            else:
                x_t = x_prev

        imgs = (x_t.clamp(-1, 1) + 1.0) * 0.5

        ts = time.strftime("%Y%m%d_%H%M%S")
        paths: list[str] = []
        for i in range(n):
            arr = (imgs[i, 0].cpu().numpy() * 255).astype("uint8")
            img = Image.fromarray(arr, mode="L")
            fname = f"{prefix}_{ts}_{i+1:02d}.png"
            p = Path(out_dir) / fname
            img.save(p, format="PNG")
            paths.append(str(p.resolve()))

        return paths
