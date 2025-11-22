# app/services/ebm_service.py
import io
import base64
import time
from pathlib import Path

import torch
from PIL import Image

from helper_lib.model import get_model


class EBMSampler:
    """
    Wraps the EBM model (EnergyCNN) and provides sampling via Langevin dynamics.
    """

    def __init__(
        self,
        weights_path: str = "weights/ebm_mnist.pt",
        device: str = "cpu",
    ):
        self.device = device
        self.model = get_model("EBM", dataset="mnist").to(self.device)
        try:
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"[INFO] Loaded EBM weights: {weights_path}")
        except Exception as e:
            print(
                f"[WARN] Could not load EBM weights ({weights_path}): "
                f"{e}. Using untrained model."
            )
        self.model.eval()

    def _langevin(
        self,
        n: int,
        steps: int,
        step_size: float,
        noise_scale: float,
        seed: int | None = None,
    ):
        if seed is not None:
            torch.manual_seed(seed)

        x = torch.randn(n, 1, 28, 28, device=self.device).clamp(-1.0, 1.0)

        for _ in range(steps):
            x.requires_grad_(True)
            energy = self.model(x).sum()
            grad_x, = torch.autograd.grad(
                energy, x, create_graph=False, retain_graph=False
            )
            x = x - step_size * grad_x + noise_scale * torch.randn_like(x)
            x = x.clamp(-1.0, 1.0).detach()

        return x

    def sample_base64_pngs(
        self,
        n: int = 4,
        steps: int = 60,
        step_size: float = 0.1,
        noise_scale: float = 0.01,
        seed: int | None = None,
    ) -> list[str]:
        """
        Return n images as base64 PNG data URLs using Langevin sampling.
        """
        with torch.no_grad():
            x = self._langevin(n, steps, step_size, noise_scale, seed)
            imgs = (x.clamp(-1.0, 1.0) + 1.0) * 0.5

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

    def save_pngs(
        self,
        n: int = 16,
        steps: int = 60,
        step_size: float = 0.1,
        noise_scale: float = 0.01,
        seed: int | None = None,
        out_dir: str = "weights/ebm_samples",
        prefix: str = "ebm",
    ) -> list[str]:
        """
        Save generated images as PNG files on disk and return absolute paths.
        """
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            x = self._langevin(n, steps, step_size, noise_scale, seed)
            imgs = (x.clamp(-1.0, 1.0) + 1.0) * 0.5

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
