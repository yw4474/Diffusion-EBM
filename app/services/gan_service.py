# app/services/gan_service.py
import io, base64, os, time, torch
from pathlib import Path
from PIL import Image
from helper_lib.model import TConvGenerator  # 你的 TConvGenerator 在 helper_lib/model.py 里

class GANSampler:
    def __init__(self, weights_path: str = "weights/gan_mnist_gen.pt",
                 device: str = "cpu", latent_dim: int = 100):
        self.device = device
        self.latent_dim = latent_dim
        self.G = TConvGenerator(latent_dim).to(device)
        try:
            state = torch.load(weights_path, map_location=device)
            self.G.load_state_dict(state)
            self.G.eval()
            print(f"[INFO] Loaded GAN generator: {weights_path}")
        except Exception as e:
            print(f"[WARN] Failed to load {weights_path}: {e}. Using random init.")

    @torch.no_grad()
    def sample_base64_pngs(self, n: int = 4) -> list[str]:
        z = torch.randn(n, self.latent_dim, device=self.device)
        imgs = self.G(z).cpu()                           # [-1,1]
        imgs = ((imgs + 1.0) * 0.5).clamp(0, 1)          # [0,1]
        out = []
        for i in range(n):
            arr = (imgs[i, 0].numpy() * 255).astype("uint8")
            img = Image.fromarray(arr, mode="L")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            out.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"))
        return out

    @torch.no_grad()
    def sample_png_files(self, n: int = 4, out_dir: str = "weights/gan_samples", prefix: str = "gan") -> list[str]:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        z = torch.randn(n, self.latent_dim, device=self.device)
        imgs = self.G(z).cpu()
        imgs = ((imgs + 1.0) * 0.5).clamp(0, 1)

        ts = time.strftime("%Y%m%d_%H%M%S")
        paths = []
        for i in range(n):
            arr = (imgs[i, 0].numpy() * 255).astype("uint8")
            img = Image.fromarray(arr, mode="L")
            fname = f"{prefix}_{ts}_{i+1:02d}.png"
            fpath = str(Path(out_dir) / fname)
            img.save(fpath, format="PNG")
            paths.append(os.path.abspath(fpath))
        return paths

