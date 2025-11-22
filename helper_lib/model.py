# helper_lib/model.py
from __future__ import annotations
import torch
import torch.nn as nn

# ------------------------ FCNN ------------------------
class FCNN(nn.Module):
    """
    Fully-connected network. Uses LazyLinear to auto-infer input features
    from whatever image size/channels you pass (MNIST or CIFAR-10).
    """
    def __init__(self, hidden: int = 256, num_classes: int = 10, p_drop: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ------------------------ CNN ------------------------
class CNN(nn.Module):
    """
    Small CNN that works for 1-channel (MNIST) or 3-channel (CIFAR-10).
    AdaptiveAvgPool2d(4,4) fixes the spatial size before the linear head (no shape math headaches).
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10, p_drop: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),          # 28->14 or 32->16
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),          # 14->7 or 16->8
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))  # -> (128,4,4) regardless of input size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(256, num_classes),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)

# ------------------------ Enhanced CNN (optional) ------------------------
class EnhancedCNN(nn.Module):
    """
    Slightly deeper CNN with BatchNorm; still uses AdaptiveAvgPool2d to keep FC size stable.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10, p_drop: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(256, num_classes),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)
# ------------------------ CNN64 (optional) ------------------------
class CNN64(nn.Module):
    """
    CNN for 3x64x64 inputs (CIFAR-10 resized to 64).
    """
    def __init__(self, num_classes: int = 10, p_drop: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(128,128, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8*8, 256), nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
# ------------------------ SpecCNN (optional) ------------------------
import torch
import torch.nn as nn

class SpecCNN(nn.Module):
    """
    Assignment-spec CNN:
    64x64x3 -> [Conv16 3x3 s1 p1, ReLU, MaxPool2] -> [Conv32 3x3 s1 p1, ReLU, MaxPool2]
    -> Flatten -> FC100 -> ReLU -> FC10
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 64 -> 32
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 32 -> 16
        )
        # After two pools: (C=32, H=W=16) => 32*16*16 = 8192
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 100), nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

# ==== GAN (DCGAN-lite for 1×28×28 MNIST) ====================================
import torch
import torch.nn as nn

class GANGenerator(nn.Module):
    """
    z ~ N(0,1) -> 1×28×28 image (tanh in [-1,1]).
    Latent dim defaults to 100. Works well for MNIST.
    """
    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.BatchNorm1d(256), nn.ReLU(True),
            nn.Linear(256, 512),        nn.BatchNorm1d(512), nn.ReLU(True),
            nn.Linear(512, 1024),       nn.BatchNorm1d(1024), nn.ReLU(True),
            nn.Linear(1024, 1*28*28), nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.net(z)
        return x.view(z.size(0), 1, 28, 28)

class GANDiscriminator(nn.Module):
    """
    1×28×28 -> real/fake logit. BCEWithLogitsLoss for training.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1*28*28, 512), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),     nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)  # logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)

class GAN(nn.Module):
    """
    Wrapper holding generator and discriminator together.
    """
    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = GANGenerator(latent_dim)
        self.discriminator = GANDiscriminator()

    @torch.no_grad()
    def sample(self, n: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        imgs = self.generator(z)
        return imgs

# ====== Transposed-Conv GAN for MNIST  ===================
import torch, torch.nn as nn

class TConvGenerator(nn.Module):
    """
    z: (B, 100) -> (B, 1, 28, 28)
    FC -> 7x7x128 -> ConvT(128->64, k4,s2,p1) -> BN+ReLU -> ConvT(64->1, k4,s2,p1) -> Tanh
    """
    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 7*7*128),
            nn.BatchNorm1d(7*7*128),
            nn.ReLU(True),
        )
        self.deconv = nn.Sequential(
            # 128x7x7 -> 64x14x14
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64x14x14 -> 1x28x28
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)                       # (B, 7*7*128)
        x = x.view(z.size(0), 128, 7, 7)     # (B,128,7,7)
        x = self.deconv(x)                   # (B,1,28,28)
        return x

class TConvDiscriminator(nn.Module):
    """
    x: (B,1,28,28) -> logit
    Conv(1->64, k4,s2,p1) + LReLU(0.2) -> Conv(64->128, k4,s2,p1) + BN + LReLU -> Flatten -> Linear(1)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=True),  # 28->14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 14->7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = nn.Linear(128*7*7, 1)  # logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)                    # (B,128,7,7)
        h = h.view(x.size(0), -1)          # (B, 128*7*7)
        return self.head(h).view(-1)       # (B,)

class TConvGAN(nn.Module):
    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = TConvGenerator(latent_dim)
        self.discriminator = TConvDiscriminator()

    @torch.no_grad()
    def sample(self, n: int, device: str = "cpu"):
        z = torch.randn(n, self.latent_dim, device=device)
        imgs = self.generator(z)
        return imgs

class DiffusionMLP(nn.Module):
    """
    Simple noise-prediction network for DDPM-style diffusion on 28x28 images.

    x:  (B, 1, 28, 28) or (B, 784)
    t:  (B,) integer timesteps, e.g. 1..T
    out: predicted noise epsilon, same shape as x
    """
    def __init__(self, img_size: int = 28, time_dim: int = 128, hidden_dim: int = 1024):
        super().__init__()
        self.img_size = img_size
        self.img_dim = img_size * img_size

        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        self.net = nn.Sequential(
            nn.Linear(self.img_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.img_dim),
        )

    def forward(self, x, t):
        """
        x: (B,1,28,28) or (B,784)
        t: (B,) int timesteps
        """
        if x.dim() == 4:
            b, c, h, w = x.shape
            x = x.view(b, -1)
        else:
            b = x.size(0)

        # normalize timesteps to [0,1]
        t = t.view(-1, 1).float()
        t = t / (t.max() + 1e-8)
        t_emb = self.time_mlp(t)          # (B, time_dim)

        x_in = torch.cat([x, t_emb], dim=1)
        out = self.net(x_in)              # (B, img_dim)
        out = out.view(b, 1, self.img_size, self.img_size)
        return out


class EnergyCNN(nn.Module):
    """
    Simple energy-based model for 28x28 grayscale images (e.g. MNIST).
    Input:  x (B,1,28,28)
    Output: energy (B,)  -- lower for more "plausible" images
    """
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # 14x14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),          # 7x7
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),         # 7x7
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                     # 128*7*7
            nn.Linear(128 * 7 * 7, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),                # scalar energy
        )

    def forward(self, x):
        h = self.features(x)
        e = self.fc(h)                        # (B,1)
        return e.squeeze(-1)                  # (B,)

# ------------------------ Factory ------------------------
def get_model(name: str, dataset: str = "mnist"):
    """
    Return a model by name.
      - FCNN, CNN, EnhancedCNN respect `dataset` for input channels (1 for MNIST, 3 for CIFAR-10).
      - CNN64 is a fixed 3-channel model for 64x64 RGB inputs (e.g., CIFAR-10 resized to 64).
    """
    n = name.lower()
    in_ch = 1 if dataset.lower() == "mnist" else 3

    if n == "fcnn":
        return FCNN()  # LazyLinear inside FCNN will infer features
    if n == "cnn":
        return CNN(in_channels=in_ch)
    if n == "enhancedcnn":
        return EnhancedCNN(in_channels=in_ch)
    if n == "cnn64":
        return CNN64()  # fixed 3-channel model for 64x64
    if n == "speccnn":
        return SpecCNN()
    if n == "gan":          
        return TConvGAN(latent_dim=100) 
    if n == "diffusion":
        return DiffusionMLP(img_size=28)
    if n in ("ebm", "energy", "energybased"):
        return EnergyCNN(in_channels=in_ch)


    raise ValueError("Unknown model name. Use 'FCNN' | 'CNN' | 'EnhancedCNN' | 'CNN64'| 'SpecCNN' | 'GAN' | 'Diffusion' | 'EBM'.")

