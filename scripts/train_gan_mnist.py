from helper_lib.utils import set_seed, get_device
from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from pathlib import Path
import torchvision.transforms as T
import torch

def main():
    set_seed(42)
    device = get_device()
    print("Device:", device)

    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])  # [-1,1]
    train_dl = get_data_loader(dataset="mnist", train=True, batch_size=128, transform=tf)

    gan = get_model("gan")  # your TConvGAN via factory
    gan = train_gan(gan, train_dl, device=device, epochs=3, latent_dim=100)

    Path("weights").mkdir(exist_ok=True)
    torch.save(gan.generator.state_dict(), "weights/gan_mnist_gen.pt")
    print("Saved generator -> weights/gan_mnist_gen.pt")

if __name__ == "__main__":
    main()
