# helper_lib/data_loader.py

from typing import Optional
from torch.utils.data import DataLoader

def get_data_loader(
    data_dir: str = "data",
    batch_size: int = 128,
    train: bool = True,
    dataset: str = "mnist",
    transform: Optional[object] = None,
    num_workers: int = 2,
    pin_memory: bool = True,
):
    """
    Generic loader for MNIST / CIFAR-10.
    - If `transform` is provided, it will be used.
    - If not, a sensible default is applied per dataset.
    - For GAN on MNIST, pass Normalize((0.5,), (0.5,)) to map images to [-1, 1].
    """
    from torchvision import datasets, transforms

    ds = dataset.lower()
    if ds == "mnist":
        # Default: ToTensor only (0..1). Override for GAN with [-1,1].
        tf = transform or transforms.Compose([
            transforms.ToTensor()
            # For GAN, call with: transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        ])
        dset = datasets.MNIST(root=data_dir, train=bool(train), download=True, transform=tf)
        return DataLoader(dset, batch_size=batch_size, shuffle=bool(train),
                          num_workers=num_workers, pin_memory=pin_memory)

    elif ds == "cifar10":
        # Default normalization for CIFAR-10 classification pipeline
        tf = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
        dset = datasets.CIFAR10(root=data_dir, train=bool(train), download=True, transform=tf)
        return DataLoader(dset, batch_size=batch_size, shuffle=bool(train),
                          num_workers=num_workers, pin_memory=pin_memory)

    else:
        raise ValueError(f"Unsupported dataset '{dataset}'. Use 'mnist' or 'cifar10'.")
