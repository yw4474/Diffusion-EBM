# scripts/test_ebm.py
import torch.nn as nn
import torch.optim as optim

from helper_lib.utils import set_seed, get_device
from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_ebm
from helper_lib.generator import generate_ebm_samples


def main():
    set_seed(42)
    device = get_device()
    print("device:", device)

    # 1) MNIST dataloader（注意要和 Diffusion 一样，把像素缩放到 [-1,1]）
    train_loader = get_data_loader(
        data_dir="data",
        batch_size=64,
        train=True,
        dataset="mnist",
    )
    print("num train batches:", len(train_loader))

    # 2) 拿到 EBM 模型
    model = get_model("EBM", dataset="mnist")
    print("model class:", model.__class__.__name__)

    # 3) 优化器（EBM 一般 lr 会略小一点）
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 4) 训练 1 个 epoch 试一下
    model = train_ebm(
        model=model,
        data_loader=train_loader,
        optimizer=optimizer,
        device=device,
        epochs=1,
        steps_per_batch=10,   # 先少一点，加快测试
        step_size=0.1,
        noise_scale=0.01,
    )

    # 5) 生成几张图片看看效果
    generate_ebm_samples(
        model=model,
        device=device,
        num_samples=9,
        steps=40,
        step_size=0.1,
        noise_scale=0.01,
    )


if __name__ == "__main__":
    main()
