# scripts/test_diffusion.py

import torch.nn as nn
import torch.optim as optim

from helper_lib.utils import set_seed, get_device
from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_diffusion
from helper_lib.generator import generate_samples


def main():
    # 1) 固定随机种子 + 设备
    set_seed(42)
    device = get_device()
    print("device:", device)

    # 2) MNIST 数据加载（你之前的 get_data_loader 已支持 mnist）
    train_loader = get_data_loader(
        data_dir="data",
        batch_size=64,
        train=True,
        dataset="mnist",
    )
    print("num train batches:", len(train_loader))

    # 3) 拿到 Diffusion 模型
    model = get_model("Diffusion", dataset="mnist")
    print("model class:", model.__class__.__name__)

    # 4) 定义优化器和损失（MSE 预测噪声）
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 5) 训练 1 个 epoch（步数设的小一点，先确认能跑通）
    model = train_diffusion(
        model=model,
        data_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=1,          # 可以先 1 轮看看
        num_steps=200,     # diffusion 步数也先设少一点
    )

    # 6) 采样几张图看看效果（会弹出一个 matplotlib 窗口）
    generate_samples(
        model=model,
        device=device,
        num_samples=9,
        diffusion_steps=200,
    )


if __name__ == "__main__":
    main()
