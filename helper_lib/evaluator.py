# helper_lib/evaluator.py
import torch
import torch.nn as nn

def evaluate_model(model: torch.nn.Module, data_loader, criterion: nn.Module, device: str = "cpu"):
    model.eval()
    model.to(device)
    losssum = 0.0
    correct = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            losssum += criterion(logits, y).item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
    avg_loss = losssum / len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)
    return {"loss": avg_loss, "acc": acc}
