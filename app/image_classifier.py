# app/image_classifier.py (replace the __init__ with this safer version)
from pathlib import Path
from typing import Literal
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from helper_lib.model import SpecCNN

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

class SimpleImageClassifier:
    def __init__(self, dataset: Literal["cifar10"]="cifar10", device: str="cpu",
                 weights_path: str = "weights/speccnn_cifar10.pt" ):
        self.device = device
        self.transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
        ])
        self.model = SpecCNN().to(self.device)
        p = Path(weights_path)
        if p.exists():
            try:
                state = torch.load(p, map_location=self.device)
                self.model.load_state_dict(state)
                self.model.eval()
                print(f"[INFO] Loaded weights: {p}")
            except Exception as e:
                print(f"[WARN] Failed to load weights ({p}): {e}. Using untrained model.")
        else:
            print(f"[WARN] Weights not found at {p}. Using untrained model.")

    def predict(self, image_path: str) -> dict:
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)[0]
            idx = int(torch.argmax(probs).item())
            conf = float(probs[idx].item())
        return {"class_index": idx, "class_name": CIFAR10_CLASSES[idx], "confidence": round(conf, 4)}
