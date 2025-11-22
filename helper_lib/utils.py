def set_seed(seed: int = 42):
    # placeholder; real version will use random/numpy/torch
    return seed

def get_device() -> str:
    return "cpu"

def save_model(model, path: str = "model.pt"):
    # placeholder; real version will torch.save
    with open(path, "wb") as f:
        f.write(b"placeholder")
