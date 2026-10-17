import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models, transforms as T
from PIL import Image

PREPROC_VERSION = "1.0"  # grayscale->3ch, resize, ImageNet normalisation (no CLAHE: train == inference)
_N = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def build_model(arch: str, n_classes: int, pretrained: bool = True) -> nn.Module:
    w = "DEFAULT" if pretrained else None
    if arch == "resnet18": m = models.resnet18(weights=w)
    elif arch == "resnet50": m = models.resnet50(weights=w)
    else: raise ValueError(f"unsupported arch {arch}")
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m

def eval_tf(size: int): return T.Compose([T.Grayscale(3), T.Resize((size, size)), T.ToTensor(), _N])
def train_tf(size: int):
    return T.Compose([T.Grayscale(3), T.Resize((size, size)), T.RandomRotation(8),
                      T.RandomAffine(0, translate=(0.05, 0.05)), T.ColorJitter(0.2, 0.2), T.ToTensor(), _N])

class SplitDS(Dataset):
    def __init__(self, items, tf): self.items, self.tf = items, tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]
        return self.tf(Image.open(it["path"])), it["label"]
