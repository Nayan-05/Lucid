"""Shared model functions for the research groups (rule book R4).

Input for all three: a list of uint8 grayscale arrays (H x W), any size.
Each image gets the same preprocessing as training and the app
(app.ml.common.eval_tf: grayscale to 3 channels, resize, ImageNet normalisation).
Class order: 0 = NORMAL, 1 = PNEUMONIA.

The model file is models/classifier.pt (settings.model_path). It is loaded once.
"""
import numpy as np

from ..config import settings

_cache = {}
BATCH = 32


def _load():
    if "m" in _cache:
        return _cache["m"]
    import torch
    from .common import build_model
    ck = torch.load(settings.model_path, map_location="cpu")
    if list(ck["classes"]) != ["NORMAL", "PNEUMONIA"]:
        raise ValueError(f"unexpected class order {ck['classes']}")
    m = build_model(ck["arch"], len(ck["classes"]), False)
    m.load_state_dict(ck["state"])
    m.eval()
    _cache["m"] = (m, ck)
    return _cache["m"]


def _batches(images, size):
    from PIL import Image
    from .common import eval_tf
    import torch
    tf = eval_tf(size)
    for i in range(0, len(images), BATCH):
        chunk = []
        for a in images[i:i + BATCH]:
            a = np.asarray(a)
            if a.ndim != 2 or a.dtype != np.uint8:
                raise ValueError("each image must be a uint8 grayscale array (H x W)")
            chunk.append(tf(Image.fromarray(a)))
        yield torch.stack(chunk)


def predict_logits(images):
    """[N, 2] raw scores before softmax."""
    import torch
    m, ck = _load()
    out = []
    with torch.no_grad():
        for x in _batches(images, ck["img_size"]):
            out.append(m(x).numpy())
    return np.concatenate(out) if out else np.zeros((0, 2), np.float32)


def predict_proba(images):
    """[N, 2] probabilities. Column 1 is PNEUMONIA."""
    z = predict_logits(images).astype(np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def extract_features(images):
    """[N, 512] for ResNet18: the vector that goes into the final layer."""
    import torch
    m, ck = _load()
    body = torch.nn.Sequential(*list(m.children())[:-1])  # everything except fc
    out = []
    with torch.no_grad():
        for x in _batches(images, ck["img_size"]):
            out.append(torch.flatten(body(x), 1).numpy())
    return np.concatenate(out) if out else np.zeros((0, m.fc.in_features), np.float32)
