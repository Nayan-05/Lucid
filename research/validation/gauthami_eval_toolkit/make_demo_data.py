import os

import numpy as np
from PIL import Image


def write_demo(folder, n_per_class=100, seed=42):
    """Synthetic grey squares (not X-rays). Brightness depends a little on the label."""
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(seed)
    out = {}
    for split in ("val", "test"):
        records = []
        for label in (0, 1):
            for i in range(n_per_class):
                mean = rng.normal(110 if label == 0 else 145, 18)
                img = rng.normal(mean, 10, (64, 64))
                img = np.clip(img, 0, 255).astype(np.uint8)
                path = os.path.join(folder, f"{split}_{label}_{i:03d}.png")
                Image.fromarray(img).save(path)
                records.append({
                    "path": path,
                    "label": label,
                    "patient": f"{split}_{label}_{i}",
                    "source": "demo",
                    "view": None,
                    "age": None,
                })
        out[split] = records
    return out
