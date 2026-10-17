"""SYNTHETIC noise images for pipeline smoke tests ONLY. Not medical data; models trained on it mean nothing."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
out, n = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 12
rng = np.random.default_rng(0)
for c, name in enumerate(["NORMAL", "PNEUMONIA"]):
    (out / name).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        a = rng.normal(110, 25, (256, 256)); a += np.linspace(0, 40, 256)[None]
        if c: yy, xx = np.ogrid[:256, :256]; a += 80 * np.exp(-((yy - 128) ** 2 + (xx - 90) ** 2) / 1500)
        Image.fromarray(np.clip(a, 0, 255).astype("uint8")).save(out / name / f"person{c}{i // 2}_{i}.png")
