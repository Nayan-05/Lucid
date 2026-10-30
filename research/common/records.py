"""Records: one dict per image (rule book Appendix C).

{"path": "data/raw/NORMAL/IM-0001-0001.jpeg", "label": 0, "patient": "IM-0001",
 "source": "kermany", "view": "PA", "age": 5}
label 0 = NORMAL / negative, 1 = PNEUMONIA / positive.
"""
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]          # the Lucid folder
SPLITS = ROOT / "data" / "processed" / "splits.json"  # frozen file
KEYS = ("path", "label", "patient", "source", "view", "age")
_PID = re.compile(r"^(person\d+|(?:NORMAL2-)?IM-\d+)")  # same rule as prepare_dataset.py


def _fix(path):
    """Windows paths in splits.json use backslashes; use / everywhere."""
    return str(path).replace("\\", "/")


def kermany_records(split):
    """split: "train", "val" or "test" (from the frozen splits.json)."""
    split = {"validation": "val"}.get(split, split)
    s = json.loads(SPLITS.read_text())
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be train, val or test, not {split!r}")
    out = []
    for it in s[split]:
        p = _fix(it["path"])
        name = os.path.basename(p)
        m = _PID.match(name)
        out.append({"path": p, "label": int(it["label"]),
                    "patient": m.group(1) if m else os.path.splitext(name)[0],
                    "source": "kermany", "view": None, "age": None})
    return out


def _resolve(path):
    p = Path(_fix(path))
    if p.exists() or p.is_absolute():
        return p
    return ROOT / p  # relative paths are relative to the Lucid folder


def read_gray(path):
    """Image file -> uint8 grayscale array (H x W). Works for JPEG and PNG (8 or 16 bit)."""
    p = _resolve(path)
    if not p.exists():
        raise FileNotFoundError(f"image not found: {p}")
    with Image.open(p) as im:
        im.load()
        if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            a = np.asarray(im, dtype=np.float64)
            lo, hi = a.min(), a.max()
            a = (a - lo) / (hi - lo) * 255 if hi > lo else np.zeros_like(a)
            return a.astype(np.uint8)
        return np.asarray(im.convert("L"), dtype=np.uint8)


def _check(records):
    if not isinstance(records, list):
        raise ValueError("records must be a list of dicts")
    for i, r in enumerate(records):
        missing = [k for k in ("path", "label", "patient", "source") if k not in r]
        if missing:
            raise ValueError(f"record {i} is missing {missing}")
        if r["label"] not in (0, 1):
            raise ValueError(f"record {i} has label {r['label']!r}; use 0 or 1")


def _plain(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if np.isnan(v) else float(v)
    if isinstance(v, float) and v != v:  # NaN
        return None
    return v


def save_records_json(records, path):
    """Writes a records list (local file: never commit or upload it)."""
    _check(records)
    clean = [{k: _plain(r.get(k)) if k in ("label", "age") else r.get(k) for k in KEYS} for r in records]
    for r, c in zip(records, clean):
        c["label"] = int(c["label"])
        c["path"] = _fix(c["path"])
        for k, v in r.items():  # keep any extra fields the owner added
            if k not in c:
                c[k] = _plain(v)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(clean, indent=1))
    return str(path)


def load_records_json(path):
    records = json.loads(Path(path).read_text())
    _check(records)
    return records
