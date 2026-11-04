import os

import numpy as np
import pandas as pd

from .metrics import evaluate_arrays


def predict_records(records, predict_fn, read_fn, batch=32):
    """Run the classifier over records. Returns (y, p); p is P(pneumonia), column 1."""
    y = np.array([int(r["label"]) for r in records])
    n = len(records)
    chunks = []
    for start in range(0, n, batch):
        images = [read_fn(r["path"]) for r in records[start:start + batch]]
        out = np.asarray(predict_fn(images))
        chunks.append(out[:, 1])
        print(f"  predicted {min(start + batch, n)}/{n}", flush=True)
    p = np.concatenate(chunks).astype(float) if chunks else np.array([], dtype=float)
    return y, p


def evaluate(records, predict_fn, threshold, n_boot=1000, seed=42, name="dataset",
             read_fn=None, threshold_source="kermany_val"):
    """Fixed rule-book interface: returns one summary row (dict)."""
    if read_fn is None:
        from research.common.records import read_gray
        read_fn = read_gray
    y, p = predict_records(records, predict_fn, read_fn)
    return evaluate_arrays(y, p, threshold, name, n_boot=n_boot, seed=seed,
                           threshold_source=threshold_source)


def per_image_table(records, p, name):
    """One row per image, for Iris (failure analysis) and Nayan (independent AUC check)."""
    rows = []
    for r, prob in zip(records, p):
        path = str(r["path"]).replace("\\", "/")
        rows.append({
            "image_id": os.path.basename(path),
            "patient": r.get("patient"),
            "label": int(r["label"]),
            "prob_pneumonia": round(float(prob), 6),
            "source": r.get("source"),
            "dataset": name,
            "view": r.get("view"),
            "age": r.get("age"),
            "path": path,
        })
    return pd.DataFrame(rows)
