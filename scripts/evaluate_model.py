"""PYTHONPATH=backend python scripts/evaluate_model.py -> models/metrics.json (held-out test split only)"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from app.ml.common import SplitDS, build_model, eval_tf

ap = argparse.ArgumentParser()
ap.add_argument("--splits", default="data/processed/splits.json"); ap.add_argument("--ckpt", default="models/classifier.pt")
ap.add_argument("--out", default="models/metrics.json"); a = ap.parse_args()
ck = torch.load(a.ckpt, map_location="cpu"); S = json.loads(Path(a.splits).read_text())
m = build_model(ck["arch"], len(ck["classes"]), False); m.load_state_dict(ck["state"]); m.eval()
ys, ps = [], []
with torch.no_grad():
    for x, y in DataLoader(SplitDS(S["test"], eval_tf(ck["img_size"])), 32):
        ps.append(torch.softmax(m(x), 1).numpy()); ys += y.tolist()
P, y = np.concatenate(ps), np.array(ys); pred = P.argmax(1); C = ck["classes"]
res = {"n_test": len(y), "classes": C, "test_accuracy": accuracy_score(y, pred),
       "macro_f1": f1_score(y, pred, average="macro"), "confusion_matrix": confusion_matrix(y, pred).tolist(),
       "report": classification_report(y, pred, target_names=C, output_dict=True, zero_division=0),
       "model": {k: ck[k] for k in ("arch", "version", "trained_at", "preprocessing_version")},
       "limitations": "Single held-out split; no external validation; results are research-grade only."}
try:
    if len(C) == 2:
        res["roc_auc"] = roc_auc_score(y, P[:, 1]); f, t, _ = roc_curve(y, P[:, 1])
        res["roc_curve"] = {"fpr": f.tolist(), "tpr": t.tolist()}
    else: res["roc_auc"] = roc_auc_score(y, P, multi_class="ovr")
except ValueError: res["roc_auc"] = None  # e.g. only one class present in test split
Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps({k: res[k] for k in ("test_accuracy", "macro_f1", "roc_auc")}))
