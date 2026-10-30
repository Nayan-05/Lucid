"""One-time check that predict.py gives the same numbers as scripts/evaluate_model.py.

    python research/common/check_predict.py
Needs models/classifier.pt and the Kermany images in data/raw. Takes a few minutes on a CPU.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from app.ml.predict import extract_features, predict_logits, predict_proba
from research.common.records import kermany_records, read_gray

recs = kermany_records("test")
imgs = [read_gray(r["path"]) for r in recs]
y = np.array([r["label"] for r in recs])
p = predict_proba(imgs)
auc = roc_auc_score(y, p[:, 1])
acc = float(((p[:, 1] >= 0.5) == y).mean())
print(f"n={len(y)}  accuracy={acc:.4f}  auc={auc:.4f}")
print("rows sum to 1:", bool(np.allclose(p.sum(1), 1)))
print("logits shape:", predict_logits(imgs[:4]).shape, " features shape:", extract_features(imgs[:4]).shape)
print("Group A model (Nayan, Oct 19) should give accuracy 0.9805 and auc 0.9973.")
