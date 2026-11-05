import numpy as np
from sklearn.metrics import roc_auc_score


def roc_auc_ci(y, p, n_boot=1000, seed=42):
    """AUC with a 95% bootstrap interval. Returns (auc, low, high)."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    auc = float(roc_auc_score(y, p))
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)          # resample with replacement
        yb = y[idx]
        if yb.min() == yb.max():             # skip: only one class present
            continue
        aucs.append(roc_auc_score(yb, p[idx]))
    low, high = np.percentile(aucs, [2.5, 97.5])
    return auc, float(low), float(high)


def threshold_at_specificity(y, p, spec=0.90):
    """Threshold so that about `spec` of negatives score below it."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    neg = np.sort(p[y == 0])
    k = int(np.ceil(spec * len(neg))) - 1
    return float(np.nextafter(neg[k], np.inf))   # next float just above


def sens_spec(y, p, threshold):
    """Predict positive when p >= threshold. Returns (sensitivity, specificity)."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    sens = float(np.mean(p[y == 1] >= threshold))
    spec = float(np.mean(p[y == 0] < threshold))
    return sens, spec


def evaluate_arrays(y, p, threshold, name, n_boot=1000, seed=42,
                    threshold_source="kermany_val"):
    """One row of summary.csv."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    auc, low, high = roc_auc_ci(y, p, n_boot=n_boot, seed=seed)
    sens, spec = sens_spec(y, p, threshold)
    acc = float(np.mean((p >= threshold).astype(int) == y))
    return {
        "dataset": name,
        "n": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "auc": round(auc, 4),
        "auc_ci_low": round(low, 4),
        "auc_ci_high": round(high, 4),
        "threshold": round(float(threshold), 4),
        "threshold_source": threshold_source,
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "accuracy": round(acc, 4),
    }