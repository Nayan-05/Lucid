import argparse
import datetime
import json
import os
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_curve

from research.common.results import write_results

from .evaluate import per_image_table, predict_records
from .make_demo_data import write_demo
from .metrics import evaluate_arrays, threshold_at_specificity

SEED = 42
N_BOOT = 1000


def read_png(path):
    """Open a PNG as a uint8 grayscale array (demo only)."""
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def load_records(spec):
    """'kermany_val' / 'kermany_test', or the path of a records JSON file."""
    from research.common.records import kermany_records, load_records_json
    if spec.startswith("kermany_") and not os.path.isfile(spec):
        records = kermany_records(spec[len("kermany_"):])
    else:
        records = load_records_json(spec)
    fixed = []
    for r in records:
        r = dict(r)
        r["path"] = str(r["path"]).replace("\\", "/")
        fixed.append(r)
    return fixed


def upsert_csv(path, df, key="dataset"):
    """Replace rows of the datasets in df instead of duplicating them."""
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[~old[key].isin(df[key].unique())]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)
    return df


def draw_roc(per_image, summary, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    main_rows = summary[summary["threshold_source"] != "fixed_0.5"]
    for _, row in main_rows.iterrows():
        sub = per_image[per_image["dataset"] == row["dataset"]]
        fpr, tpr, _ = roc_curve(sub["label"], sub["prob_pneumonia"])
        label = (f"{row['dataset']}: AUC {row['auc']:.3f} "
                 f"({row['auc_ci_low']:.3f}-{row['auc_ci_high']:.3f}), n={int(row['n'])}")
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC curves, 95% bootstrap CI in legend")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate a classifier on a records file.")
    ap.add_argument("--records", help="records JSON file, or kermany_test / kermany_val")
    ap.add_argument("--name", default="dataset", help="dataset name in the table")
    ap.add_argument("--threshold-from", choices=["kermany_val"], default=None)
    ap.add_argument("--threshold", type=float, default=None, help="a fixed threshold")
    ap.add_argument("--out-dir", default="results/validation/gauthami")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args(argv)

    print(f"seed = {SEED}")
    print(f"bootstrap resamples = {N_BOOT}")

    if args.demo:
        from research.common.stubs import stub_predict_proba
        tmp = tempfile.mkdtemp(prefix="lucid_demo_")
        demo = write_demo(os.path.join(tmp, "images"), n_per_class=100)
        predict_fn, read_fn = stub_predict_proba, read_png
        yv, pv = predict_records(demo["val"], predict_fn, read_fn)
        threshold = threshold_at_specificity(yv, pv, 0.90)
        source = "demo_val"
        records, name = demo["test"], "demo_dataset"
        checkpoint = "STUB (stub_predict_proba)"
        out_dir = os.path.join(tmp, "results")
    else:
        if not args.records:
            ap.error("--records is required (or use --demo)")
        from app.ml.predict import predict_proba
        from research.common.records import read_gray
        predict_fn, read_fn = predict_proba, read_gray
        out_dir = args.out_dir
        os.makedirs(out_dir, exist_ok=True)
        name = args.name
        checkpoint = "models/classifier.pt"
        thr_path = os.path.join(out_dir, "threshold.json")
        if args.threshold is not None:
            threshold = float(args.threshold)
            source = "kermany_val (given with --threshold)"
        elif args.threshold_from == "kermany_val":
            if os.path.exists(thr_path):
                with open(thr_path) as f:
                    saved = json.load(f)
                threshold, source = float(saved["threshold"]), saved.get("source", "kermany_val")
            else:
                val = load_records("kermany_val")
                yv, pv = predict_records(val, predict_fn, read_fn)
                threshold = threshold_at_specificity(yv, pv, 0.90)
                source = "kermany_val"
                with open(thr_path, "w") as f:
                    json.dump({"threshold": threshold, "source": source,
                               "spec_target": 0.90, "n_val": len(val)}, f, indent=2)
        else:
            ap.error("give --threshold-from kermany_val or --threshold NUMBER")
        records = load_records(args.records)

    os.makedirs(out_dir, exist_ok=True)
    y, p = predict_records(records, predict_fn, read_fn)
    rows = [
        evaluate_arrays(y, p, threshold, name, n_boot=N_BOOT, seed=SEED, threshold_source=source),
        evaluate_arrays(y, p, 0.5, name, n_boot=N_BOOT, seed=SEED, threshold_source="fixed_0.5"),
    ]
    summary_path = os.path.join(out_dir, "summary.csv")
    per_path = os.path.join(out_dir, "per_image.csv")
    summary_all = upsert_csv(summary_path, pd.DataFrame(rows))
    per_all = upsert_csv(per_path, per_image_table(records, p, name))
    roc_path = os.path.join(out_dir, "roc.png")
    draw_roc(per_all, summary_all, roc_path)

    meta = {"group": "A", "author": "gauthami", "date": datetime.date.today().isoformat(),
            "git_commit": git_commit(), "seed": SEED, "checkpoint": checkpoint,
            "dataset": name, "n": int(len(y))}
    data = {"threshold": threshold, "threshold_source": source, "rows": rows}
    results_path = os.path.join(out_dir, "results.json")
    write_results(results_path, meta, data)

    print(pd.DataFrame(rows).to_string(index=False))
    print("Files written:")
    for pth in (summary_path, per_path, roc_path, results_path):
        print("  ", pth)
    return rows


if __name__ == "__main__":
    main()
