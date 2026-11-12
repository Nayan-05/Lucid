import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .run import upsert_csv


def draw_auc_comparison(summary, path):
    """One point per dataset: AUC with its 95% bootstrap interval, and n in the label."""
    main = summary[summary["threshold_source"] != "fixed_0.5"].drop_duplicates("dataset")
    fig, ax = plt.subplots(figsize=(7, 1.2 + 0.7 * len(main)))
    ypos = list(range(len(main)))
    for y, (_, r) in zip(ypos, main.iterrows()):
        ax.errorbar(r["auc"], y, xerr=[[r["auc"] - r["auc_ci_low"]], [r["auc_ci_high"] - r["auc"]]],
                    fmt="o", capsize=4)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{r['dataset']} (n={int(r['n'])})" for _, r in main.iterrows()])
    ax.invert_yaxis()
    ax.axvline(0.5, color="k", linestyle="--", linewidth=1)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("AUC (95% bootstrap interval); dashed line = guessing")
    ax.set_title("AUC by dataset, same Kermany threshold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Combine summary.csv files into one table and AUC plot.")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="folders that each contain summary.csv")
    ap.add_argument("--out-dir", default="results/validation/gauthami")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "summary.csv")
    for folder in args.inputs:
        s = pd.read_csv(os.path.join(folder, "summary.csv"))
        upsert_csv(summary_path, s)
        print(f"merged {folder}: {sorted(s['dataset'].unique())}")

    summary = pd.read_csv(summary_path)
    draw_auc_comparison(summary, os.path.join(args.out_dir, "auc_comparison.png"))
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    main()
