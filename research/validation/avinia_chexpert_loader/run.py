
import argparse
from datetime import date
import json
from pathlib import Path
import subprocess
import tempfile

import pandas as pd

from research.common.records import save_records_json
from research.common.results import write_results
from .chexpert import map_chexpert, read_labels
from .make_demo_data import write_demo


def git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run_loader(csv_path, labels_path, root, n, ext, out_path,
               counts_path, dataset):
    meta_table = pd.read_csv(csv_path)
    labels_table = read_labels(labels_path)

    seed = 42
    records, counts = map_chexpert(
        meta_table,
        labels_table,
        root,
        n,
        n,
        seed,
        ext,
    )

    result_meta = {
        "group": "A",
        "author": "avinia",
        "date": date.today().isoformat(),
        "git_commit": git_commit(),
        "seed": seed,
        "checkpoint": "none (loader)",
        "dataset": dataset,
        "n": len(records),
        "labels_file": Path(labels_path).name,
    }

    save_records_json(records, out_path)
    write_results(counts_path, result_meta, counts)

    print(f"seed = {seed}")
    print(json.dumps(counts, indent=2))
    print(f"records written to: {out_path}")
    print(f"counts written to: {counts_path}")

    return counts


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv")
    parser.add_argument("--labels")
    parser.add_argument("--root")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--ext")
    parser.add_argument("--out", default="records_chexpert.json")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args(argv)

    if args.demo:
        with tempfile.TemporaryDirectory(prefix="chexpert_demo_") as temp:
            temp_folder = Path(temp)
            csv_path, labels_path = write_demo(temp_folder / "demo_data")

            return run_loader(
                csv_path,
                labels_path,
                "demo_images",
                args.n,
                args.ext,
                temp_folder / "records_chexpert.json",
                temp_folder / "counts.json",
                "chexpert_demo",
            )

    if not args.csv or not args.labels or not args.root:
        parser.error("without --demo, --csv, --labels and --root are required")

    return run_loader(
        args.csv,
        args.labels,
        args.root,
        args.n,
        args.ext,
        args.out,
        "results/validation/avinia/counts.json",
        "chexpert_plus",
    )


if __name__ == "__main__":
    main()