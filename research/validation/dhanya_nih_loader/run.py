import argparse
import json
import subprocess
from datetime import date

import pandas as pd

from .nih import map_nih, build_image_index
from .make_demo_data import make_demo_df, make_demo_index
from research.common.results import write_results


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv")
    p.add_argument("--root")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--out", default="records_nih.json")
    p.add_argument("--max-age", type=int, default=None)
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()

    if args.demo:
        df = make_demo_df()
        image_index = make_demo_index()
    else:
        df = pd.read_csv(args.csv)
        image_index = build_image_index(args.root)

    recs, counts = map_nih(df, image_index, args.n, args.n, 42, args.max_age)

    if not args.demo:
        with open(args.out, "w") as f:
            json.dump(recs, f, indent=2)

        children = args.max_age is not None
        meta = {
            "author": "bashma",
            "date": date.today().isoformat(),
            "git_commit": git_commit(),
            "seed": 42,
            "checkpoint": "none (loader)",
            "dataset": "nih_children" if children else "nih",
            "n": len(recs),
        }
        name = "counts_children.json" if children else "counts.json"
        write_results(f"results/validation/bashma/{name}", meta, counts)

    print(json.dumps(counts))


if __name__ == "__main__":
    main()