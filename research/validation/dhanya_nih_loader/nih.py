import os
import pandas as pd


def has_finding(cell, name):
    """Exact match on the pipe-separated list. 'Pneumonia' must not match 'Pneumothorax'."""
    return name in str(cell).split("|")


def build_image_index(image_root):
    """Walk the image folders once and return {file name: full path}."""
    index = {}
    for dirpath, _, filenames in os.walk(image_root):
        for f in filenames:
            index[f] = os.path.join(dirpath, f)
    return index


def map_nih(df, image_index, n_pos=500, n_neg=500, seed=42, max_age=None):
    counts = {"rows": len(df)}

    bad_age = df["Patient Age"] > 100
    counts["impossible_ages_dropped"] = int(bad_age.sum())
    df = df[~bad_age].copy()

    if max_age is not None:
        df = df[df["Patient Age"] <= max_age]

    is_pos = df["Finding Labels"].apply(lambda c: has_finding(c, "Pneumonia"))
    is_neg = df["Finding Labels"] == "No Finding"

    pos = df[is_pos].drop_duplicates("Patient ID")
    neg = df[is_neg & ~df["Patient ID"].isin(pos["Patient ID"])].drop_duplicates("Patient ID")
    counts.update(positives_available=len(pos), negatives_available=len(neg))

    n = min(n_pos, n_neg, len(pos), len(neg))
    pos = pos.sample(n, random_state=seed)
    neg = neg.sample(n, random_state=seed)
    counts.update(final_per_class=n, final_total=2 * n)

    recs = []
    for label, part in ((1, pos), (0, neg)):
        for _, r in part.iterrows():
            recs.append({
                "path": image_index.get(r["Image Index"], r["Image Index"]),
                "label": label,
                "patient": str(r["Patient ID"]),
                "source": "nih",
                "view": r["View Position"],
                "age": int(r["Patient Age"]),
            })
    return recs, counts


def load_nih(csv_path, image_root, n_pos=500, n_neg=500, seed=42, max_age=None):
    df = pd.read_csv(csv_path)
    image_index = build_image_index(image_root)
    recs, _ = map_nih(df, image_index, n_pos, n_neg, seed, max_age)
    return recs