"""Scan data/raw/<class>/* -> dedupe, drop corrupt, patient-level split, stats.
Patient id comes from --pid-regex on the filename stem (default fits 'person123_...'); no match => per-image group."""
import argparse, hashlib, json, random, re
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/raw"); ap.add_argument("--out", default="data/processed")
ap.add_argument("--pid-regex", default=r"^(person\d+)"); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--val", type=float, default=0.15); ap.add_argument("--test", type=float, default=0.15)
a = ap.parse_args()
raw, out = Path(a.raw), Path(a.out); out.mkdir(parents=True, exist_ok=True)
classes = sorted(d.name for d in raw.iterdir() if d.is_dir())
items, bad, dups, seen, dims, matched = [], [], 0, set(), Counter(), 0
for ci, c in enumerate(classes):
    for p in sorted((raw / c).rglob("*")):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}: continue
        try:
            with Image.open(p) as im: im.verify()
            with Image.open(p) as im: size = im.size
        except Exception: bad.append(str(p)); continue
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h in seen: dups += 1; continue
        seen.add(h); dims[f"{size[0]}x{size[1]}"] += 1
        m = re.match(a.pid_regex, p.stem); matched += bool(m)
        items.append({"path": str(p), "label": ci, "group": m.group(1) if m else h})
groups = defaultdict(list)
for it in items: groups[it["group"]].append(it)
keys = sorted(groups); random.Random(a.seed).shuffle(keys)
splits, n, cut = {"train": [], "val": [], "test": []}, len(items), {"test": a.test * len(items), "val": (a.test + a.val) * len(items)}
for k in keys:  # whole patient goes to exactly one split
    s = "test" if len(splits["test"]) < cut["test"] else "val" if len(splits["val"]) + len(splits["test"]) < cut["val"] else "train"
    splits[s] += groups[k]
cnt = Counter(it["label"] for it in items)
stats = {"total": n, "bad_files": len(bad), "duplicates_removed": dups, "classes": classes,
         "patient_level_split": matched > 0, "class_distribution": {classes[k]: v for k, v in cnt.items()},
         "imbalance_ratio": round(max(cnt.values()) / max(min(cnt.values()), 1), 2) if cnt else None,
         "top_dims": dict(dims.most_common(5)),
         "splits": {s: {"n": len(v), "classes": {classes[k]: c for k, c in Counter(i["label"] for i in v).items()}} for s, v in splits.items()}}
(out / "splits.json").write_text(json.dumps({"classes": classes, **{s: [{"path": i["path"], "label": i["label"]} for i in v] for s, v in splits.items()}}))
(out / "stats.json").write_text(json.dumps(stats, indent=1))
print(json.dumps(stats, indent=1))
