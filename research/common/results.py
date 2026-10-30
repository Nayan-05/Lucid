"""write_results(path, meta, data): every JSON result file goes through this (Appendix C).

{"meta": {"project": "lucid", "group": "A", "author": ..., "date": ..., "git_commit": ...,
          "seed": 42, "checkpoint": ..., "dataset": ..., "n": ...},
 "results": data}
"""
import json
import math
from pathlib import Path

import numpy as np

REQUIRED = ("author", "date", "git_commit", "seed", "checkpoint", "dataset", "n")


def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_results(path, meta, data):
    missing = [k for k in REQUIRED if k not in meta]
    if missing:
        raise ValueError(f"meta is missing {missing} (see rule book Appendix C)")
    m = {"project": "lucid"}
    m.update(meta)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"meta": _plain(m), "results": _plain(data)}, indent=1))
    return str(path)
