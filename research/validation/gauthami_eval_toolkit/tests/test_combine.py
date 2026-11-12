import numpy as np
import pandas as pd

from research.validation.gauthami_eval_toolkit.combine import main
from research.validation.gauthami_eval_toolkit.metrics import evaluate_arrays


def _fake_folder(base, name, seed):
    rng = np.random.default_rng(seed)
    y = np.array([0] * 30 + [1] * 30)
    p = np.clip(np.where(y == 1, rng.normal(0.6, 0.2, 60), rng.normal(0.3, 0.2, 60)), 0, 1)
    rows = [evaluate_arrays(y, p, 0.4, name, n_boot=50),
            evaluate_arrays(y, p, 0.5, name, n_boot=50, threshold_source="fixed_0.5")]
    folder = base / name
    folder.mkdir()
    pd.DataFrame(rows).to_csv(folder / "summary.csv", index=False)
    return str(folder)


def test_combine_merges_without_per_image_and_does_not_duplicate(tmp_path):
    a = _fake_folder(tmp_path, "aaa", 1)
    b = _fake_folder(tmp_path, "bbb", 2)
    out = tmp_path / "out"
    main(["--inputs", a, b, "--out-dir", str(out)])
    main(["--inputs", a, b, "--out-dir", str(out)])  # rerun: no duplicates
    s = pd.read_csv(out / "summary.csv")
    assert sorted(s["dataset"].unique()) == ["aaa", "bbb"]
    assert len(s) == 4
    assert (out / "auc_comparison.png").exists()
    assert not (out / "per_image.csv").exists()
