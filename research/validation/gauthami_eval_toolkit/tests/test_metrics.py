import numpy as np

from research.validation.gauthami_eval_toolkit.metrics import (
    roc_auc_ci,
    sens_spec,
    threshold_at_specificity,
)


def test_auc_perfect_and_random():
    y = np.array([0] * 50 + [1] * 50)
    assert roc_auc_ci(y, np.linspace(0, 1, 100), n_boot=200)[0] == 1.0
    rng = np.random.default_rng(42)
    y2 = rng.integers(0, 2, 5000)
    auc = roc_auc_ci(y2, rng.random(5000), n_boot=50)[0]
    assert abs(auc - 0.5) < 0.03


def test_ci_same_seed_and_contains_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.3 + rng.random(300) * 0.7, 0, 1)
    a = roc_auc_ci(y, p, 300, seed=42)
    assert a == roc_auc_ci(y, p, 300, seed=42)
    assert a[1] <= a[0] <= a[2]


def test_threshold_gives_requested_specificity():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = rng.random(500)
    t = threshold_at_specificity(y, p, 0.90)
    assert sens_spec(y, p, t)[1] >= 0.90


def test_evaluate_with_stub(tmp_path):
    from research.common.stubs import stub_predict_proba  # needs the shared code
    from research.validation.gauthami_eval_toolkit.evaluate import evaluate
    from research.validation.gauthami_eval_toolkit.make_demo_data import write_demo

    demo = write_demo(str(tmp_path), n_per_class=30)
    from PIL import Image

    read = lambda path: np.array(Image.open(path).convert("L"))
    row = evaluate(demo["test"], stub_predict_proba, 0.5, n_boot=100, name="demo", read_fn=read)
    assert row["n"] == 60 and 0.5 < row["auc"] <= 1.0
