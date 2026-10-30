"""Tests for the shared research code. No model file and no X-ray images needed."""
import json

import numpy as np
import pytest
from PIL import Image

from research.common.records import kermany_records, load_records_json, read_gray, save_records_json
from research.common.results import write_results
from research.common.stubs import stub_predict_proba


def test_kermany_records_match_frozen_split():
    sizes = {s: len(kermany_records(s)) for s in ("train", "val", "test")}
    assert sizes == {"train": 4076, "val": 874, "test": 874}
    r = kermany_records("test")[0]
    assert set(r) == {"path", "label", "patient", "source", "view", "age"}
    assert "\\" not in r["path"] and r["source"] == "kermany" and r["label"] in (0, 1)


def test_kermany_patients_do_not_cross_splits():
    pats = {s: {r["patient"] for r in kermany_records(s)} for s in ("train", "val", "test")}
    assert not (pats["train"] & pats["test"]) and not (pats["val"] & pats["test"])


def test_records_round_trip(tmp_path):
    recs = [{"path": "a\\b.png", "label": np.int64(1), "patient": "p1", "source": "nih",
             "view": "PA", "age": np.int64(40), "extra": "kept"}]
    p = save_records_json(recs, tmp_path / "r.json")
    back = load_records_json(p)
    assert back == [{"path": "a/b.png", "label": 1, "patient": "p1", "source": "nih",
                     "view": "PA", "age": 40, "extra": "kept"}]


def test_records_reject_bad_label(tmp_path):
    with pytest.raises(ValueError):
        save_records_json([{"path": "x", "label": -1, "patient": "p", "source": "nih"}], tmp_path / "r.json")


def test_read_gray_8_and_16_bit(tmp_path):
    a = (np.arange(64 * 64) % 256).astype(np.uint8).reshape(64, 64)
    Image.fromarray(a).save(tmp_path / "g.png")
    Image.fromarray(np.stack([a] * 3, -1)).save(tmp_path / "c.jpg", quality=100)
    Image.fromarray((a.astype(np.uint16) * 257)).save(tmp_path / "w.png")
    for name in ("g.png", "c.jpg", "w.png"):
        g = read_gray(tmp_path / name)
        assert g.dtype == np.uint8 and g.shape == (64, 64)
    assert np.array_equal(read_gray(tmp_path / "g.png"), a)


def test_stub_is_brightness_and_deterministic():
    imgs = [np.full((8, 8), v, np.uint8) for v in (0, 51, 255)]
    p = stub_predict_proba(imgs)
    assert p.shape == (3, 2)
    assert np.allclose(p[:, 1], [0.0, 0.2, 1.0]) and np.allclose(p.sum(1), 1)
    assert np.array_equal(p, stub_predict_proba(imgs))


def test_write_results_meta(tmp_path):
    meta = {"group": "A", "author": "x", "date": "2026-10-30", "git_commit": "abc1234", "seed": 42,
            "checkpoint": "none", "dataset": "demo", "n": np.int64(3)}
    p = write_results(tmp_path / "sub" / "r.json", meta, {"auc": np.float64(0.9), "nan": float("nan")})
    d = json.loads(open(p).read())
    assert d["meta"]["project"] == "lucid" and d["meta"]["n"] == 3
    assert d["results"] == {"auc": 0.9, "nan": None}
    with pytest.raises(ValueError):
        write_results(tmp_path / "bad.json", {"author": "x"}, {})
