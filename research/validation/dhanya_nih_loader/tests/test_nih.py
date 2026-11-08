import pandas as pd
from research.validation.dhanya_nih_loader.nih import has_finding, map_nih
from research.validation.dhanya_nih_loader.make_demo_data import make_demo_df, make_demo_index


def _df(rows):
    return pd.DataFrame(rows, columns=["Image Index", "Finding Labels", "Patient ID", "Patient Age", "View Position"])


def test_pneumonia_in_combined_label_is_positive():
    assert has_finding("Pneumonia|Infiltration", "Pneumonia")


def test_infiltration_alone_is_not_pneumonia():
    assert not has_finding("Infiltration", "Pneumonia")


def test_no_finding_is_not_positive():
    assert not has_finding("No Finding", "Pneumonia")


def test_pneumothorax_does_not_match_pneumonia():
    assert not has_finding("Pneumothorax", "Pneumonia")


def test_ages_above_100_are_dropped_and_counted():
    df = _df([
        ("a.png", "No Finding", 1, 150, "PA"),
        ("b.png", "No Finding", 2, 30, "PA"),
    ])
    _, counts = map_nih(df, {}, 10, 10, 42, None)
    assert counts["rows"] == 2
    assert counts["impossible_ages_dropped"] == 1


def test_one_image_per_patient():
    df = _df([
        ("a.png", "Pneumonia", 1, 40, "PA"),
        ("b.png", "Pneumonia", 1, 40, "AP"),
        ("c.png", "No Finding", 2, 30, "PA"),
    ])
    _, counts = map_nih(df, {}, 10, 10, 42, None)
    assert counts["positives_available"] == 1


def test_max_age_keeps_only_18_and_below():
    df = _df([
        ("a.png", "Pneumonia", 1, 17, "PA"),
        ("b.png", "Pneumonia", 2, 40, "PA"),
        ("c.png", "No Finding", 3, 18, "AP"),
        ("d.png", "No Finding", 4, 60, "AP"),
    ])
    recs, _ = map_nih(df, {}, 10, 10, 42, 18)
    assert all(r["age"] <= 18 for r in recs)


def test_seeded_sampling_is_repeatable():
    rows = [(f"p{i}.png", "Pneumonia", i, 40, "PA") for i in range(20)]
    rows += [(f"n{i}.png", "No Finding", 100 + i, 40, "AP") for i in range(20)]
    df = _df(rows)
    a, _ = map_nih(df, {}, 5, 5, 42, None)
    b, _ = map_nih(df, {}, 5, 5, 42, None)
    assert [r["path"] for r in a] == [r["path"] for r in b]


def test_demo_counts_match_expected():
    _, counts = map_nih(make_demo_df(), make_demo_index(), 500, 500, 42, None)
    assert counts == {
        "rows": 8,
        "impossible_ages_dropped": 1,
        "positives_available": 2,
        "negatives_available": 3,
        "final_per_class": 2,
        "final_total": 4,
    }