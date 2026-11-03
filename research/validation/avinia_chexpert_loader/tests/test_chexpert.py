
from research.validation.avinia_chexpert_loader.chexpert import map_chexpert
from research.validation.avinia_chexpert_loader.make_demo_data import demo_tables


def run(seed=42, n=500):
    meta, labels = demo_tables()
    return map_chexpert(meta, labels, "root", n, n, seed)


def test_label_mapping():
    recs, _ = run()
    by_patient = {record["patient"]: record["label"] for record in recs}

    assert by_patient["P01"] == 1
    assert by_patient["P06"] == 0
    assert "P12" not in by_patient
    assert "P13" not in by_patient
    assert "P14" not in by_patient


def test_lateral_dropped():
    recs, counts = run()

    assert counts["frontal"] == counts["rows"] - 1
    assert all("lateral" not in record["path"] for record in recs)


def test_one_image_per_patient():
    recs, _ = run()
    patient_ids = [record["patient"] for record in recs]

    assert len(patient_ids) == len(set(patient_ids))


def test_balanced_and_counts():
    recs, counts = run()

    assert counts == {
        "rows": 16,
        "frontal": 15,
        "frontal_with_labels": 15,
        "uncertain_pneumonia_dropped": 2,
        "not_pos_not_neg_dropped": 3,
        "positives_available": 4,
        "negatives_available": 6,
        "final_per_class": 4,
        "final_total": 8,
    }
    assert sum(record["label"] for record in recs) == 4


def test_fewer_positives_than_requested():
    _, counts = run(n=500)

    assert counts["final_per_class"] == 4


def test_seed():
    first = [record["patient"] for record in run(seed=42, n=3)[0]]
    second = [record["patient"] for record in run(seed=42, n=3)[0]]
    other_seed = [record["patient"] for record in run(seed=7, n=3)[0]]

    assert first == second
    assert first != other_seed


def test_record_format():
    rec = run()[0][0]

    assert set(rec) == {"path", "label", "patient", "source", "view", "age"}
    assert rec["source"] == "chexpert"
    assert isinstance(rec["age"], int)