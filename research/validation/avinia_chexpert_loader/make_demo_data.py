from pathlib import Path

import pandas as pd


def demo_tables():
    # Each row is:
    # patient, view, AP/PA, age, Pneumonia, No Finding
    blank = float("nan")

    rows = [
        ("P01", "Frontal", "PA", 41, 1.0, blank),
        ("P02", "Frontal", "AP", 67, 1.0, blank),
        ("P03", "Frontal", "AP", 55, 1.0, blank),
        ("P04", "Frontal", "PA", 30, 1.0, blank),
        ("P04", "Frontal", "PA", 30, 1.0, blank),
        ("P05", "Lateral", blank, 50, 1.0, blank),
        ("P06", "Frontal", "PA", 22, blank, 1.0),
        ("P07", "Frontal", "PA", 35, 0.0, 1.0),
        ("P08", "Frontal", "AP", 61, blank, 1.0),
        ("P09", "Frontal", "PA", 44, blank, 1.0),
        ("P10", "Frontal", "PA", 29, blank, 1.0),
        ("P11", "Frontal", "AP", 73, blank, 1.0),
        ("P12", "Frontal", "AP", 80, -1.0, blank),
        ("P13", "Frontal", "PA", 58, blank, blank),
        ("P14", "Frontal", "PA", 47, -1.0, 1.0),
        ("P01", "Frontal", "PA", 41, 0.0, 1.0),
    ]

    meta_rows = []
    label_rows = []

    for row_number, row in enumerate(rows, start=1):
        patient, view, ap_pa, age, pneumonia, no_finding = row
        patient_number = int(patient[1:])

        path = (
            f"train/patient{patient_number:02d}/"
            f"study{row_number}/view1_{view.lower()}.jpg"
        )

        meta_rows.append(
            {
                "path_to_image": path,
                "frontal_lateral": view,
                "ap_pa": ap_pa,
                "deid_patient_id": patient,
                "age": float(age),
            }
        )
        label_rows.append(
            {
                "path_to_image": path,
                "Pneumonia": pneumonia,
                "No Finding": no_finding,
            }
        )

    meta = pd.DataFrame(meta_rows)
    labels = pd.DataFrame(label_rows)
    return meta, labels


def write_demo(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    meta, labels = demo_tables()

    csv_path = folder / "demo_meta.csv"
    json_path = folder / "demo_labels.json"

    meta.to_csv(csv_path, index=False)
    labels.to_json(json_path, orient="records", lines=True)

    return str(csv_path), str(json_path)

