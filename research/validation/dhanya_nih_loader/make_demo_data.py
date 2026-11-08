import pandas as pd


def make_demo_df():
    rows = [
        ("img1.png", "Pneumonia|Infiltration", 1, 45, "PA"),
        ("img2.png", "No Finding", 2, 30, "AP"),
        ("img3.png", "Pneumothorax", 3, 60, "PA"),
        ("img4.png", "Pneumonia", 4, 52, "AP"),
        ("img5.png", "Pneumonia", 5, 414, "PA"),
        ("img6.png", "Infiltration", 6, 40, "PA"),
        ("img7.png", "No Finding", 7, 25, "AP"),
        ("img8.png", "No Finding", 8, 70, "PA"),
    ]
    return pd.DataFrame(rows, columns=["Image Index", "Finding Labels", "Patient ID", "Patient Age", "View Position"])


def make_demo_index():
    return {f"img{i}.png": f"demo/img{i}.png" for i in range(1, 9)}
