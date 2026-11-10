# NIH ChestX-ray14 loader (Bashma)

Turns the NIH ChestX-ray14 metadata into clean, balanced records (one image per patient, equal positives and negatives) with an optional children-only subset. Code folder: `research/validation/dhanya_nih_loader/` (the folder keeps the rule book's original name).

## Label rule

A row is **positive** if the pipe-separated `Finding Labels` cell contains the exact label `Pneumonia`.
A row is **negative** only if the cell is exactly `No Finding`. Anything else is skipped.

Worked example: `Pneumonia|Infiltration` is positive, because `Pneumonia` is one of its parts. `Pneumothorax` is not positive, because it does not match `Pneumonia`. `Infiltration` alone is neither positive nor negative.

Other rules: ages above 100 are dropped and counted; one image per patient; patients with any pneumonia image are removed from the negatives; sampling uses seed 42.

## Counts after each filter step (real file)

| Step | Count |
| --- | --- |
| Rows read | 112,120 |
| Impossible ages (> 100) dropped | 0 |
| Patients | 30,805 |
| Pneumonia images | 1,431 |
| No Finding images | 60,361 |
| Positives available (one image per patient) | 1,008 |
| Negatives available (one per patient, no pneumonia patients) | 24,044 |
| Final sample: per class / total | 500 / 1,000 |
| Children (age 18 or under): positives / negatives available | 76 / 1,510 |
| Children final sample: per class / total | 76 / 152 |

## Warning

NIH labels are text-mined from radiology reports and are noisy. Treat them as weak labels, not clinical ground truth. Lucid is research and educational software only, not a diagnostic tool.

## How to run (Windows PowerShell, from the repo folder, venv active)

Tests (expect 9 passed):
```
python -m pytest research/validation/dhanya_nih_loader/tests -v
```

Demo on made-up data (rows 8, impossible ages 1, positives 2, negatives 3, final 2 per class / 4 total):
```
python -m research.validation.dhanya_nih_loader.run --demo
```

Real run and children subset (records stay private; counts go to `results/validation/bashma/`):
```
$csv = "$HOME\LUCID\datasets\nih\Data_Entry_2017_v2020.csv"
python -m research.validation.dhanya_nih_loader.run --csv $csv --root "$HOME\LUCID\datasets\nih\images" --n 500 --out "$HOME\LUCID\private\records_nih.json"
python -m research.validation.dhanya_nih_loader.run --csv $csv --root "$HOME\LUCID\datasets\nih\images" --n 500 --max-age 18 --out "$HOME\LUCID\private\records_nih_children.json"
```

Outputs: `results/validation/bashma/counts.json` and `counts_children.json` (each with a meta block). The records files and the CSV are never committed.
