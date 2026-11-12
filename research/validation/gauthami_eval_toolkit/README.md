# Evaluation toolkit with confidence intervals

One-line problem: I built a toolkit that evaluates any binary classifier on any records file and reports AUC with a 95% bootstrap interval, sensitivity and specificity at a fixed threshold, to measure how much a pneumonia model degrades on other hospitals' data.

## Why it matters

A single AUC from a few hundred images is uncertain; without intervals, differences between datasets may be noise.

## Inputs and outputs

- Input: a records JSON file (path, label, ...) and a predict function (probabilities, column 1 = positive)
- Output: summary.csv, per_image.csv, results.json (meta block), roc.png
- Privacy: per_image.csv lists image names and patient IDs. Do not share or commit it for datasets with restricted licences (CheXpert, NIH); share only summary.csv, results.json and roc.png.

## Quick start

    export PYTHONPATH="backend:."          # Windows: $env:PYTHONPATH="backend;."
    python -m research.validation.gauthami_eval_toolkit.run --demo

The demo uses fake grey squares and a stub model and finishes in under a minute.

## Reproduce the main result

    python -m research.validation.gauthami_eval_toolkit.run --records kermany_test --name kermany_test --threshold-from kermany_val

To evaluate another dataset with the same fixed threshold:

    python -m research.validation.gauthami_eval_toolkit.run --records PATH_TO_RECORDS.json --name DATASET_NAME --threshold 0.017392893988712622 --out-dir OUT_DIR

To merge result folders from several dataset owners into one table and one ROC plot:

    python -m research.validation.gauthami_eval_toolkit.combine --inputs FOLDER1 FOLDER2   # each folder needs summary.csv only

## Result

Seed 42, 1,000 bootstrap resamples. Threshold chosen once on the Kermany validation split (n = 874) at 90% specificity.

| dataset | n | n_pos | n_neg | AUC (95% CI) | threshold | threshold source | sensitivity | specificity | accuracy |
|---|---|---|---|---|---|---|---|---|---|
| demo_dataset (fake data) | 200 | 100 | 100 | 0.9294 (0.8919-0.9593) | 0.5224 | demo_val | 0.74 | 0.89 | 0.815 |
| demo_dataset (fake data) | 200 | 100 | 100 | 0.9294 (0.8919-0.9593) | 0.5000 | fixed_0.5 | 0.83 | 0.86 | 0.845 |
| kermany_test | 874 | 650 | 224 | 0.9973 (0.9947-0.9993) | 0.0174 | kermany_val | 0.9985 | 0.8616 | 0.9634 |
| kermany_test | 874 | 650 | 224 | 0.9973 (0.9947-0.9993) | 0.5000 | fixed_0.5 | 0.9815 | 0.9777 | 0.9805 |
| chexpert | 1000 | 500 | 500 | 0.7598 (0.7260-0.7908) | 0.0174 | kermany_val | 0.944 | 0.218 | 0.581 |
| chexpert | 1000 | 500 | 500 | 0.7598 (0.7260-0.7908) | 0.5000 | fixed_0.5 | 0.842 | 0.446 | 0.644 |
| nih | 1000 | 500 | 500 | 0.7854 (0.7580-0.8156) | 0.0174 | kermany_val | 0.902 | 0.408 | 0.655 |
| nih | 1000 | 500 | 500 | 0.7854 (0.7580-0.8156) | 0.5000 | fixed_0.5 | 0.700 | 0.728 | 0.714 |
| nih_children | 152 | 76 | 76 | 0.7273 (0.6473-0.8050) | 0.0174 | kermany_val | 0.8684 | 0.3684 | 0.6184 |
| nih_children | 152 | 76 | 76 | 0.7273 (0.6473-0.8050) | 0.5000 | fixed_0.5 | 0.6447 | 0.6579 | 0.6513 |

Reading it: on its own hospital's test children the model ranks pneumonia above normal almost perfectly (AUC 0.9973). At the validation threshold, specificity on the test split is 0.86 rather than 0.90 because the threshold was set on a different set of children. Nayan independently recomputed the Kermany AUC and got 0.9973.

On every external dataset the AUC falls well below Kermany: CheXpert 0.760, NIH 0.785, NIH children 0.727 (the children set is small, n = 152, so its interval is wide). The external intervals do not overlap Kermany's, so the drop is not chance. At the Kermany threshold specificity is only 0.22 to 0.41, so the model flags many normal images. This measures the drop; it does not explain it. The NIH rows were run with the rounded threshold 0.0174 and CheXpert with 0.017392893988712622.

## How to read the numbers (for non-experts)

- **AUC**: the chance that a randomly chosen pneumonia image gets a higher score than a randomly chosen normal image. 0.5 is guessing, 1.0 is perfect ranking, and it does not depend on a threshold.
- **Sensitivity**: of the images that truly show pneumonia, the share the model flagged.
- **Specificity**: of the truly normal images, the share the model left alone.
- **95% interval**: if we drew a different sample of images, the AUC would very likely fall in this range. If two datasets' intervals do not overlap, the difference is unlikely to be noise.

## Threshold

Chosen once on the Kermany validation split at 90% specificity and reused unchanged on every dataset, because a deployed model cannot be re-tuned on each new hospital's patients.

## Method

1. Every dataset is turned into the same records list (path, label, patient, source, view, age).
2. The model predicts a probability of pneumonia (column 1 of predict_proba) for each image.
3. The threshold is the next float above the k-th sorted normal score on Kermany validation, where k = ceil(0.90 x number of normals) - 1. Positive means probability >= threshold.
4. AUC is computed with scikit-learn.
5. The interval comes from 1,000 bootstrap resamples of the images (seed 42), skipping resamples with only one class, then taking the 2.5th and 97.5th percentiles.
6. A second row at threshold 0.5 is reported for comparison.
7. summary.csv and per_image.csv are upserted, so a rerun replaces rows instead of duplicating them.
8. results.json carries a meta block (group, author, date, git commit, seed, checkpoint, dataset, n).

## Limitations

- Small n: a few hundred images per external dataset gives wide intervals, and the Kermany test split is not balanced (650 pneumonia, 224 normal).
- The bootstrap resamples images, not patients, so it can be slightly too optimistic if one patient has several images.
- Labels on the external datasets (CheXpert, NIH) are noisy: they are extracted from radiology reports by software, not confirmed by a clinician.
- The model was trained on children from one hospital, so any drop on adult datasets mixes domain shift with age and label-method differences; this toolkit measures the drop, it does not explain it.
- One threshold was fixed on one validation set, so sensitivity and specificity on other datasets reflect that choice.

## Files

- `metrics.py`: AUC with bootstrap interval, threshold rule, sensitivity and specificity, one summary row
- `evaluate.py`: runs a predict function over records, builds the per-image table
- `run.py`: the one-command entry point (demo, Kermany, any records file)
- `combine.py`: merges the summary.csv files of several result folders into one table and an AUC-with-interval plot (it never reads per-image data)
- `make_demo_data.py`: fake grey squares for the demo and tests
- `tests/`: metric tests and combine test

## Credits and references

- Group A, Lucid project. Toolkit by Gauthami; shared code and review by Nayan.
- Zech et al. 2018, https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002683
- scikit-learn roc_auc_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

Research and educational use only. Not a clinical or diagnostic tool.

## Note on the threshold
The Kermany-val threshold is 0.017392893988712622. The NIH and NIH children rows were run with the rounded value 0.0174 by the dataset owner. The difference is negligible, and the results were not rerun.
