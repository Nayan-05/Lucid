# Pipeline
1. `scripts/prepare_dataset.py` – drop corrupt/duplicate files, patient-level split (pid from filename regex), writes `data/processed/{splits,stats}.json`.
2. `PYTHONPATH=backend python scripts/train_classifier.py` – ResNet18/50 transfer learning, class-weighted loss, early stopping, checkpoint + history + plots.
3. `PYTHONPATH=backend python scripts/evaluate_model.py` – test-split metrics -> `models/metrics.json`.
4. API loads `models/classifier.pt` automatically; without it the UI stays in labelled DEMO mode.
Privacy: DICOM tags are read via an allowlist; DB stores only hash/modality/size/scores/prediction.
Note: `generate_sample_data.py` makes synthetic noise for smoke tests only; never report its metrics.
