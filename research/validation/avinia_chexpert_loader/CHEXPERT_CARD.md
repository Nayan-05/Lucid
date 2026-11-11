# CheXpert Plus dataset card (Lucid Group A)

## Source
- **Dataset:** CheXpert Plus (Stanford AIMI) — https://aimi.stanford.edu/datasets/chexpert-plus
- **Paper:** Chambon et al. (2024) — https://arxiv.org/abs/2405.19538
- **Licence:** Stanford University Dataset Research Use Agreement; research use only, no redistribution.
- **Downloaded by:** Avinia. CSV dated 20 October 2026; JSON dated 1 November 2026.
- **Files used:** `df_chexpert_plus_240401.csv` (404.94 MB) and `impression_fixed.json` (83.78 MB).
- **Not downloaded:** DICOM and PNG tables. Approximately 1,000 sampled images are planned for later use.
- **Storage:** Downloader's laptop only. Dataset files must not be committed or shared.

## What the data looks like
Metadata CSV: 223,462 rows; 64,725 distinct patients.
View counts: Frontal 191,071; Lateral 32,391.
ap_pa counts: AP 161,622; PA 29,432; blank 32,391; LL 16; RL 1.
Labels JSON: 223,462 rows. Relevant label columns: Pneumonia and No Finding.
Pneumonia counts: blank 195,926; -1.0 19,367; 1.0 4,847; 0.0 3,322.
No Finding counts: blank 202,203; 1.0 21,259.
Metadata rows matching a label row: 223,462.

## Label rule
label 1 if Pneumonia = 1.0; label 0 if No Finding = 1.0 and Pneumonia is neither 1.0 nor -1.0; drop everything else (uncertain, blank, and other diseases); keep frontal views only; one image per patient; balanced sample of up to 500 per class with seed 42; labels from impression_fixed.json.


## Counts after every filter step

| Step | Count |
|---|---:|
| All metadata rows | 223,462 |
| Frontal only | 191,071 |
| Frontal with a label row | 191,071 |
| Uncertain Pneumonia rows dropped | 16,594 |
| Neither positive nor negative rows dropped | 171,204 |
| Positive patients available, one image each | 3,073 |
| Negative patients available, one image each | 12,889 |
| Final balanced sample, per class | 500 |
| Final balanced sample, total | 1,000 |

The two dropped-count measures overlap; do not add them together. The available positive and negative counts are deduplicated patient counts.

## Image download

- **Method:** Redivis Python package (`redivis`) with browser sign-in; no API token was used.
- **Tables:** `PNG_train` and `PNG_valid`.
- **Selection:** 1,000 images listed in `needed_images.txt`; `.jpg` paths were mapped to `.png`.
- **Download date:** 2026-11-08.
- **Local destination:** `datasets/chexpert/images/`, preserving split, patient, and study folders.
- **Verification:** 1,000 images found; 0 not found.
