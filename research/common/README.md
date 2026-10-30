# research/common: shared code for Group A

Added by Nayan (Group A lead) in place of the architect's starter commit (rule book R4 and Appendix H).
Everyone imports these. Do not edit them in your own branch: if something is wrong, tell Nayan.

| Function | File | What it does |
|---|---|---|
| predict_proba(images) | backend/app/ml/predict.py | list of uint8 grayscale arrays -> [N, 2] probabilities, column 1 = PNEUMONIA |
| predict_logits(images) | backend/app/ml/predict.py | same input -> [N, 2] raw scores before softmax |
| extract_features(images) | backend/app/ml/predict.py | same input -> [N, 512] vectors before the final layer |
| kermany_records(split) | records.py | "train", "val" or "test" from the frozen splits.json, as records |
| read_gray(path) | records.py | image file -> uint8 grayscale array |
| save_records_json(records, path), load_records_json(path) | records.py | write / read a records list (local only, never commit) |
| stub_predict_proba(images) | stubs.py | STUB for demo and tests: P(PNEUMONIA) = mean brightness / 255 |
| write_results(path, meta, data) | results.py | JSON result file with the meta block (Appendix C) |
| apply_style() | plot_style.py | minimal figure style, 150 dpi |

Tests: `python -m pytest research/common` (7 passed). Model check: `python research/common/check_predict.py`.
