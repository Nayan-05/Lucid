"""STUB model for demo mode and tests only. Not a real model.

P(PNEUMONIA) = mean brightness / 255. Deterministic, no model file needed.
"""
import numpy as np

STUB_LABEL = "STUB (stub_predict_proba)"


def stub_predict_proba(images):
    """List of uint8 grayscale arrays -> [N, 2]; column 1 = fake P(PNEUMONIA)."""
    p1 = np.array([float(np.asarray(a, dtype=np.float64).mean()) / 255.0 for a in images])
    p1 = p1.reshape(-1)
    return np.stack([1.0 - p1, p1], axis=1)
