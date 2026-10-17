import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, pytest
ROOT = Path(__file__).resolve().parents[2]

def test_split_has_no_patient_leakage():
    d = tempfile.mkdtemp()
    sh = lambda *c: subprocess.run([sys.executable, *c], cwd=ROOT, check=True, capture_output=True)
    sh("scripts/generate_sample_data.py", f"{d}/raw", "20")
    sh("scripts/prepare_dataset.py", "--raw", f"{d}/raw", "--out", f"{d}/out")
    s = json.loads(Path(f"{d}/out/splits.json").read_text())
    pid = lambda k: {Path(i["path"]).stem.split("_")[0] for i in s[k]}
    assert not (pid("train") & pid("test")) and not (pid("train") & pid("val")) and not (pid("val") & pid("test"))

def test_gradcam():
    torch = pytest.importorskip("torch")
    from app.ml.common import build_model
    from app.services.explainability_service import gradcam_overlay
    m = build_model("resnet18", 2, False).eval()
    out = gradcam_overlay(m, torch.randn(1, 3, 224, 224), 0, np.random.randint(0, 255, (300, 300), dtype=np.uint8))
    assert out.startswith("data:image/png;base64,")
