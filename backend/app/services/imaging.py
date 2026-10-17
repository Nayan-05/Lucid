import io, math
import cv2, numpy as np, pydicom
from PIL import Image
from ..config import settings

ALLOWED = {"png", "jpg", "jpeg", "dcm"}
# allowlist = sanitization: anything not listed (name, ID, address...) is never read out
SAFE_TAGS = ["Modality", "StudyDate", "StudyDescription", "PatientAge", "Rows", "Columns",
             "PixelSpacing", "BitsStored", "PhotometricInterpretation"]

def load(data: bytes, filename: str) -> tuple[np.ndarray, dict]:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED:
        raise ValueError("unsupported file type (use PNG, JPEG or DICOM)")
    try:
        if ext == "dcm":
            ds = pydicom.dcmread(io.BytesIO(data))
            if "PixelData" not in ds:
                raise ValueError("pixel data unavailable")
            a = ds.pixel_array.astype(np.float32)
            if a.ndim != 2:
                raise ValueError("multi-frame/colour DICOM not supported yet")
            lo, hi = float(a.min()), float(a.max())
            if hi == lo:
                raise ValueError("image is empty (constant intensity)")
            a = (a - lo) / (hi - lo) * 255
            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                a = 255 - a
            meta = {t: str(getattr(ds, t)) for t in SAFE_TAGS if hasattr(ds, t)}
            return a.astype(np.uint8), meta
        img = Image.open(io.BytesIO(data)); img.load()
        return np.array(img.convert("L")), {"Format": ext.upper()}
    except ValueError:
        raise
    except Exception:
        raise ValueError("corrupted or invalid file")

def validate(a: np.ndarray) -> list[str]:
    issues, h, w = [], *a.shape
    if min(h, w) < 64: issues.append(f"image too small ({w}x{h})")
    if a.std() < 1: issues.append("image is empty / no contrast")
    if a.mean() < 5 or a.mean() > 250: issues.append("extreme intensity values")
    return issues

def preprocess(a: np.ndarray, clahe: bool) -> np.ndarray:
    out = cv2.resize(a, (settings.img_size,) * 2, interpolation=cv2.INTER_AREA)
    if clahe: out = cv2.createCLAHE(2.0, (8, 8)).apply(out)
    return out  # original is never modified

def _lvl(v, lo, hi, rev=False):
    s = "LOW" if v < lo else "HIGH" if v > hi else "GOOD"
    return {"LOW": "HIGH", "HIGH": "LOW", "GOOD": "GOOD"}[s] if rev else s

def quality(a: np.ndarray) -> dict:
    """Transparent baseline (not a trained model, not clinical QA).
    score = 25*brightness + 25*contrast + 20*sharpness + 15*noise + 15*clipping, each 0..1."""
    f = a.astype(np.float32)
    bright, contrast = float(f.mean()), float(f.std())
    sharp = float(cv2.Laplacian(f, cv2.CV_32F).var())
    noise = float(np.median(np.abs(f - cv2.GaussianBlur(f, (5, 5), 0))) * 1.4826)
    clip = float(((a <= 2) | (a >= 253)).mean() * 100)
    sc = lambda v, lo, hi: float(np.clip((v - lo) / (hi - lo), 0, 1))
    parts = [25 * (1 - min(abs(bright - 128) / 100, 1)), 25 * sc(contrast, 20, 60),
             20 * sc(sharp, 20, 300), 15 * (1 - sc(noise, 2, 12)), 15 * (1 - sc(clip, 1, 20))]
    ind = {"Brightness": (bright, "GOOD" if 60 < bright < 190 else "POOR"),
           "Contrast": (contrast, "GOOD" if contrast > 30 else "LOW"),
           "Sharpness": (sharp, "GOOD" if sharp > 50 else "LOW"),
           "Noise": (noise, "GOOD" if noise < 5 else "MODERATE" if noise < 9 else "HIGH"),
           "Clipping %": (clip, "LOW" if clip < 5 else "HIGH")}
    return {"score": round(sum(parts)),
            "indicators": {k: {"value": round(v, 2), "label": l} for k, (v, l) in ind.items()}}

def stats(a: np.ndarray) -> dict:
    h = np.bincount(a.ravel(), minlength=256) / a.size
    ent = -sum(p * math.log2(p) for p in h if p > 0)
    edges = float((cv2.Canny(a, 100, 200) > 0).mean())
    return {"mean": round(float(a.mean()), 2), "std": round(float(a.std()), 2),
            "entropy": round(ent, 3), "edge_density": round(edges, 4),
            "hist": [round(float(x), 5) for x in h.reshape(32, 8).sum(1)]}

def png64(a) -> str:
    import base64
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", a)[1]).decode()
