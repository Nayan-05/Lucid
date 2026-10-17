import io, numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app
c = TestClient(app)

def png(arr):
    b = io.BytesIO(); Image.fromarray(arr).save(b, "PNG"); return b.getvalue()

def test_health(): assert c.get("/api/health").json()["status"] == "ok"
def test_ok():
    a = (np.random.rand(256, 256) * 200 + 20).astype("uint8")
    r = c.post("/api/analyze", files={"file": ("x.png", png(a))}); assert r.status_code == 200
    assert 0 <= r.json()["quality"]["score"] <= 100 and r.json()["prediction"]["label"] is None
def test_empty(): assert c.post("/api/analyze", files={"file": ("x.png", png(np.zeros((128, 128), "uint8")))}).status_code == 422
def test_bad_ext(): assert c.post("/api/analyze", files={"file": ("x.exe", b"hi")}).status_code == 422
def test_corrupt(): assert c.post("/api/analyze", files={"file": ("x.dcm", b"junk")}).status_code == 422
