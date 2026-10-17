import cv2, numpy as np
from .imaging import png64

def gradcam_overlay(model, x, cls: int, gray: np.ndarray) -> str:
    """Grad-CAM on the last residual block. Model attribution only, not clinical evidence."""
    import torch  # lazy: API must still run in DEMO mode without torch
    acts, grads = [], []
    layer = model.layer4[-1]
    def fwd(m, i, o):
        acts.append(o)
        o.register_hook(lambda g: grads.append(g))
    h = layer.register_forward_hook(fwd)
    model.zero_grad()
    try:
        model(x)[0, cls].backward()
    finally:
        h.remove()
    w = grads[0].mean((2, 3), keepdim=True)
    cam = torch.relu((w * acts[0]).sum(1))[0].detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (gray.shape[1], gray.shape[0]))
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return png64(cv2.addWeighted(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.55, heat, 0.45, 0))
