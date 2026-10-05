import numpy as np
import torch as th
from safetensors.torch import load_file

sd = load_file("assets/model.safetensors")
xb = np.fromfile("x_xb.bin", dtype="<f4").reshape(62, 1151, 256)
h = np.fromfile("x_hidden.bin", dtype="<u4").reshape(6 * 62, 1151, 512)

for s in range(6):
    w1 = sd[f"mask_estimators.{s}.to_freqs.0.0.0.weight"].numpy()
    b1 = sd[f"mask_estimators.{s}.to_freqs.0.0.0.bias"].numpy()
    exp = np.tanh(xb[0, 0] @ w1.T + b1)[:4]
    words = h[s * 62 + 0, 0, :2]
    got = []
    for wd in words:
        lo = th.tensor((wd & 0xFFFF).astype(np.uint16)).view(th.float16).item()
        hi = th.tensor((wd // 65536).astype(np.uint16)).view(th.float16).item()
        got.extend([lo, hi])
    print(f"stem {s}: exp {np.round(exp, 4)}")
    print(f"        got {np.round(got, 4)}")
