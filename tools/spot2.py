import numpy as np
import torch as th
from safetensors.torch import load_file

sd = load_file("assets/model.safetensors")
xb = np.fromfile("x_xb.bin", dtype="<f4").reshape(62, 1151, 256)
glu = np.fromfile("x_glu.bin", dtype="<f4").reshape(6, 62, 1151, 516)

# band0: dim_in = 2*freqs*2? 参考键 .0.2.weight (16, 1024) → 输出 16 = 2*8, glu 后 8
for s in range(6):
    w1 = sd[f"mask_estimators.{s}.to_freqs.0.0.0.weight"].numpy()
    b1 = sd[f"mask_estimators.{s}.to_freqs.0.0.0.bias"].numpy()
    w2 = sd[f"mask_estimators.{s}.to_freqs.0.0.2.weight"].numpy()
    b2 = sd[f"mask_estimators.{s}.to_freqs.0.0.2.bias"].numpy()
    h1 = np.tanh(xb[0, 0] @ w1.T + b1)
    pre = h1 @ w2.T + b2  # (16,)
    half = pre.shape[0] // 2
    exp_glu = pre[:half] * (1.0 / (1.0 + np.exp(-pre[half:])))
    got = glu[s, 0, 0, :half]
    print(f"stem {s} band0 t0:")
    print(f"  pre   {np.round(pre, 3)}")
    print(f"  exp   {np.round(exp_glu, 4)}")
    print(f"  got   {np.round(got, 4)}")
