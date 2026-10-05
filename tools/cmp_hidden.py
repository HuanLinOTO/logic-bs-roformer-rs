import numpy as np
from safetensors.torch import load_file

sd = load_file("assets/model.safetensors")
xb = np.fromfile("x_xb.bin", dtype="<f4").reshape(62, 1151, 256).astype(np.float64)
hidden = np.fromfile("x_hidden.bin", dtype="<f4")

# f16x2 词解包（torch 原生）
def words_to_f32(ws):
    import torch as th
    w = ws.astype(np.uint32)
    lo = th.from_numpy((w & 0xFFFF).astype(np.uint16)).view(th.float16).float().numpy()
    hi = th.from_numpy((w // 65536).astype(np.uint16)).view(th.float16).float().numpy()
    return lo, hi

T = 1151
# hidden 布局: (stem*62+band, T, 512 words) -> 解包 (…, T, 1024)
ws = hidden.reshape(6 * 62, T, 512)
wflat = ws.ravel()
lo, hi = words_to_f32(wflat)
h1024 = np.empty((wflat.size * 2,), dtype=np.float32)
h1024[0::2] = lo
h1024[1::2] = hi
h = h1024.reshape(6 * 62, T, 1024)

for s in (0, 1, 2, 3):
    sig = 0.0
    noise = 0.0
    for b in (0, 24, 61):
        w1 = sd[f"mask_estimators.{s}.to_freqs.{b}.0.0.weight"].numpy().astype(np.float64)  # (1024, 256)
        b1 = sd[f"mask_estimators.{s}.to_freqs.{b}.0.0.bias"].numpy().astype(np.float64)
        exp = np.tanh(xb[b] @ w1.T + b1)  # (T, 1024)
        got = h[s * 62 + b].astype(np.float64)
        sig += (exp ** 2).sum()
        noise += ((got - exp) ** 2).sum()
    print(f"stem {s} (bands 0/24/61): hidden SNR {10 * np.log10(sig / (noise + 1e-30)):7.2f} dB")
