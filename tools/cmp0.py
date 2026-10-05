import struct
from pathlib import Path

import numpy as np

ref = np.load("assets/chunk0_ref.npy")
tot_sig = 0.0
tot_noise = 0.0
for si in range(6):
    b = Path(f"separated0/{si}_{['bass','drums','other','vocals','guitar','piano'][si]}.wav").read_bytes()
    off = 20 + struct.unpack("<I", b[16:20])[0]
    n = struct.unpack("<I", b[off + 4:off + 8])[0]
    x = np.frombuffer(b[off + 8:off + 8 + n], dtype="<f4").reshape(-1, 2).T
    n = min(x.shape[1], ref.shape[2])
    a, r = x[:, :n].astype(np.float64), ref[si, :, :n].astype(np.float64)
    sig = (r ** 2).sum()
    noise = ((a - r) ** 2).sum()
    tot_sig += sig
    tot_noise += noise
    print(f"stem {si}: SNR {10 * np.log10(sig / (noise + 1e-30)):6.2f} dB corr {np.corrcoef(a.ravel(), r.ravel())[0, 1]:.5f}")
print(f"TOTAL SNR: {10 * np.log10(tot_sig / (tot_noise + 1e-30)):.2f} dB")
