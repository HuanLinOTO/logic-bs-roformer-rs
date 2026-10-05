import struct
from pathlib import Path

import numpy as np

ref = np.load("assets/chunk0_ref.npy")
names = ["bass", "drums", "other", "vocals", "guitar", "piano"]
tot = (ref.astype(np.float64) ** 2).sum()
print(f"{'stem':8s} {'refRMS':>9s} {'errRMS':>9s} {'SNR dB':>7s} {'能量占比':>8s}")
for si in range(6):
    b = Path(f"separated0/{si}_{names[si]}.wav").read_bytes()
    off = 20 + struct.unpack("<I", b[16:20])[0]
    n = struct.unpack("<I", b[off + 4:off + 8])[0]
    x = np.frombuffer(bytearray(b[off + 8:off + 8 + n]), dtype="<f4").reshape(-1, 2).T.astype(np.float64)
    r = ref[si].astype(np.float64)
    nn = min(x.shape[1], r.shape[1])
    e = x[:, :nn] - r[:, :nn]
    rr = r[:, :nn]
    frac = (rr ** 2).sum() / tot
    snr = 10 * np.log10((rr ** 2).sum() / ((e ** 2).sum() + 1e-30))
    print(f"{names[si]:8s} {np.sqrt((rr**2).mean()):9.5f} {np.sqrt((e**2).mean()):9.5f} {snr:7.2f} {frac*100:7.2f}%")
