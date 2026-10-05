import struct
from pathlib import Path

import numpy as np


def read_stem(p):
    b = p.read_bytes()
    off = 20 + struct.unpack("<I", b[16:20])[0]
    n = struct.unpack("<I", b[off + 4:off + 8])[0]
    return np.frombuffer(bytearray(b[off + 8:off + 8 + n]), dtype="<f4").reshape(-1, 2).T.astype(np.float64)


names = ["bass", "drums", "other", "vocals", "guitar", "piano"]
a_dir, b_dir = Path("separated"), Path("separated_ref")
tot = 0.0
rows = []
for si in range(6):
    a = read_stem(a_dir / f"{si}_{names[si]}.wav")
    b = read_stem(b_dir / f"{si}_stem.wav")
    nn = min(a.shape[1], b.shape[1])
    a, b = a[:, :nn], b[:, :nn]
    mask = np.isfinite(a).all(0) & np.isfinite(b).all(0)
    a, b = a[:, mask], b[:, mask]
    tot += (b ** 2).sum()
    rows.append((si, a, b))
print(f"{'stem':8s} {'pymssRMS':>9s} {'errRMS':>9s} {'SNR dB':>7s} {'能量占比':>8s}")
for si, a, b in rows:
    e = a - b
    frac = (b ** 2).sum() / tot
    snr = 10 * np.log10((b ** 2).sum() / ((e ** 2).sum() + 1e-30))
    print(f"{names[si]:8s} {np.sqrt((b**2).mean()):9.5f} {np.sqrt((e**2).mean()):9.5f} {snr:7.2f} {frac*100:7.2f}%")
# 全局
sig = sum((b ** 2).sum() for _, _, b in rows)
noi = sum(((a - b) ** 2).sum() for _, a, b in rows)
print(f"TOTAL energy-weighted SNR: {10 * np.log10(sig / (noi + 1e-30)):.2f} dB")
