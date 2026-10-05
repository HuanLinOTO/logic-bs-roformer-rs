import struct
from pathlib import Path

import numpy as np

ref = np.load("assets/chunk0_demixref.npy")
names = ["bass", "drums", "other", "vocals", "guitar", "piano"]
for si in range(6):
    b = Path(f"separated0/{si}_{names[si]}.wav").read_bytes()
    off = 20 + struct.unpack("<I", b[16:20])[0]
    n = struct.unpack("<I", b[off + 4:off + 8])[0]
    x = np.frombuffer(bytearray(b[off + 8:off + 8 + n]), dtype="<f4").reshape(-1, 2).T.astype(np.float64)
    r = ref[si].astype(np.float64)
    nn = min(x.shape[1], r.shape[1])
    e = x[:, :nn] - r[:, :nn]
    rr = r[:, :nn]
    snr = 10 * np.log10((rr ** 2).sum() / ((e ** 2).sum() + 1e-30))
    print(f"stem {si} ({names[si]:7s}): SNR {snr:6.2f} dB refRMS {np.sqrt((rr**2).mean()):.5f}")
