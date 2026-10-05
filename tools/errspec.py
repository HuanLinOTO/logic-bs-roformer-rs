import struct
from io import BytesIO
from pathlib import Path

import numpy as np

ref = np.load("assets/chunk0_ref.npy")
names = ["bass", "drums", "other", "vocals", "guitar", "piano"]
for si in (0, 3):
    b = Path(f"separated0/{si}_{names[si]}.wav").read_bytes()
    off = 20 + struct.unpack("<I", b[16:20])[0]
    n = struct.unpack("<I", b[off + 4:off + 8])[0]
    x = np.frombuffer(bytearray(b[off + 8:off + 8 + n]), dtype="<f4").reshape(-1, 2).T.astype(np.float64)
    r = ref[si].astype(np.float64)
    nn = min(x.shape[1], r.shape[1])
    err = x[0, :nn] - r[0, :nn]
    # STFT 误差幅度（分段 FFT，粗粒度）
    seg = 8192
    nseg = nn // seg
    print(f"stem {si} ({names[si]}) 误差能谱（行=时间段，列=频带 0..15）：")
    for k in range(0, nseg, max(1, nseg // 12)):
        e = err[k * seg:(k + 1) * seg]
        sp = np.abs(np.fft.rfft(e)) ** 2
        bands = np.array_split(sp, 16)
        row = [f"{bb.sum():8.1e}" for bb in bands]
        print(f"  t={k * seg / 44100:5.1f}s  " + " ".join(row[:8]))
