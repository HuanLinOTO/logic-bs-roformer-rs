import struct
from pathlib import Path

import numpy as np

ref = np.load("assets/chunk0_ref.npy")
for si in (0, 1, 2):
    b = Path(f"separated0/{si}_{['bass','drums','other','vocals','guitar','piano'][si]}.wav").read_bytes()
    off = 20 + struct.unpack("<I", b[16:20])[0]
    n = struct.unpack("<I", b[off + 4:off + 8])[0]
    x = np.frombuffer(b[off + 8:off + 8 + n], dtype="<f4").reshape(-1, 2).T.astype(np.float64)
    r = ref[si].astype(np.float64)
    nn = min(x.shape[1], r.shape[1])
    a, rr = x[0, :nn], r[0, :nn]
    a = (a - a.mean()) / (a.std() + 1e-12)
    rr = (rr - rr.mean()) / (rr.std() + 1e-12)
    # FFT 互相关找全局最优滞后
    N = 1 << 23
    cc = np.fft.irfft(np.fft.rfft(a, N) * np.conj(np.fft.rfft(rr, N)), N)
    k = int(np.argmax(cc))
    lag = k if k < N // 2 else k - N
    print(f"stem {si}: best lag = {lag} samples ({lag / 44100:.4f} s), corr = {cc[k] / nn:.4f}")
