import struct
from pathlib import Path

import numpy as np

ref = np.load("assets/chunk0_demixref.npy")
b = Path("separated0/2_other.wav").read_bytes()
off = 20 + struct.unpack("<I", b[16:20])[0]
n = struct.unpack("<I", b[off + 4:off + 8])[0]
x = np.frombuffer(bytearray(b[off + 8:off + 8 + n]), dtype="<f4").reshape(-1, 2).T.astype(np.float64)
a, r = x[0], ref[2, 0]
a = (a - a.mean()) / (a.std() + 1e-12)
r = (r - r.mean()) / (r.std() + 1e-12)
N = 1 << 21
cc = np.fft.irfft(np.fft.rfft(a, N) * np.conj(np.fft.rfft(r, N)), N)
k = int(np.argmax(cc))
lag = k if k < N // 2 else k - N
print("best lag:", lag, "corr:", cc[k] / 588800)
