import struct
from pathlib import Path

import numpy as np

ref = np.load("assets/chunk0_ref.npy")
for si in (0, 1, 4):
    b = Path(f"separated0/{si}_{['bass','drums','other','vocals','guitar','piano'][si]}.wav").read_bytes()
    off = 20 + struct.unpack("<I", b[16:20])[0]
    n = struct.unpack("<I", b[off + 4:off + 8])[0]
    x = np.frombuffer(b[off + 8:off + 8 + n], dtype="<f4").reshape(-1, 2).T
    r = ref[si].astype(np.float64)
    nn = min(x.shape[1], r.shape[1])
    a = x[:, :nn].astype(np.float64)
    # 每 10 段（每段 ~1.8s）的相关
    segs = 10
    sl = nn // segs
    cors = []
    for k in range(segs):
        s0 = k * sl
        aa, rr = a[0, s0:s0 + sl], r[0, s0:s0 + sl]
        cors.append(np.corrcoef(aa, rr)[0, 1])
    print(f"stem {si}: seg corr L:", " ".join(f"{c:+.2f}" for c in cors))
    # 好坏样本幅值比
    good = (np.abs(r) > 0.05 * np.abs(r).max())
    print(f"  rms ratio got/ref (active): {np.sqrt((a[:, good] ** 2).mean() / (r[:, good] ** 2).mean()):.3f}")
