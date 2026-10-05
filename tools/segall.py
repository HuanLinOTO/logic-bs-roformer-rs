import struct
from pathlib import Path

import numpy as np

ref = np.load("assets/chunk0_ref.npy")
names = ["bass", "drums", "other", "vocals", "guitar", "piano"]
for si in range(6):
    b = Path(f"separated0/{si}_{names[si]}.wav").read_bytes()
    off = 20 + struct.unpack("<I", b[16:20])[0]
    n = struct.unpack("<I", b[off + 4:off + 8])[0]
    x = np.frombuffer(b[off + 8:off + 8 + n], dtype="<f4").reshape(-1, 2).T.astype(np.float64)
    r = ref[si].astype(np.float64)
    nn = min(x.shape[1], r.shape[1])
    a = x[0, :nn]
    rr = r[0, :nn]
    segs = 16
    sl = nn // segs
    cors = []
    for k in range(segs):
        aa, rseg = a[k * sl:(k + 1) * sl], rr[k * sl:(k + 1) * sl]
        cors.append(np.corrcoef(aa, rseg)[0, 1] if aa.std() > 1e-9 else 0.0)
    print(f"stem {si} ({names[si]:7s}):", " ".join(f"{c:+.2f}" for c in cors))
