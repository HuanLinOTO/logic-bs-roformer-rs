import struct
from pathlib import Path

import numpy as np

names = ["bass", "drums", "other", "vocals", "guitar", "piano"]
for si in (2, 3):
    b = Path(f"separated/{si}_{names[si]}.wav").read_bytes()
    off = 20 + struct.unpack("<I", b[16:20])[0]
    n = struct.unpack("<I", b[off + 4:off + 8])[0]
    x = np.frombuffer(bytearray(b[off + 8:off + 8 + n]), dtype="<f4")
    i = int(np.argmax(np.abs(x)))
    print(f"stem {si}: max |v| = {x[i]:.1f} at sample {i} (t={i/44100:.2f}s)")
    bad = np.abs(x) > 10
    print(f"  samples >10: {bad.sum()}, range [{int(np.argmax(bad)) if bad.any() else -1}, {int(len(bad) - np.argmax(bad[::-1]) - 1) if bad.any() else -1}]")
    # 分段查
    seg = 44100
    for k in range(0, len(x) // seg, 10):
        v = x[k * seg:(k + 1) * seg]
        if np.abs(v).max() > 10:
            print(f"  t={k}s max {np.abs(v).max():.0f}")
