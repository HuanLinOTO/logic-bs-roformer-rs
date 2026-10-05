#!/usr/bin/env python3
"""Compare two separation output dirs (32-bit float WAVs) per stem."""
import struct
import sys
from pathlib import Path
import numpy as np


def read_f32_wav(p: Path):
    b = p.read_bytes()
    assert b[:4] == b"RIFF" and b[8:16] == b"WAVEfmt "
    fmt_size = struct.unpack("<I", b[16:20])[0]
    tag, ch, sr, _, _, bits = struct.unpack("<HHIIHH", b[20:36])
    assert tag == 3 and bits == 32, (tag, bits)
    off = 20 + fmt_size
    while b[off:off + 4] != b"data":
        off += 8 + struct.unpack("<I", b[off + 4:off + 8])[0]
    n = struct.unpack("<I", b[off + 4:off + 8])[0]
    x = np.frombuffer(b[off + 8:off + 8 + n], dtype="<f4")
    return x.reshape(-1, ch).T, sr


def main():
    a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
    for i in range(6):
        fa = sorted(a_dir.glob(f"{i}_*.wav"))
        fb = sorted(b_dir.glob(f"{i}_*.wav"))
        assert fa and fb, (i, fa, fb)
        a, _ = read_f32_wav(fa[0])
        b, _ = read_f32_wav(fb[0])
        n = min(a.shape[1], b.shape[1])
        a, b = a[:, :n].astype(np.float64), b[:, :n].astype(np.float64)
        nan_frac = np.mean(np.isnan(b))
        # mask positions where either side is non-finite
        ok = np.isfinite(a).all(0) & np.isfinite(b).all(0)
        a, b = a[:, ok], b[:, ok]
        sig = np.sum(a ** 2)
        noise = np.sum((a - b) ** 2)
        snr = 10 * np.log10(sig / (noise + 1e-30))
        corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
        print(f"stem {i}: SNR(us vs pymss) = {snr:6.2f} dB, corr = {corr:.5f}, "
              f"peak = {np.abs(a).max():.3f}/{np.abs(b).max():.3f}, pymss-nan {nan_frac*100:.2f}%")


if __name__ == "__main__":
    main()
