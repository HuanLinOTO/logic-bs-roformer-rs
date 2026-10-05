#!/usr/bin/env python3
# Energy-weighted SNR between two sets of stem wavs (reference, candidate).
# Handles float32 (format 3) and PCM16 RIFF wavs directly (py3.13 wave
# module rejects format 3).
import sys, os, struct
import numpy as np

def load(p):
    b = open(p, "rb").read()
    assert b[0:4] == b"RIFF" and b[8:12] == b"WAVE", p
    pos = 12
    fmt = None
    while pos + 8 <= len(b):
        cid = b[pos:pos+4]
        (sz,) = struct.unpack("<I", b[pos+4:pos+8])
        body = b[pos+8:pos+8+sz]
        if cid == b"fmt ":
            tag, ch, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            fmt = (tag, ch, bits)
        elif cid == b"data":
            assert fmt, p
            tag, ch, bits = fmt
            if tag == 3 or bits == 32:
                raw = np.frombuffer(body, dtype="<f4").astype(np.float64)
            else:
                raw = np.frombuffer(body, dtype="<i2").astype(np.float64)
            return raw.reshape(-1, ch)
        pos += 8 + sz + (sz & 1)
    raise AssertionError("no data chunk: " + p)

ref_dir, cand_dir = sys.argv[1], sys.argv[2]
names = sorted(f for f in os.listdir(ref_dir) if f.endswith(".wav"))
tot_e, tot_n = 0.0, 0.0
for f in names:
    a = load(os.path.join(ref_dir, f))
    b = load(os.path.join(cand_dir, f))
    assert a.shape == b.shape, (f, a.shape, b.shape)
    d = a - b
    e = np.sum(a * a)
    tot_e += e
    tot_n += np.sum(d * d)
    snr = 10 * np.log10(e / max(np.sum(d * d), 1e-30))
    print(f"{f}: SNR {snr:.2f} dB")
print(f"OVERALL: {10 * np.log10(tot_e / max(tot_n, 1e-30)):.2f} dB")
