import struct
import numpy as np
def load(p):
    b = open(p, "rb").read()
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(b):
        cid = b[pos:pos+4]; (sz,) = struct.unpack("<I", b[pos+4:pos+8])
        body = b[pos+8:pos+8+sz]
        if cid == b"fmt ": tag, ch, rate = struct.unpack("<HHI", body[:8]); fmt = (tag, ch, rate)
        elif cid == b"data": data = body
        pos += 8 + sz + (sz & 1)
    tag, ch, rate = fmt
    if tag == 3: x = np.frombuffer(data, dtype="<f4").astype(np.float64)
    else: x = np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768.0
    return x.reshape(-1, ch)
base = "/data/dsh/logic-bs-roformer-rs/separated_user/"
tot = None
for s in ["0_bass","1_drums","2_other","3_vocals","4_guitar","5_piano"]:
    x = load(base + s + ".wav")
    tot = x if tot is None else tot + x
src = load("/data/dsh/logic-bs-roformer-rs/user_cyberangel.wav")
print("frames:", len(tot), len(src))
m = min(len(tot), len(src))
d = tot[:m] - src[:m]
e = np.sum(src[:m]**2); en = np.sum(d**2)
print("6-stem recon SNR vs source:", round(10*np.log10(e/max(en,1e-30)), 2), "dB")
g = np.sum(tot[:m]*src[:m]) / np.sum(tot[:m]**2)
print("best-fit gain:", round(float(g), 4), " sum peak:", round(float(np.abs(tot).max()), 3))
