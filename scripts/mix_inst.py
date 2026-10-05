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
def write_pcm16(path, x, rate):
    x = np.clip(x, -1.0, 1.0 - 2.0/32768.0)
    pcm = (x * 32768.0).astype("<i2").tobytes()
    n = len(pcm)
    hdr = b"RIFF" + struct.pack("<I", 36 + n) + b"WAVEfmt " + struct.pack("<IHHIIHH",
        16, 1, x.shape[1], rate, rate * x.shape[1] * 2, x.shape[1] * 2, 16) + b"data" + struct.pack("<I", n)
    open(path, "wb").write(hdr + pcm)
base = "/data/dsh/logic-bs-roformer-rs/separated_user/"
acc, rate = None, None
rate = 44100
for s in ["0_bass", "1_drums", "2_other", "4_guitar", "5_piano"]:
    x = load(base + s + ".wav")
    acc = x if acc is None else acc + x
peak = float(np.abs(acc).max())
over = float((np.abs(acc) > 1.0).mean())
print("inst raw peak", round(peak, 3), "frac>1:", f"{over:.5%}")
scale = min(1.0, 0.98 / peak)
print("scale", round(scale, 4))
write_pcm16(base + "inst.wav", acc * scale, rate)
v = load(base + "3_vocals.wav")
vp = float(np.abs(v).max())
vs = min(1.0, 0.98 / vp)
print("vocals peak", round(vp, 3), "scale", round(vs, 4))
write_pcm16(base + "vocals.wav", v * vs, rate)
print("OK", v.shape, rate)
