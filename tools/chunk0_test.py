#!/usr/bin/env python3
"""Single-chunk A/B: build the padded chunk-0 input, run pymss once, save ref."""
import struct
import sys
import wave
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from separate_ref import load_model  # noqa: E402


def read_wav_i16(path):
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def write_wav_i16(path, x_int, sr):
    xi = x_int.T.astype("<i2").tobytes()
    n, ch = x_int.shape[1], 2
    hdr = b"RIFF" + struct.pack("<I", 36 + len(xi)) + b"WAVEfmt " + struct.pack(
        "<IHHIIHH", 16, 1, ch, sr, sr * ch * 2, ch * 2, 16
    ) + b"data" + struct.pack("<I", len(xi))
    Path(path).write_bytes(hdr + xi)


def main():
    x = read_wav_i16("assets/cyberangel.wav").reshape(-1, 2).T  # (2, N)
    border = 29440
    xp = np.pad(x, ((0, 0), (border, border * 2)), mode="reflect")
    chunk = xp[:, :588800].copy()
    write_wav_i16("assets/chunk0.wav", (np.clip(chunk, -1, 1) * 32767).astype(np.int16), 44100)
    model = load_model(Path("assets"))
    with torch.inference_mode():
        out = model(torch.from_numpy(chunk)[None].cuda())[0].cpu().numpy()  # (6,2,588800)
    np.save("assets/chunk0_ref.npy", out)
    print("ref saved", out.shape, "peak", np.abs(out).max())


if __name__ == "__main__":
    main()
