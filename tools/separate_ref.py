#!/usr/bin/env python3
"""pymss reference separation on the same song for A/B comparison.

Usage: python tools/separate_ref.py --model-dir assets --input assets/cyberangel.wav --outdir separated_ref
Writes six stereo stems as 32-bit float WAVs and prints demix timing.
"""
import argparse
import struct
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch

from bench_ref import load_model, demix


def read_wav_f32(path: Path):
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 2 and w.getsampwidth() == 2
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return x.reshape(-1, 2).T.copy(), sr  # (2, N)


def write_wav_f32(path: Path, x: np.ndarray, sr: int):
    # x: (channels, N) float32 -> interleaved 32-bit float WAV
    data = x.T.astype("<f4").tobytes()
    ch = x.shape[0]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(4)
        w.setframerate(sr)
        w.setcomptype("NONE", "IEEE float")  # format tag 3 via raw? stdlib lacks it
        w.writeframes(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="assets")
    ap.add_argument("--input", required=True)
    ap.add_argument("--reference-root", type=Path)
    ap.add_argument("--outdir", default="separated_ref")
    args = ap.parse_args()
    model_dir = Path(args.model_dir)
    model = load_model(model_dir, args.reference_root)
    x, sr = read_wav_f32(Path(args.input))
    x = torch.from_numpy(x)[None].cuda()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = demix(model, x)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    dur = x.shape[-1] / sr
    from bench_ref import chunk_starts
    print(f"DEMIX total: {dt:.3f} s, RTF {dt / dur:.4f} ({len(chunk_starts(x.shape[-1]))} chunks, {dur:.1f} s song)")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    names = model.num_instruments if hasattr(model, "num_instruments") else None
    stems = result[0].cpu().numpy()  # (6, 2, N)
    for si in range(stems.shape[0]):
        # 32-bit float WAV written manually (stdlib wave lacks format 3)
        xi = stems[si].T.astype("<f4").tobytes()
        n, ch = stems[si].shape[1], 2
        hdr = b"RIFF" + struct.pack("<I", 36 + len(xi)) + b"WAVEfmt " + struct.pack(
            "<IHHIIHH", 16, 3, ch, sr, sr * ch * 4, ch * 4, 32
        ) + b"data" + struct.pack("<I", len(xi))
        (outdir / f"{si}_stem.wav").write_bytes(hdr + xi)
    print(f"wrote {stems.shape[0]} stems to {outdir}")


if __name__ == "__main__":
    sys.exit(main())
