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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pymss_core.modules.bs_roformer.bs_roformer import BSRoformer  # noqa: E402

import yaml  # noqa: E402


def load_model(model_dir: Path) -> BSRoformer:
    with open(model_dir / "logic_bs_roformer.yaml") as f:
        cfg = yaml.unsafe_load(f)
    mcfg = dict(cfg["model"])
    for k in [
        "multi_stft_resolution_loss_weight",
        "multi_stft_resolutions_window_sizes",
        "multi_stft_hop_size",
        "multi_stft_normalized",
        "linear_transformer_depth",
        "use_torch_checkpoint",
        "dim_freqs_in",
    ]:
        mcfg.pop(k, None)
    model = BSRoformer(**mcfg)
    from safetensors.torch import load_file
    sd = load_file(str(model_dir / "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected[:5]
    model.eval().cuda()
    return model


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
    ap.add_argument("--outdir", default="separated_ref")
    args = ap.parse_args()
    model_dir = Path(args.model_dir)
    model = load_model(model_dir)
    x, sr = read_wav_f32(Path(args.input))
    x = torch.from_numpy(x)[None].cuda()
    C = 588800
    step = 559360
    border = 29440
    # pymss parity: keep every step-grid start that the output span needs and
    # widen the reflect padding so the final chunk is full-width (the old
    # range(0, xp_len - C + 1, step) grid dropped the trailing partial chunk).
    needed = border + x.shape[-1]
    starts = []
    s = 0
    while s + C < needed:
        starts.append(s)
        s += step
    starts.append(s)
    right = starts[-1] + C - needed
    xp = torch.nn.functional.pad(x, (border, right), mode="reflect")
    fade = torch.cat([
        torch.linspace(0, 1, border),
        torch.ones(C - border * 2),
        torch.linspace(1, 0, border),
    ]).cuda()
    result = torch.zeros(1, 6, 2, needed, device="cuda")
    counter = torch.zeros(1, 1, 1, result.shape[-1], device="cuda")
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i, s in enumerate(starts):
            wi = fade.clone()
            if i == 0:
                wi[:border] = 1.0
            if i == len(starts) - 1:
                wi[-border:] = 1.0
            chunk = xp[..., s:s + C]
            out = model(chunk)
            take = min(C, result.shape[-1] - s)
            result[..., s:s + take] += out[..., :take] * wi[:take]
            counter[..., s:s + take] += (wi * wi)[:take]
        result = result / counter
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    dur = x.shape[-1] / sr
    print(f"DEMIX total: {dt:.3f} s, RTF {dt / dur:.4f} ({len(starts)} chunks, {dur:.1f} s song)")
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
