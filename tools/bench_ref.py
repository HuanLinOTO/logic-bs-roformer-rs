#!/usr/bin/env python3
"""PyTorch fp32 baseline benchmark for logic_bs_roformer on this machine.

Usage: python tools/bench_ref.py [--model-dir assets] [--full]
Covers: single-chunk forward (588800 samples) and, with --full, the
chunked overlap-add demix path over a 30s signal.
"""
import argparse
import sys
import time
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


def synth(sr: int, dur: float) -> np.ndarray:
    rs = np.random.RandomState(0)
    t = np.arange(int(sr * dur)) / sr
    sig = np.stack([
        0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t),
        0.25 * np.sin(2 * np.pi * 443 * t) + 0.2 * np.sin(2 * np.pi * 1760 * t),
    ]).astype(np.float32)
    sig += rs.randn(2, sig.shape[1]).astype(np.float32) * 0.01
    return sig


def timeit(fn, warmup=3, iters=10):
    torch.cuda.empty_cache()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="assets")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    model_dir = Path(args.model_dir)
    model = load_model(model_dir)
    sr = 44100

    # single chunk
    x3 = torch.from_numpy(synth(sr, 3.0))[None].cuda()
    @torch.inference_mode()
    def fwd3():
        model(x3)
    dt = timeit(fwd3)
    print(f"BENCH single-chunk(3s): {dt*1000:.1f} ms, RTF {dt/3.0:.4f}")

    # verify against the golden reference output
    ref = np.load(model_dir / "ref_output.npz")
    with torch.no_grad():
        y = model(x3)
    out = y.cpu().numpy()
    ref_out = ref["out"]
    err = np.abs(out - ref_out).max()
    denom = np.abs(ref_out).max()
    print(f"golden check: rel err {err/denom:.3e} (must be ~0; ref was this very model)")

    if args.full:
        x30 = torch.from_numpy(synth(sr, 30.0))[None].cuda()
        C = 588800
        step = 559360
        border = 29440
        xp = torch.nn.functional.pad(x30, (border, border * 2), mode="reflect")
        starts = list(range(0, xp.shape[-1] - C + 1, step))
        n_stems = y.shape[1]

        @torch.inference_mode()
        def demix():
            result = torch.zeros(1, n_stems, 2, xp.shape[-1] - border * 2, device="cuda")
            counter = torch.zeros(1, 1, 1, result.shape[-1], device="cuda")
            fade = torch.cat([
                torch.linspace(0, 1, border),
                torch.ones(C - border * 2),
                torch.linspace(1, 0, border),
            ])
            w = fade.cuda()
            for i, s in enumerate(starts):
                wi = w.clone()
                if i == 0:
                    wi[:border] = 1.0
                if i == len(starts) - 1:
                    wi[-border:] = 1.0
                chunk = xp[..., s : s + C]
                out = model(chunk)
                take = min(C, result.shape[-1] - s)
                result[..., s : s + take] += out[..., :take] * wi[:take]
                counter[..., s : s + take] += wi[:take] * wi[:take]
            return result / counter

        dt = timeit(demix, warmup=1, iters=3)
        print(f"BENCH demix(30s): {dt*1000:.1f} ms, RTF {dt/30.0:.4f}")


if __name__ == "__main__":
    sys.exit(main())
