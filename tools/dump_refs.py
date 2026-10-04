#!/usr/bin/env python3
"""Generate parity reference npz files with the PyTorch reference path.

Run with .venv2. Each entry writes parity/<name>.npz with fixed-seed inputs
and the PyTorch fp32 outputs, ready for the Rust side to compare against.
"""
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent.parent / "parity"

N_FFT = 2048
HOP = 512
WIN = 2048
CH = 2
CHUNK = 588800
T_FRAMES = 1151  # floor(CHUNK / HOP) + 1


def gen_stft():
    rs = np.random.RandomState(42)
    x = rs.randn(CH, CHUNK).astype(np.float32) * 0.1
    window = torch.hann_window(WIN)  # fp32, periodic
    xt = torch.from_numpy(x)
    spec = torch.stft(
        xt, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
        window=window, center=True, pad_mode="reflect",
        normalized=False, return_complex=True,
    )  # (CH, 1025, T)
    assert spec.shape == (CH, 1025, T_FRAMES), spec.shape
    np.savez(
        OUT / "stft.npz",
        x=x,
        window=window.numpy(),
        spec=torch.view_as_real(spec).numpy(),  # (CH, 1025, T, 2)
    )
    print("stft.npz:", spec.shape)


def gen_rmsnorm():
    """RMSNorm parity: F.normalize(x) * sqrt(dim) * gamma, dim=256."""
    rs = np.random.RandomState(7)
    rows, dim = 4096, 256
    x = rs.randn(rows, dim).astype(np.float32)
    gamma = (rs.randn(dim) * 0.1 + 1.0).astype(np.float32)
    xt = torch.from_numpy(x)
    normed = torch.nn.functional.normalize(xt, dim=-1) * (dim ** 0.5)
    out = (normed * torch.from_numpy(gamma)).numpy()
    np.savez(OUT / "rmsnorm.npz", x=x, gamma=gamma, out=out)
    print("rmsnorm.npz:", out.shape)


FREQS_PER_BANDS = [2]*24 + [4]*12 + [12]*8 + [24]*8 + [48]*8 + [128, 129]
DIM = 256
ROWS = 1151  # frames


def gen_bandsplit():
    """BandSplit parity: per-band RMSNorm + Linear(dim_in->256), random weights."""
    import json, torch
    from torch import nn
    rs = np.random.RandomState(11)
    nb = len(FREQS_PER_BANDS)
    x = rs.randn(ROWS, 2 * sum(FREQS_PER_BANDS) * 2).astype(np.float32) * 0.5  # (t, 4100)
    gammas, ws, bs = [], [], []
    outs = []
    off = 0
    for b, f in enumerate(FREQS_PER_BANDS):
        dim_in = 2 * f * 2
        seg = torch.from_numpy(x[:, off:off + dim_in])
        gamma = torch.from_numpy((rs.randn(dim_in) * 0.1 + 1.0).astype(np.float32))
        w = torch.from_numpy((rs.randn(DIM, dim_in) * 0.05).astype(np.float32))
        bias = torch.from_numpy((rs.randn(DIM) * 0.05).astype(np.float32))
        # reference: RMSNorm = F.normalize(seg) * sqrt(dim_in) * gamma, then Linear
        normed = torch.nn.functional.normalize(seg, dim=-1) * (dim_in ** 0.5)
        out = torch.nn.functional.linear(normed * gamma, w, bias)  # (t, 256)
        outs.append(out.numpy())
        gammas.append(gamma.numpy()); ws.append(w.numpy().reshape(-1)); bs.append(bias.numpy())
        off += dim_in
    y = np.stack(outs, axis=1)  # (t, 62, 256)
    np.savez(OUT / "bandsplit.npz", x=x, gamma=np.concatenate(gammas), w=np.concatenate(ws), b=np.concatenate(bs), out=y)
    print("bandsplit.npz:", y.shape)


def gen_gemm():
    """GEMM parity: Y = X @ W.T + bias for the 4 model shapes, M=2048."""
    rs = np.random.RandomState(13)
    shapes = [(256, 1536), (512, 256), (256, 1024), (1024, 256)]  # (K, N)
    M = 2048
    data = {}
    for gi, (K, N) in enumerate(shapes):
        x = (rs.randn(M, K) * 0.3).astype(np.float32)
        w = (rs.randn(N, K) * 0.05).astype(np.float32)
        bias = (rs.randn(N) * 0.05).astype(np.float32)
        y = torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(w), torch.from_numpy(bias)).numpy()
        data[f"x{gi}"], data[f"w{gi}"], data[f"b{gi}"], data[f"y{gi}"] = x.reshape(-1), w.reshape(-1), bias, y.reshape(-1)
    np.savez(OUT / "gemm.npz", **data)
    print("gemm.npz: 4 shapes M=2048")


def main():
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(0)
    gen_stft()
    gen_rmsnorm()
    gen_bandsplit()
    gen_gemm()


if __name__ == "__main__":
    sys.exit(main())