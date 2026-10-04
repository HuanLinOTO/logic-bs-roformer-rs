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


def main():
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(0)
    gen_stft()
    gen_rmsnorm()


if __name__ == "__main__":
    sys.exit(main())