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


def gen_qkv_rope():
    """QKV+RoPE+scale parity: time-axis layout (b*seq folding), small scale.
    x (b,seq,256) -> RMSNorm -> Linear -> rotate q/k (interleaved pairs) -> q*scale,k*scale."""
    rs = np.random.RandomState(17)
    b, seq, dim, heads, dh = 2, 64, 256, 8, 64
    x = (rs.randn(b, seq, dim) * 0.5).astype(np.float32)
    gamma = (rs.randn(dim) * 0.1 + 1.0).astype(np.float32)
    w = (rs.randn(3 * heads * dh, dim) * 0.05).astype(np.float32)
    bias = (rs.randn(3 * heads * dh) * 0.05).astype(np.float32)
    xt = torch.from_numpy(x)
    h = torch.nn.functional.normalize(xt, dim=-1) * (dim ** 0.5) * torch.from_numpy(gamma)
    qkv = torch.nn.functional.linear(h, torch.from_numpy(w), torch.from_numpy(bias))
    q, k, v = qkv.view(b, seq, 3, heads, dh).unbind(2)
    # rotary: freqs = 1/10000^(2i/dh), pos*freqs, cos/sin on repeat_interleave layout
    freqs = 1.0 / (10000 ** (torch.arange(0, dh, 2).float() / dh))
    ang = torch.arange(seq).float()[:, None] * freqs[None]           # (seq, 32)
    ang = ang.repeat_interleave(2, -1)                                # (seq, 64)
    ang_c = ang[..., ::2]  # (seq, 32) after taking even positions
    cos = ang_c.cos()[None, :, None, :]
    sin = ang_c.sin()[None, :, None, :]
    def rot(t):
        te, to = t[..., ::2], t[..., 1::2]
        out = torch.empty_like(t)
        torch.sub(te * cos, to * sin, out=out[..., ::2])
        torch.add(to * cos, te * sin, out=out[..., 1::2])
        return out
    scale = dh ** -0.5
    qr, kr = rot(q) * scale, rot(k)
    np.savez(OUT / "qkvrope.npz", x=x.reshape(-1), gamma=gamma, w=w.reshape(-1), bias=bias,
             cos=cos.numpy().reshape(-1), sin=sin.numpy().reshape(-1),
             q=qr.numpy().reshape(-1), k=kr.numpy().reshape(-1), v=v.numpy().reshape(-1))
    print("qkvrope.npz: b=2 seq=64")


def gen_attn_short():
    """Short-seq SDPA parity: q,k,v (BH, 62, 64), scale=8^-0.5, fp32 math."""
    rs = np.random.RandomState(19)
    bh, seq, dh = 16, 62, 64   # small BH for the npz; math is per-(b,h)
    q = (rs.randn(bh, seq, dh) * 0.3).astype(np.float32)
    k = (rs.randn(bh, seq, dh) * 0.3).astype(np.float32)
    v = (rs.randn(bh, seq, dh) * 0.3).astype(np.float32)
    qt, kt, vt = map(torch.from_numpy, (q, k, v))
    out = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt)
    np.savez(OUT / "attn_short.npz", q=q.reshape(-1), k=k.reshape(-1), v=v.reshape(-1), out=out.numpy().reshape(-1))
    print("attn_short.npz: bh=16 seq=62")


def gen_attn_long():
    """Long-seq SDPA parity (time axis), 2-pass reference: scores->softmax->PV."""
    rs = np.random.RandomState(23)
    bh, seq, dh = 4, 256, 64
    q = (rs.randn(bh, seq, dh) * 0.3).astype(np.float32)  # unscaled; SDPA applies 1/sqrt(64)
    k = (rs.randn(bh, seq, dh) * 0.3).astype(np.float32)
    v = (rs.randn(bh, seq, dh) * 0.3).astype(np.float32)
    qt, kt, vt = map(torch.from_numpy, (q, k, v))
    out = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt)
    np.savez(OUT / "attn_long.npz", q=q.reshape(-1), k=k.reshape(-1), v=v.reshape(-1), out=out.numpy().reshape(-1))
    print("attn_long.npz: bh=4 seq=256")


def gen_gateout_ff():
    """gate_out (gates sigmoid*attn -> out proj -> +residual) and FF (GELU erf)."""
    rs = np.random.RandomState(29)
    M, dim, heads, dh, ff = 512, 256, 8, 64, 1024
    h = (rs.randn(M, dim) * 0.5).astype(np.float32)          # post-norm activations
    x_res = (rs.randn(M, dim) * 0.5).astype(np.float32)      # attention input (residual)
    attn_raw = (rs.randn(M, heads * dh) * 0.3).astype(np.float32)
    wg = (rs.randn(heads, dim) * 0.1).astype(np.float32)
    bg = (rs.randn(heads) * 0.1).astype(np.float32)
    wo = (rs.randn(dim, heads * dh) * 0.05).astype(np.float32)
    bo = (rs.randn(dim) * 0.05).astype(np.float32)
    gates = torch.sigmoid(torch.nn.functional.linear(torch.from_numpy(h), torch.from_numpy(wg), torch.from_numpy(bg)))
    scaled = torch.from_numpy(attn_raw).view(M, heads, dh) * gates.unsqueeze(-1)
    attn_out = torch.nn.functional.linear(scaled.reshape(M, heads * dh), torch.from_numpy(wo), torch.from_numpy(bo)) + torch.from_numpy(x_res)
    # FF
    hff = attn_out
    w1 = (rs.randn(ff, dim) * 0.05).astype(np.float32)
    b1 = (rs.randn(ff) * 0.05).astype(np.float32)
    w2 = (rs.randn(dim, ff) * 0.05).astype(np.float32)
    b2 = (rs.randn(dim) * 0.05).astype(np.float32)
    ff1 = torch.nn.functional.gelu(torch.nn.functional.linear(hff, torch.from_numpy(w1), torch.from_numpy(b1)))
    ff_out = torch.nn.functional.linear(ff1, torch.from_numpy(w2), torch.from_numpy(b2)) + hff
    np.savez(OUT / "gateff.npz",
             h=h.reshape(-1), x_res=x_res.reshape(-1), attn_raw=attn_raw.reshape(-1),
             wg=wg.reshape(-1), bg=bg, wo=wo.reshape(-1), bo=bo,
             w1=w1.reshape(-1), b1=b1, w2=w2.reshape(-1), b2=b2,
             attn_out=attn_out.numpy().reshape(-1), ff1=ff1.numpy().reshape(-1), ff_out=ff_out.numpy().reshape(-1))
    print("gateff.npz: M=512")


def gen_e2e_mid(model_dir):
    """Capture the post-12-layer final_norm activation on the golden 3s input."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pymss_core.modules.bs_roformer.bs_roformer import BSRoformer
    import yaml
    cfg = yaml.unsafe_load(open(model_dir / "logic_bs_roformer.yaml"))
    mcfg = dict(cfg["model"])
    for k in ["multi_stft_resolution_loss_weight", "multi_stft_resolutions_window_sizes",
              "multi_stft_hop_size", "multi_stft_normalized", "linear_transformer_depth",
              "use_torch_checkpoint", "dim_freqs_in"]:
        mcfg.pop(k, None)
    model = BSRoformer(**mcfg)
    from safetensors.torch import load_file
    sd = load_file(str(model_dir / "model.safetensors"))
    model.load_state_dict(sd, strict=False)
    model.eval().cuda()
    ref = np.load(model_dir / "ref_output.npz")
    x = torch.from_numpy(ref["inp"])[None].cuda()
    cap = {}
    def hook(mod, inp, out):
        cap["x"] = out.detach()
    model.final_norm.register_forward_hook(hook)
    with torch.inference_mode():
        y = model(x)
    np.savez(OUT / "e2e_mid.npz", x_final=cap["x"].cpu().numpy().reshape(-1))
    print("e2e_mid.npz:", cap["x"].shape, "out:", tuple(y.shape))


def main():
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(0)
    gen_stft()
    gen_rmsnorm()
    gen_bandsplit()
    gen_gemm()
    gen_qkv_rope()
    gen_attn_short()
    gen_attn_long()
    gen_gateout_ff()
    gen_e2e_mid(Path(__file__).resolve().parent.parent / "assets")


if __name__ == "__main__":
    sys.exit(main())