#!/usr/bin/env python3
"""Export logic_bs_roformer.ckpt -> model.safetensors (keys unchanged).

Run with the reference venv:  .venv2/Scripts/python tools/export_weights.py
Output: assets/model.safetensors (~700 MB, gitignored).
"""
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

SRC = Path(r"D:/Projects/.model-research/logic_bs_roformer.ckpt")
DST = Path(__file__).resolve().parent.parent / "assets" / "model.safetensors"


def main() -> None:
    sd = torch.load(SRC, map_location="cpu", weights_only=True)
    # The checkpoint shares storage for the 24x shared qkv/out biases and the
    # per-axis rotary freqs (use_shared_bias=True). Clone to detach so
    # safetensors accepts them; the Rust loader re-deduplicates via the §3 map.
    sd = {k: v.clone() for k, v in sd.items()}
    assert isinstance(sd, dict), f"unexpected ckpt type {type(sd)}"
    n = len(sd)
    total = sum(v.numel() for v in sd.values())
    print(f"{n} tensors, {total/1e6:.2f}M params, {sum(v.numel()*4 for v in sd.values())/1e6:.1f} MB fp32")
    DST.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(DST))
    print(f"saved {DST} ({DST.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
