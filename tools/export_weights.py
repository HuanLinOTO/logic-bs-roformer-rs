#!/usr/bin/env python3
"""Export logic_bs_roformer.ckpt -> model.safetensors (keys unchanged).

The positional arguments default to the original Windows development paths:
    python tools/export_weights.py [logic_bs_roformer.ckpt] [assets/model.safetensors]
"""
import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

DEFAULT_SRC = Path(r"D:/Projects/.model-research/logic_bs_roformer.ckpt")
DEFAULT_DST = Path(__file__).resolve().parent.parent / "assets" / "model.safetensors"


def export_checkpoint(src: Path, dst: Path) -> None:
    sd = torch.load(src, map_location="cpu", weights_only=True)
    if not isinstance(sd, dict):
        raise TypeError(f"unexpected ckpt type {type(sd)}")
    # The checkpoint shares storage for the 24x shared qkv/out biases and the
    # per-axis rotary freqs (use_shared_bias=True). Clone to detach so
    # safetensors accepts them; the Rust loader re-deduplicates via the §3 map.
    sd = {k: v.clone() for k, v in sd.items()}
    n = len(sd)
    total = sum(v.numel() for v in sd.values())
    print(f"{n} tensors, {total/1e6:.2f}M params, {sum(v.numel()*4 for v in sd.values())/1e6:.1f} MB fp32")
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(dst))
    print(f"saved {dst} ({dst.stat().st_size/1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", nargs="?", type=Path, default=DEFAULT_SRC, help="input logic_bs_roformer.ckpt")
    parser.add_argument("dst", nargs="?", type=Path, default=DEFAULT_DST, help="output model.safetensors")
    args = parser.parse_args()
    export_checkpoint(args.src, args.dst)


if __name__ == "__main__":
    sys.exit(main())
