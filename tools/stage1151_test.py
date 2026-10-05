#!/usr/bin/env python3
"""Dump pymss intermediates for chunk0: trunk output + final stems' mask input."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from separate_ref import load_model, read_wav_f32  # noqa: E402


def main():
    x, sr = read_wav_f32(Path("assets/chunk0.wav"))
    model = load_model(Path("assets"))
    caps = {}

    def post_hook(mod, inp, out):
        caps["trunk_out"] = out.detach().cpu().numpy()
        raise RuntimeError("STOP")

    hs = [model.final_norm.register_forward_hook(post_hook)]
    try:
        with torch.inference_mode():
            model(torch.from_numpy(x)[None].cuda())
    except RuntimeError as e:
        print("caught:", str(e)[:200])
        assert "STOP" in str(e)
    finally:
        for h in hs:
            h.remove()
    arr = caps["trunk_out"][0]  # (b, t, f, d)
    print("trunk_out shape", arr.shape, "peak", np.abs(arr).max())
    np.save("assets/chunk0_trunk.npy", arr.astype(np.float32))


if __name__ == "__main__":
    main()
