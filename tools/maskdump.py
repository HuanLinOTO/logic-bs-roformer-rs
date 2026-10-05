import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
from separate_ref import load_model, read_wav_f32  # noqa: E402


def main():
    x, sr = read_wav_f32(Path("assets/chunk0.wav"))
    model = load_model(Path("assets"))
    orig = model._estimate_masks

    def wrapped(xx):
        out = orig(xx)
        if not hasattr(wrapped, "done"):
            wrapped.done = True
            a = out.detach().cpu().numpy()
            print("masks", a.shape, "peak", np.abs(a).max())
            np.save("assets/masks0.npy", a.astype(np.float32))
        return out

    model._estimate_masks = wrapped
    with torch.inference_mode():
        model(torch.from_numpy(x)[None].cuda())


if __name__ == "__main__":
    main()
