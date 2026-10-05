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
    caps = {}

    def mk(name):
        def hook(mod, inp, out):
            caps[name] = out.detach().cpu().numpy()
        return hook

    hs = [model.band_split.register_forward_hook(mk("band_split"))]
    li = 0
    for time_t, freq_t in model.layers:
        if li in (0, 5, 11):
            hs.append(time_t.register_forward_hook(mk(f"L{li}_time")))
            hs.append(freq_t.register_forward_hook(mk(f"L{li}_freq")))
        li += 1

    with torch.inference_mode():
        model(torch.from_numpy(x)[None].cuda())
    for h in hs:
        h.remove()
    for k, v in caps.items():
        a = v[0]  # (t, f, d) after the permutes? verify shape
        print(k, a.shape)
        np.save(f"assets/bisect_{k}.npy", a.astype(np.float32))


if __name__ == "__main__":
    main()
