import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
from separate_ref import load_model, read_wav_f32  # noqa: E402


def main():
    x, sr = read_wav_f32(Path("assets/chunk0.wav"))
    border = 29440
    xp = np.pad(x, ((0, 0), (border, border * 2)), mode="reflect")
    chunk = xp[:, :588800].copy()
    model = load_model(Path("assets"))
    with torch.inference_mode():
        out = model(torch.from_numpy(chunk)[None].cuda())[0].cpu().numpy()
    np.save("assets/chunk0_demixref.npy", out)
    print("saved", out.shape)


if __name__ == "__main__":
    main()
