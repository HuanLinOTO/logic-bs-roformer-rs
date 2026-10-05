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
    with torch.inference_mode():
        raw = torch.from_numpy(x)[None].cuda()
        batch, ch, L = raw.shape
        stft_repr = torch.stft(
            raw.reshape(batch * ch, L),
            **model.stft_kwargs, window=model.stft_window(raw.device), return_complex=True,
        )
        stft_repr = torch.view_as_real(stft_repr).reshape(batch, ch, -1, stft_repr.shape[-1], 2)
        b, s, f_, t, c = stft_repr.shape
        spec = stft_repr.permute(0, 2, 1, 3, 4).reshape(b, f_ * s, t, c)
        bs_in = spec.permute(0, 2, 1, 3).reshape(b, t, f_ * s * c)
        np.save("assets/bsin0.npy", bs_in[0].cpu().numpy().astype(np.float32))
        # raw stft in (ch*T, 1025, 2) frame order for direct comparison
        raw = stft_repr.reshape(b, s, f_, t, c)  # (b, ch, f, t, c)
        raw_ft = raw.permute(0, 1, 3, 2, 4).reshape(b, -1)[0]  # (ch*t*f*c,)
        np.save("assets/stft0.npy", raw_ft.cpu().numpy().astype(np.float32))
        x_t = model.band_split(bs_in)
        np.save("assets/bs0.npy", x_t[0].cpu().numpy().astype(np.float32))
        for li, (time_t, freq_t) in enumerate(model.layers):
            b_, t_, f_, d = x_t.shape
            x_t = time_t(x_t.permute(0, 2, 1, 3).reshape(b_ * f_, t_, d)).reshape(b_, f_, t_, d).permute(0, 2, 1, 3)
            if li == 0:
                np.save("assets/L0t.npy", x_t[0].cpu().numpy().astype(np.float32))
            x_t = freq_t(x_t.reshape(b_ * t_, f_, d)).reshape(b_, t_, f_, d)
            if li == 0:
                np.save("assets/L0f.npy", x_t[0].cpu().numpy().astype(np.float32))
            if li == 1:
                np.save("assets/L1f.npy", x_t[0].cpu().numpy().astype(np.float32))
            if li == 11:
                np.save("assets/L11f.npy", x_t[0].cpu().numpy().astype(np.float32))
    print("done")


if __name__ == "__main__":
    main()
