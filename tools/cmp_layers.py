import numpy as np


def cmp(name, ref_path):
    ref = np.load(ref_path).ravel().astype(np.float64)
    got = np.fromfile(name, dtype="<f4").astype(np.float64)
    n = min(len(ref), len(got))
    a, b = got[:n], ref[:n]
    sig = (b ** 2).sum()
    noise = ((a - b) ** 2).sum()
    snr = 10 * np.log10(sig / (noise + 1e-30))
    corr = np.corrcoef(a, b)[0, 1]
    print(f"{name}: SNR {snr:7.2f} dB corr {corr:.5f} peaks {np.abs(a).max():.3f}/{np.abs(b).max():.3f}")
    return snr


cmp("x_bs.bin", "assets/bs0.npy")
cmp("x_L00.bin", "assets/L0t.npy")
cmp("x_L01.bin", "assets/L0f.npy")
cmp("x_L11.bin", "assets/L1f.npy")
cmp("x_L111.bin", "assets/L11f.npy")
