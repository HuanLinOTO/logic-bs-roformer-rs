import numpy as np


def snr(a, b):
    n = min(len(a), len(b))
    a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
    sig = (b ** 2).sum()
    noise = ((a - b) ** 2).sum()
    print(f"  SNR {10 * np.log10(sig / (noise + 1e-30)):8.2f} dB corr {np.corrcoef(a, b)[0, 1]:.5f} peaks {np.abs(a).max():.4f}/{np.abs(b).max():.4f} n={n}")


spec = np.fromfile("x_spec.bin", dtype="<f4")
stft_ref = np.load("assets/stft0.npy").ravel()
print("STFT raw (ch-major frames):")
snr(spec, stft_ref)

xin = np.fromfile("x_xin.bin", dtype="<f4")
bsin = np.load("assets/bsin0.npy").ravel()
print("reorder/band_split input (t, 4100):")
snr(xin, bsin)

# 若 STFT 错：定位首个差异通道/帧
if True:
    n = min(len(spec), len(stft_ref))
    d = np.abs(spec[:n].astype(np.float64) - stft_ref[:n].astype(np.float64))
    bad = np.argsort(d)[-5:]
    print("worst spec idx:", [(int(i), float(d[i])) for i in bad])
    T = 1151
    for i in bad[:3]:
        fr = i // (1025 * 2)
        rest = i % (1025 * 2)
        fbin = rest // 2
        print(f"  idx {i}: frame {fr} (ch {fr // T}, t {fr % T}), bin {fbin}, val {spec[i]:.6f} ref {stft_ref[i]:.6f}")
