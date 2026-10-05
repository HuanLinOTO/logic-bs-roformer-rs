import numpy as np

# 参考 mask: (1, 6, T, 4100) band-packed (band, fi, ch, c)
ref = np.load("assets/masks0.npy")[0]  # (6, 1151, 4100)
# 我们的 glu_all: (s, band, t, 516 padded) — (fi*4 + ch*2 + cc) 列
glu = np.fromfile("x_glu.bin", dtype="<f4").reshape(6, 62, 1151, 516)

T = 1151
freqs = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,4,4,4,4,8,8,8,8,8,8,8,8,8,8,8,8,16,16,16,16,16,16,16,16,16,16,16,16,32,32]
f0 = np.cumsum([0] + freqs)  # band start bin

# 参考 4100 列 = (band) 分组，每组 freqs[b]*4 (fi, ch, c)
# 逐 stem SNR
for s in range(6):
    sig = 0.0
    noise = 0.0
    for b in range(62):
        d = freqs[b] * 4
        # 参考 band 段: ref[s, t, off:off+d]
        off = int(f0[b] * 4)
        r = ref[s, :, off:off + d]  # (T, d)
        g = glu[s, b, :, :d]  # (T, d)
        sig += (r.astype(np.float64) ** 2).sum()
        noise += ((g.astype(np.float64) - r.astype(np.float64)) ** 2).sum()
    print(f"stem {s}: mask SNR {10 * np.log10(sig / (noise + 1e-30)):7.2f} dB")
