import numpy as np

ref = np.load("assets/chunk0_trunk.npy").ravel()
got = np.fromfile("trunk0.bin", dtype="<f4")
n = min(len(ref), len(got))
a, b = got[:n].astype(np.float64), ref[:n].astype(np.float64)
sig = (b ** 2).sum()
noise = ((a - b) ** 2).sum()
print("trunk SNR:", 10 * np.log10(sig / (noise + 1e-30)), "corr:", np.corrcoef(a, b)[0, 1])
print("peaks got/ref:", np.abs(a).max(), np.abs(b).max())
print("got[:6]", got[:6])
print("ref[:6]", ref[:6])
print("mid got", got[9000000:9000004], "ref", ref[9000000:9000004])
# 分层对比：按 (t, f) 行找最差
g2 = got[:n].reshape(-1, 256)
r2 = ref[:n].reshape(-1, 256)
err = np.abs(g2 - r2).max(1)
den = np.abs(r2).max(1) + 1e-9
rel = err / den
worst = np.argsort(rel)[-5:]
print("worst rows (t*62+f):", [(int(i), float(rel[i])) for i in worst])
print("best rows:", [(int(i), float(rel[i])) for i in np.argsort(rel)[:5]])
