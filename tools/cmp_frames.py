import wave

import numpy as np

with wave.open("assets/chunk0.wav", "rb") as wv:
    raw = wv.readframes(wv.getnframes())
x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
x = x.reshape(-1, 2).T  # (2, N)
N = x.shape[1]
win = np.asarray([0.5 * (1 - np.cos(2 * np.pi * i / 2048)) for i in range(2048)], dtype=np.float32)
T = N // 512 + 1
pad = 1024
# reflect pad
xp = np.empty((2, N + 2 * pad), dtype=np.float32)
xp[:, pad:pad + N] = x
xp[:, :pad] = x[:, pad:0:-1]
xp[:, pad + N:] = x[:, N - 2:N - 2 - pad:-1]
frames = np.zeros((2 * T, 2048), dtype=np.float32)
for ch in range(2):
    for t in range(T):
        frames[ch * T + t] = xp[ch, t * 512:t * 512 + 2048] * win
np.save("assets/frames_ref.npy", frames.ravel())
print("frames ref saved", frames.shape)
