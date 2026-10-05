import numpy as np

h = np.fromfile("x_hidden.bin", dtype="<u4")
print("hidden words", h.shape, "nonzero", int((h != 0).sum()), "max", int(h.max()))
g = np.fromfile("x_xb.bin", dtype="<f4")
print("xb", g.shape, "nonzero", int((g != 0).sum()), "absmax", float(np.abs(g).max()))
