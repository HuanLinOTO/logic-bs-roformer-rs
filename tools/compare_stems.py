import numpy as np, glob, os
a_dir, b_dir = "/data/dsh/logic-bs-roformer-rs/separated2", "/data/dsh/logic-bs-roformer-rs/separated_ldmx"
ok = True
for f in sorted(glob.glob(a_dir + "/*.wav")):
    name = os.path.basename(f)
    g = os.path.join(b_dir, name)
    if not os.path.exists(g):
        print("MISSING", name); ok = False; continue
    x = np.frombuffer(open(f, "rb").read(), dtype=np.uint8)
    y = np.frombuffer(open(g, "rb").read(), dtype=np.uint8)
    same = x.shape == y.shape and bool((x == y).all())
    print(name, "BITWISE", same, len(x), len(y))
    ok = ok and same
print("ALL-BITWISE-EQUAL" if ok else "MISMATCH-FOUND")
