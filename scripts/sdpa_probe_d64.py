import torch, torch.nn.functional as F
dev = 'cuda'
def bench(fn, n=100):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n

# head_dim = 64, H = 8; fold qkv16 [T*62, 768] fp16-words -> element strides
for T, tag in [(259, "bench"), (1151, "song")]:
    qkv = torch.randn(T * 62, 1536, device=dev, dtype=torch.float16)  # elements view
    # time axis: (B=62, H=8, S=T, D=64): b=1536, h=64, s=62*1536, d=1; Q/K/V offsets 0/512/1024
    st = (1536, 64, 62 * 1536, 1)
    qf = qkv.as_strided((62, 8, T, 64), st, 0)
    kf = qkv.as_strided((62, 8, T, 64), st, 512)
    vf = qkv.as_strided((62, 8, T, 64), st, 1024)
    qc, kc, vc = qf.contiguous(), kf.contiguous(), vf.contiguous()
    flops = 2 * 2 * 62 * 8 * T * T * 64
    for name, q, k, v in [("fold", qf, kf, vf), ("packed", qc, kc, vc)]:
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
            try:
                ms = bench(lambda: F.scaled_dot_product_attention(q, k, v))
                print(f"t-{tag} {name} D64 cudnn: {ms:.3f} ms  {flops/ms*1e-9:.1f} T")
            except Exception as ex:
                print(f"t-{tag} {name} D64 cudnn FAIL {str(ex)[:80]}")
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            try:
                ms = bench(lambda: F.scaled_dot_product_attention(q, k, v))
                print(f"t-{tag} {name} D64 flash: {ms:.3f} ms  {flops/ms*1e-9:.1f} T")
            except Exception as ex:
                print(f"t-{tag} {name} D64 flash FAIL {str(ex)[:80]}")
    # freq axis: (B=T, H=8, S=62, D=64): t stride=62*1536, band stride=1536
    qff = qkv.as_strided((T, 8, 62, 64), (62 * 1536, 64, 1536, 1), 0)
    kff = qkv.as_strided((T, 8, 62, 64), (62 * 1536, 64, 1536, 1), 512)
    vff = qkv.as_strided((T, 8, 62, 64), (62 * 1536, 64, 1536, 1), 1024)
    flopsf = 2 * 2 * T * 8 * 62 * 62 * 64
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
        try:
            ms = bench(lambda: F.scaled_dot_product_attention(qff, kff, vff))
            print(f"f-{tag} fold D64 cudnn: {ms:.3f} ms  {flopsf/ms*1e-9:.1f} T")
        except Exception as ex:
            print(f"f-{tag} fold D64 cudnn FAIL {str(ex)[:80]}")
print("OK")
