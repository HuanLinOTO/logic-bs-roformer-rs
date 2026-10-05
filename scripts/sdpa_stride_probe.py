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

# fold layout probe: qkv16 is [T*62, 768] fp16 words view; emulate with elements
for T, tag in [(259, "bench"), (1151, "song")]:
    qkv = torch.randn(T * 62, 768, device=dev, dtype=torch.float16)
    # time axis: (B=62, H=8, S=T, D=32) element strides: b=768, h=32, s=62*768, d=1
    st = (768, 32, 62 * 768, 1)
    qf = qkv.as_strided((62, 8, T, 32), st, 0)
    kf = qkv.as_strided((62, 8, T, 32), st, 256)
    vf = qkv.as_strided((62, 8, T, 32), st, 512)
    qc = qf.contiguous(); kc = kf.contiguous(); vc = vf.contiguous()
    flops = 2 * 2 * 62 * 8 * T * T * 32
    for name, q, k, v in [("fold-strided", qf, kf, vf), ("packed", qc, kc, vc)]:
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
            try:
                ms = bench(lambda: F.scaled_dot_product_attention(q, k, v))
                print(f"time-{tag} {name} cudnn: {ms:.3f} ms  {flops/ms*1e-9:.1f} TFLOPs")
            except Exception as ex:
                print(f"time-{tag} {name} cudnn FAIL {str(ex)[:90]}")
    # freq axis: (B=T, H=8, S=62, D=32): b=t stride=768, h=32, s(band)=768? NO:
    # freq axis token = t*62+band, so b(token t) stride = 62*768? wrong direction.
    # freq: B over t (stride 62*768), S over band (stride 768)
    qff = qkv.as_strided((T, 8, 62, 32), (62 * 768, 32, 768, 1), 0)
    kff = qkv.as_strided((T, 8, 62, 32), (62 * 768, 32, 768, 1), 256)
    vff = qkv.as_strided((T, 8, 62, 32), (62 * 768, 32, 768, 1), 512)
    flopsf = 2 * 2 * T * 8 * 62 * 62 * 32
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
        try:
            ms = bench(lambda: F.scaled_dot_product_attention(qff, kff, vff))
            print(f"freq-{tag} fold-strided cudnn: {ms:.3f} ms  {flopsf/ms*1e-9:.1f} TFLOPs")
        except Exception as ex:
            print(f"freq-{tag} fold-strided cudnn FAIL {str(ex)[:90]}")
print("OK")
