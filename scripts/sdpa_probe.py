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

shapes = [
    ("time-bench",  62, 8,  259, 32),
    ("time-song",   62, 8, 1151, 32),
    ("freq-bench", 259, 8,   62, 32),
    ("freq-song", 1151, 8,   62, 32),
]
for name, B, H, L, D in shapes:
    q = torch.randn(B, H, L, D, device=dev, dtype=torch.float16)
    k = torch.randn(B, H, L, D, device=dev, dtype=torch.float16)
    v = torch.randn(B, H, L, D, device=dev, dtype=torch.float16)
    flops = 2 * 2 * B * H * L * L * D
    for bk in ["CUDNN_ATTENTION", "FLASH_ATTENTION", "EFFICIENT_ATTENTION"]:
        try:
            with torch.nn.attention.sdpa_kernel(getattr(torch.nn.attention.SDPBackend, bk)):
                ms = bench(lambda: F.scaled_dot_product_attention(q, k, v))
            print(f"{name} {bk}: {ms:.3f} ms  {flops/ms*1e-9:.1f} TFLOPs")
        except Exception as ex:
            print(f"{name} {bk}: FAIL {str(ex)[:100]}")
print("dev:", torch.cuda.get_device_name(0), "cudnn:", torch.backends.cudnn.version())
