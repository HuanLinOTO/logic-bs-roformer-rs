# logic-bs-roformer-rs 3080 基准状态（2026-10-05）

## 测试环境

- GPU：<REMOTE_NODE_NAME>，RTX 3080 20GB（sm_86）
- Rust/CUDA：nightly-2026-08-28 + cuda-oxide，CUDA 13.3
- PyTorch：2.14.1+cu126
- 输入：3 秒合成双声道音频（与 `assets/ref_output.npz` 完全一致）

## PyTorch 基线

```text
BENCH single-chunk(3s): 120.8 ms, RTF 0.0403
golden check: rel err 4.395e-07
```

测量方式：模型加载后 warmup 3 次、迭代 10 次取平均；fp32 math SDPA
（Flash/memory-efficient/cuDNN attention 均未启用）。

## Rust/cuda-oxide 当前结果

最终正确性：

```text
E2E SNR vs ref_output: 80.90 dB
```

Nsys CUDA kernel 总时间（同一 3 秒前向，STFT 到 6 stem C2R）：

```text
Total GPU kernel time: ~148.6 ms
gemm_f16_128x64        63.45 ms / 96 launches
softmax_stats          16.50 ms / 24
attn_pv_f16_bn         12.86 ms / 24
attn_qk_f16             9.70 ms / 24
rope_qkv_to_attn        8.25 ms / 24
rmsnorm                 6.18 ms / 49
bandsplit               5.23 ms / 1
mask GEMM1+GEMM2        8.89 ms / 12
other elementwise/FFT  ~17.5 ms
```

单次冷启动前向 wall time 在 235–313 ms 间波动；主要额外开销来自首次
大 scratch 分配/释放与输出 D2H。权重加载+上传约 0.7–1.15 s，基准中已
排除。

## 结论

正确性已超过验收（SNR 80.90 dB ≫ 60 dB），性能尚未超过 PyTorch：
kernel-only 仍比 120.8 ms 基线慢约 23%，冷启动 wall time 慢约 2x。
因此“大幅度加速”的目标还未完成。

## 已完成的主要优化

1. 24 层 transformer GEMM 改为 FP16 tensor-core（f32 累加），K tile 64。
2. 双轴 attention 改为 batched FP16 QK + warp softmax stats + FP16 PV，
   消除 1,488 次 host 循环/launch。
3. RoPE 与 QKV attention-major reorder 融合，减少一次完整 QKV 读写。
4. MaskEstimator 改为每个 stem 2 个 grouped GEMM launch + 全局 GLU scatter，
   372×2 次 host GEMM 循环清零。
5. softmax 从 one-thread/row 改为 warp/row。

## 下一步（按收益排序）

1. 继续压缩 `gemm_f16_128x64`（63 ms）：权重常驻 FP16/BF16、ldmatrix、
   double-buffer shared memory。
2. attention 三个 kernel 合计 39 ms：探索 tensor-core flash attention或
   更深 K tile；避免 materialized score 的三遍访存。
3. 预分配/复用全部 mask scratch，消除冷启动 wall time 中约 80–170 ms 的
   allocator/D2H 开销。
4. BandSplit（5.2 ms）与 RMSNorm/elementwise 融合。
