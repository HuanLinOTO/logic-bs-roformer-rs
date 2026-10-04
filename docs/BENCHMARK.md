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
Total GPU kernel time: ~124.7 ms
gemm_f16_residual      26.68 ms / 48  (out projection + FF2, residual fused)
qkv gemm_f16           22.54 ms / 24
ff1 gemm + GELU        16.21 ms / 24
attn PV (raw QKV)      15.01 ms / 24
time-axis softmax       10.56 ms / 12
time-axis QK+RoPE        9.06 ms / 12
freq QK+RoPE+stats       7.86 ms / 12
bandsplit norm+GEMM       0.29 ms / 2
mask GEMM1+tanh          4.91 ms / 6
RMSNorm+gates            7.14 ms / 49
mask GEMM2               3.57 ms / 6
GLU/mask/FFT/STFT        ~1.9 ms
```

单次冷启动前向 wall time 在 218–289 ms 间波动；主要额外开销来自首次
大 scratch 分配/释放与输出 D2H。权重加载+上传约 0.65–1.10 s，基准中已
排除。

## 结论

正确性已超过验收（SNR 80.90 dB ≫ 60 dB），性能尚未超过 PyTorch：
kernel-only 124.7 ms，比 120.8 ms 基线慢约 3.2%；冷启动 wall time 仍
受 allocator/D2H 干扰。因此“大幅度加速”的目标还未完成，但差距已从
23% 缩小到 3.2%。

## 已完成的主要优化

1. 24 层 transformer GEMM 改为 FP16 tensor-core（f32 累加），K tile 64。
2. 双轴 attention 改为 batched FP16 QK/PV，消除 1,488 次 host 循环/launch。
3. QK/PV 直接读取 folded raw QKV：RoPE、q 缩放、V 展开、head gate 和
   folded 写回全部融合，删除独立 RoPE/reorder/un-reorder/gate kernel。
4. 频率轴 QK epilogue 直接输出 softmax max/scale，删除 12 次 stats launch。
5. out-proj/FF2 residual、FF1 GELU、mask GEMM1 tanh 融合进 GEMM epilogue。
6. RMSNorm 与 8 个 head gate 投影融合。
7. MaskEstimator 改为每个 stem 2 个 grouped GEMM launch + 全局 GLU scatter，
   372×2 次 host GEMM 循环清零。
8. BandSplit 拆成 padded RMSNorm + grouped FP16 tensor-core GEMM，从
   5.8 ms 降到 0.29 ms。
9. softmax 从 one-thread/row 改为 warp/row，并使用硬件 ex2 近似。

## 下一步（按收益排序）

1. 继续压缩 65 ms transformer GEMM：cp.async double buffering、更优
   warp tile / ldmatrix 组合、权重常驻 FP16。
2. 时间轴 score softmax 仍有 10.6 ms；探索更便宜的 partial reduction 或
   tensor-core flash attention，避免 materialized score 的完整读。
3. 预分配/复用全部 mask scratch，消除冷启动 wall time 中约 60–120 ms 的
   allocator/D2H 开销。
4. 预分配/复用全部 mask scratch，消除冷启动 wall time 中约 60–120 ms 的
   allocator/D2H 开销。
