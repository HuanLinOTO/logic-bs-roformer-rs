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
Total GPU kernel time: ~122.0 ms
gemm_f16_residual      26.68 ms / 48  (out projection + FF2, residual fused)
qkv gemm_f16           22.54 ms / 24
ff1 gemm + GELU        16.21 ms / 24
attn PV (raw QKV)      15.01 ms / 24
time-axis softmax       10.56 ms / 12
time-axis QK+RoPE        9.06 ms / 12
freq QK+RoPE+stats       7.86 ms / 12
bandsplit norm+GEMM       0.29 ms / 2
mask GEMM1+tanh          4.91 ms / 6
RMSNorm+gates             3.46 ms / 49
mask GEMM2               3.57 ms / 6
GLU/mask/FFT/STFT        ~1.9 ms
```

单次冷启动前向 wall time 在 218–289 ms 间波动；主要额外开销来自首次
大 scratch 分配/释放与输出 D2H。权重加载+上传约 0.65–1.10 s，基准中已
排除。

## 结论

正确性已超过验收（SNR 80.90 dB ≫ 60 dB）。kernel-only 122.0 ms，已略低于
120.8 ms 的 PyTorch 基线（约快 0.6%），但距离“大幅度加速”验收目标仍很
远；冷启动 wall time 仍受 allocator/D2H 干扰。

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
9. RMSNorm / fused gate 投影改为 lane-major 连续访问，消除 8-stride
   uncoalesced warp 事务；相关 kernel 从 7.14 ms 降到 3.46 ms。
10. softmax 从 one-thread/row 改为 warp/row，并使用硬件 ex2 近似。

## 下一步（按收益排序）

1. 继续压缩 65 ms transformer GEMM：cp.async double buffering、更优
   warp tile / ldmatrix 组合、权重常驻 FP16。
2. 时间轴 score softmax 仍有 10.6 ms；探索更便宜的 partial reduction 或
   tensor-core flash attention，避免 materialized score 的完整读。
3. 预分配/复用全部 scratch，建立 warm benchmark loop，隔离冷启动
   allocator/D2H 开销。
## Nsight Compute 定位与已否决方案

对稳定版 QKV `gemm_f16_128x64`（grid 24×126、block 256）采集 hardware counter：

```text
L1/shared throughput              76.3%
shared load bank conflicts     54,365,655
DRAM throughput                  14.8%
SM throughput                    36.4%
active warps                     49.1%
kernel duration                 1.10 ms
```

瓶颈明确是 shared-memory fragment load 的 bank conflict，而非 DRAM 或 tensor core 峰值。以下方案均已实现并验证正确性，但在 RTX 3080 上比当前 128×64 布局慢，已回退：

| 方案 | QKV 24-launch 总时间 | 结论 |
|---|---:|---|
| 稳定版 128×64 / 8 warps | 22.51 ms | 当前最优 |
| 128×128 / 16 warps / block512 | 32.06 ms | 更大 tile 增加同步与占用压力 |
| 128×128 / 8 warps / 16 accumulators | 125.29 ms | 寄存器/串行 MMA 严重限制 |
| FP16 packed A + W | 25.91 ms | 减半流量但地址/转换开销更大 |
| 33-word padded shared rows | 80.50 ms | 生成地址计算代价高 |
| corrected row-tag XOR swizzle | 26.69 ms | bank conflict 降低但总时间变慢 |
| ldmatrix.x4 + ldmatrix.x2 | 28.38 ms | fragment load 指令减少仍不敌开销 |

另外尝试了 time-axis softmax 融合：QK epilogue 用 atomic max 聚合行最大值，PV 在加载 score 时同步累计 `exp(score-max)` 行和。Layer 0 正确且无 NaN，但后续层出现 NaN；将频率轴拆回独立 proven kernel 后仍复现，判断与当前 atomic/key 状态交互不稳定，已完整回退。cuBLASLt heuristic 可找到 TF32 算法，但实际调用会使 cuda-oxide stream 出现 unspecified launch failure，继续不可用。

