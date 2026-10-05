# logic-bs-roformer-rs 3080 基准状态（2026-10-05，第二版）

## 测试环境

- GPU：<REMOTE_NODE_NAME>，RTX 3080 20GB（sm_86）
- Rust/CUDA：nightly-2026-08-28 + cuda-oxide，CUDA 13.3
- PyTorch：2.14.1+cu126
- 输入：3 秒合成双声道音频（与 `assets/ref_output.npz` 完全一致）

## PyTorch 基线

```text
BENCH single-chunk(3s): 120.8 ms wall, RTF 0.0403
nsys GPU kernel 合计:   ~119.6 ms/forward（14 次前向平均）
  ampere_sgemm_128x64_tn   36.5 ms/次
  fmha_cutlassF_f32(SDPA)  24.6 ms/次
  ampere_sgemm_128x128_tn  21.6 ms/次
  elementwise/reduce 等    ~37 ms/次
```

测量方式：模型加载后 warmup 3 次、迭代 10 次取平均；fp32 math SDPA。

## Rust/cuda-oxide 当前结果（warm bench，与基线同口径）

`lbrr --bench --iters 10`（权重与全部 scratch 预分配，warmup 3）：

```text
BENCH warm aggregate: 108.4 ms/iter, RTF 0.0361
BENCH warm per-iter (sync each): min 108.5 / med 109.0 / max 109.1 ms
BENCH final SNR vs ref_output: 81.03 dB   (验收线 >= 60 dB)
```

**对比 PyTorch：wall 120.8 → 108.4 ms（1.12x）；GPU kernel 119.6 → ~109 ms。**

nsys 每前向 kernel 分布（11 次平均）：

```text
gemm_f16_residual_128x64   24.1 ms / 48 launches   (out proj + FF2, residual fused)
gemm_f16_128x64 (float4)   18.5 ms / 24            (QKV projection)
gemm_f16_gelu_128x64       14.5 ms / 24            (FF1 + GELU)
attn_flash_tc (freq 轴)    10.8 ms / 12            (单 launch 全融合注意力)
softmax_stats (time 轴)    10.5 ms / 12
attn_pv_raw_f16 (time 轴)   9.8 ms / 12
attn_qk_rope_f16 (time 轴)  8.1 ms / 12
mask_gemm1/gemm2            8.3 ms / 12
rmsnorm(+gates)             3.3 ms / 49
STFT/GLU/mask/FFT 等        ~1.4 ms
合计                       ~109.3 ms / 237+12 launches
```

## 今日（第二轮）实验记录

### 已合入的优化

1. **warm bench 基准**（`--bench`）：权重/scratch 一次分配、C2R plan 复用、
   计时口径与 bench_ref.py 完全一致。冷启动 208–290 ms 的波动被证实主要来自
   首次分配与 D2H，warm 稳定在 108.4 ms。
2. **双端 nsys**：拿到 PyTorch kernel 级基线（见上），确认其注意力是单一
   fmha 内核 24.6 ms，而我们的三段式合计 40.3 ms —— 差距定位成功。
3. **attn_flash_tc**（tensor-core flash attention）：RoPE + QK + 在线
   softmax + PV + gate sigmoid 全融合，分数不落显存。关键布局技巧：QK 的
   C fragment 与 PV 的 A fragment 同构，概率 tile 原地转 A fragment。
   - 修复过两个正确性 bug：kt 循环 acc 未清零（分数跨 tile 累加，m 恰好
     +1.0）；row_base 未随 tile 尺寸改（rows 256–258 漏写）。
   - 频率轴（seq=62，单 kt tile）：0.90 ms/launch，替换旧链（QK stats
     0.64 + PV 0.44 = 1.08 ms），全前向省 ~2 ms。
   - 时间轴（seq=259，5 个 kt tile）：最好 3.1 ms/launch，仍输给旧三段链
     2.65 ms —— kt 循环内 load+3x sync+softmax 串行化吃掉收益。ncu 证实
     DRAM 80% 忙、L2 命中 85%，本质是 qkv 折叠布局的 gather 模式。
   - 结论：**混合模式** —— 时间轴保留旧链，频率轴用 flash。
4. **gemm_f16_128x64 float4 加载**：加载指令数 /4，QKV 20.1 → 18.5 ms。

### 失败的变体（勿重复）

- flash v2/v3/v4（128 行 tile + 256 线程 + SVt 转置/寄存器预载 B）：
  124/150/122 ms。ncu：L1TEX 56–86%、DRAM 80%、occupancy 32%（shared
  限制 2 block/SM）。多维寄存器数组 w[4][8] 触发 local memory spill。
- flash v5（64 行 tile + staging 转置 + 3 sync/kt）：113 ms，转置与额外
  sync 抵消了合并加载收益。
- **寄存器预取双缓冲 GEMM**（下一 tile 的 24 个 LDG 与 mma 重叠）：
  186 ms —— av[32]/bv[16] 数组跨循环使用，直接 spill 到 local memory。
- **gemm_f16_128x128**（B tile 加倍，LDS/mma 从 2.5 降到 1.5）：
  231 ms —— acc[4][4][4] 嵌套数组同样 spill。
- 教训：**当前 cuda-oxide 工具链下，任何跨循环/嵌套索引的寄存器数组都会
  落入 local memory**；可行的内核只持有 1D acc[[f32;4];8] + 少量标量。

### 关键发现：GEMM 效率才是最大剩余空间

ncu gemm_f16_128x64（QKV）：**L1TEX 管线 90.9% 忙**（DRAM 仅 15%、SM 30%、
occupancy 33%、96 regs）。M=16058（此前误按 71342 估算，FLOPs 虚高 4.4x）：

```text
QKV GEMM   18.5 ms / 303 GFLOP = 16.4 TFLOP/s
residual   24.1 ms / 302 GFLOP = 12.5 TFLOP/s   (fp16 tensor 峰值的 ~1/5)
gelu FF1   14.5 ms / 202 GFLOP = 13.9 TFLOP/s
mask 两级   8.3 ms / ~90 GFLOP ≈ 11 TFLOP/s
```

### 下一步（按收益排序）

1. **cp.async 流水线 + f16 预转换操作数**（唯一没试对的 GEMM 路线）：
   权重上传时转 f16、rmsnorm/PV/GELU 产出 f16 激活，GEMM 内核用
   `cuda_device::async_copy::cp_async_cg_16` 双缓冲 global→shared，
   彻底消除转换与 L1TEX 串行。cuBLAS 同形状可达 ~50-65 TFLOP/s，GEMM
   57 ms → ~20 ms，总体可到 ~65 ms（1.8x）。注意保持寄存器数组 1D。
2. 时间轴注意力：QK 单块全行（shared fp16 P 33.8KB）+ 块内 softmax +
   PV 读 fp16 P，消除 softmax_stats 的 10.5 ms 与一半 P 流量。
3. mask GEMM 同样接 cp.async 路线。
