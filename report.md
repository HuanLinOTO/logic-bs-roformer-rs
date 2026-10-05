# BS-RoFormer Rust/cuda-oxide 优化报告

日期：2026-10-05 · 节点：<REMOTE_NODE_NAME>RTX 3080 20GB（sm_86）· CUDA 13.3 · nightly-2026-08-28

## 一、结果总览

### 单 chunk 基准（3 秒合成输入，warm，与 PyTorch 完全同口径）

| 指标 | PyTorch 2.14.1（fp32） | 本实现 | 加速比 |
|---|---|---|---|
| wall（10 次均值） | 120.8 ms | **58.2 ms** | **2.07×** |
| RTF | 0.0403 | **0.0194** | — |
| 六 stem SNR（vs fp32 参考） | —（参考本身） | **80.99 dB** | 验收线 ≥60 dB |

### 真实歌曲端到端（Hanser《Cyberangel》，3:00，14 chunk demix）

| 指标 | pymss | 本实现 |
|---|---|---|
| 整曲 wall | 9.7 s | **5.53 s（1.75×）**，GPU 前向 **0.395 vs 0.69 s/chunk（1.75×）**，14 chunk 全异步流水零空转 |
| 逐 stem SNR（vs pymss fp32） | — | bass 59.2 / drums 69.6 / other 62.2 / vocals 69.3 / guitar 64.3 dB，能量加权 **63.6 dB** |

最终每前向 kernel 分布（nsys，58.3ms 口径）：时间轴 flash 19.7ms（34%）、
residual GEMM 10.9、QKV 8.4、频率轴 flash 7.6、FF1+GELU 6.3、
RMSNorm 6.4、mask 两级 3.1、rope 0.65、glu/apply 0.7、STFT/杂项 1.5。

第二轮优化（ldmatrix 系列，第 40-43 轮）在保持**六 stem 与基线
逐字节一致**的前提下：bench 67.3 → 58.2ms（1.80×→**2.07×**），
整曲 6.57 → **5.53s（1.48×→1.75×）**。

## 二、加速手段（四类）

### 1. 精度策略：fp16 tensor core + fp32 累加（有损、受控）

- 全部 GEMM 操作数转 fp16，`mma.m16n8k16`，**累加器 fp32**（≈ autocast 精度）
- 激活显存存储 f16x2 化（h16/qkv16/scaled16/ff1_16/hidden16），带宽减半
- flash attention 在线 softmax **全程 fp32**（max/exp/归一化），仅概率矩阵喂 mma 前转 fp16
- STFT/C2R/ISTFT/OLA demix **全程 fp32** 无损
- 实测误差：黄金输入 SNR 80.99 dB（幅度误差 ~1e-4，与 fp16 尾数 10bit 理论一致）

### 2. 算子融合（无损）

| 融合 | 内容 | 收益来源 |
|---|---|---|
| **attn_flash_tc** | RoPE+QK+在线 softmax+PV+门控 sigmoid 单 kernel，分数零落显存 | 消除 132MB/层 score 往返 + softmax_stats 独立内核 |
| GEMM epilogue | bias+残差+GELU/tanh+打包全在出口 | 消除逐元素内核与额外读写 |
| rmsnorm_gates | 归一化与 8 头门控投影合并，shuffle 配对直出 f16x2 | 消除 h 的 f32 写读 |

### 3. 布局与访存（无损）

- **词级 XOR swizzle**（`w' = w ^ (row & 7)`，行距保持 2 的幂）：消除 8 路 bank
  冲突——单项贡献 87.0→73.5ms（最大单笔收益）
- f16x2 词打包贯穿全链；float4/词块加载；2 的幂行距（非 2 幂有 ~5× 编译惩罚）
- fold 布局（(t,f) 折叠 token）使注意力 gather 保持 warp 内合并

### 4. ldmatrix warp 协作 fragment 加载（无损，第二轮核心）

- **工具链翻案**：cuda-device/wmma.rs 一直提供 ldmatrix_x1/x2/x4(.trans)，
  "cuda-oxide 无 ldmatrix"是误记；由此打开第二轮优化。
- 5 个热路径 GEMM + 双轴 flash 的 mma fragment 读全部换 ldmatrix：
  A 用 x4（16×16），B 用 x4（[n][k] 行主 = B 转置存储，non-trans 分布
  恰为 mma B fragment），PV 的 V 用 x2_trans（[key][d] = B 直存）。
  GEMM 每 mma 组 20 条标量 LDS → 5 条 ldmatrix；flash 每 kt 迭代
  80 LDS → 13 条。
- shared swizzle 同步升级为 **16B chunk 级 XOR**（(c4^(row&7))*4，
  与 cp.async 兼容、ldmatrix 行头 16B 对齐、8 行×4 词铺满 32 bank）。
- ncu 依据：改造前 resid GEMM 42 条 warp 指令/mma、每 scheduler 每
  21 cycle 一条（纯延迟受限）；改造后单 launch resid 1.0ms→204µs、
  QKV 925→433µs、FF1 750→321µs。

### 5. 调度（无损）

批量单 launch 替代逐 (b,h) 循环；权重/scratch 预分配；C2R plan 复用；warm 计时口径；
rmsnorm 的 gamma/8×256 gate_w 每 block staging 进 shared（gate_w 的
每 warp 8KB 重读曾造成 ~132MB/launch 的 L2 流量）。

## 三、里程碑时间线

```
120.8ms（PyTorch 基线）
  └─ warm 基建 + 双端 nsys 剖析          122.0 → 定位注意力差距
  └─ flash 注意力（先频率轴后双轴）        111
  └─ QKV GEMM float4                     108.4
  └─ flash PV 半选 + mask K64/float4      105.1
  └─ residual/gelu float4                99.1   (1.22×)
  └─ 全链 f16 激活存储（3 步）            91.9   (1.32×)
  └─ h16 链 + mask hidden f16            88.9   (1.36×)
  └─ flash 词级 swizzle                  87.0   (1.39×)
  └─ GEMM 词级 swizzle                   73.5   (1.64×)
  └─ GPU OLA + 全异步整曲流水             整曲 6.57s
  └─ cp.async 双缓冲时间轴 flash          67.3   (1.80×)
  └─ GEMM ldmatrix x4 + chunk swizzle    63.1
  └─ flash 双轴 ldmatrix                  58.7
  └─ tc/rmsnorm 收尾                      58.2   (2.07×) ✅ 整曲 5.53s (1.75×)
```

## 四、正确性验证体系

- 黄金 3s 输入全链路 SNR（每次优化后回归）：80.87–81.03 dB
- 阶段级 parity：STFT/bandsplit/每层注意力/QK 分数（host f64 重算三方对比）
- 真实歌曲 vs pymss fp32：逐 stem 59–70 dB
- 全曲模式修掉 3 个仅在长输入暴露的 bug：交错立体声 reflect_pad 声道交叉、
  flash 调试写 T=1151 越界、demix OLA 中心填充裁剪缺失 + 块内计数器误用绝对坐标

## 五、证伪记录（避免重复投入）

| 方案 | 结果 | 根因 |
|---|---|---|
| 33/9 字行距填充 | 全内核慢 5-7× | 非 2 幂行距触发代码生成惩罚 |
| chunk 级 swizzle（移位+掩码） | 360ms | 复杂地址表达式同样触发惩罚 |
| cp.async 流水线 GEMM | 与 float4 版完全同速 | 瓶颈在 shared fragment 读，非全局加载 |
| 寄存器预取双缓冲 / 128×128 tile / 双行组 warp | 186/231/217ms | 嵌套/跨循环寄存器数组 spill 到 local memory（累加上限 ~32 float） |
| flash 六种几何（128/256、192/384、持久块、双组…） | 全部更慢 | 64 行/128 线程/1 组每块为最优 |
| flash gather float4 化 | +3ms | warp 跨行拆散 128B 事务 |
| 256 行宽 q-tile（512 线程） | 时间轴 25.8→31.9ms | K/V 流量不是墙；尾 tile 浪费 + barrier/LDS 压力 |
| cp.async QKV GEMM（浅 k=256） | 67.7→112.2ms | 双缓冲开销无法在 4 个 k-tile 上摊销（cp.async 只适合深循环） |
| pre2/glu 链 f16 | 零提速，SNR −3.7dB | pre-GLU 激活的 f16 往返直接进频谱掩码 |
| 双流并发（探测实测） | +3.8%（67.3→64.7ms） | 共驻块争抢 LDS/LSU；投入产出比不足，不投入 |
| K/V 注意力主序预重排（RoPE 前移） | 时间轴 25.8→33.2ms，整曲 7.02→8.41s | 折叠布局聚合访问本就是顺序扫描；逐 grp 连续反而打散 DRAM 页局部性 |
| 128×128/512thr 大 tile QKV（warp 4×4 各 32×32，acc 无 spill） | 64.54ms（+5.8ms） | A 流量减半 < 512 线程 barrier + A-ldmatrix 翻倍；ldm/mma 5/8→6/8 |
| cargo-oxide --unchecked-indexing | 66.75ms（+3.7ms） | 去掉 predicated 边界检查反而打乱寄存器分配/调度 |
| cuBLAS / cuBLASLt | 未集成（留作天花板参照） | 719 冲突实为 context/stream 绑定可修；microbench：resid 形状 42.8 TFLOPS（我方 20× 差距），工具链内自有手段已试尽 |

工具链两条硬约束：**非 2 幂行距惩罚**、**嵌套寄存器数组必 spill**。

## 六、与 PyTorch 的效率对比（每前向）

| 组件 | PyTorch | 本实现 |
|---|---|---|
| GEMM 合计 | 58.5ms（cuBLAS SGEMM ~17 TFLOP/s） | ~28.7ms（ldmatrix f16；cuBLASLt f16 同形状 42-57 TFLOP/s 为参照上限） |
| 注意力 | 24.6ms（fmha fp32，2.56 TFLOP/s） | ~27.3ms（双轴 flash ldmatrix，含 RoPE+门控融合；L2 带宽墙） |
| elementwise | ~36ms | ~4ms（全部融合进 epilogue） |

GEMM 已超 cuBLAS fp32 一倍；距 cuBLASLt f16 还有 ~2×（需 warp-spec
producer/consumer，工具链暂无 async barrier）。

## 七、产物

- 代码：`src/main.rs`（全部 CUDA 内核 + host 编排），main 分支（ldmatrix 系列至 `0fe398e`）
- 文档：`README.md`、`docs/BENCHMARK.md`（43 轮完整实验日志）
- 工具：`tools/bench_ref.py`（PyTorch 基线）、`lbrr --bench/--separate/--forward-only`、
  `tools/separate_ref.py`（pymss 分离）、`tools/download_file.js`（分块下载）
- 本地 stem：`separated_local/cyberangel_{vocals,other}.mp3`
