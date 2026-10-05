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

- **33/9 字行填充（bank 冲突修复，第二轮重试）**：ncu 定位 fragment 读
  8-way bank 冲突（shared wavefronts 64% 峰值、54 MB excessive/launch），
  给全部 f16 GEMM/QK/PV 内核行距 32→33、8→9 后**所有内核统一变慢
  5-7×**（gemm 765→3850 µs、QK 677→3974 µs），SNR 不变。结论：**该
  工具链对非 2 次幂行距的 shared 数组访问有系统性编译惩罚**（疑为
  向量化/寻址优化失效）。第一轮的 33 字填充、XOR swizzle、ldmatrix
  结论一致 —— bank 冲突路径已两次证伪，彻底关闭。

- **cp.async 流水线 GEMM（gemm_f16_async + pack_h16 + f16 权重）**：内核
  764 µs/launch，与 float4-f32 版（765 µs）完全相同 —— 全局加载延迟根本
  不是瓶颈；ncu 的 "L1TEX 91%" 主体是 mma fragment 的 shared 读流量，
  cp.async 无法减少它。已回退调用（内核保留作参考）。

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

### 第 24 轮（2026-10-05 晚）补充

1. **行距惩罚的再发现与量化**：33/9 字填充使全部 f16 内核统一变慢 5-7×
   （gemm 765→3850 µs、QK 677→3974 µs、mask1 822→5843 µs），SNR 不变。
   非 2 次幂行距在当前 cuda-oxide 代码生成下有系统性惩罚；bank 冲突路径
   （padding/XOR swizzle）两轮证伪，彻底关闭。
2. **flash 内核回退到 32 字行距后**：频率轴 0.90→~0.79 ms/launch；时间轴
   flash 从 3.34（33 距离）降到 ~2.6 ms/launch，**追平旧三段链**（此前
   "时间轴 flash 更慢"的结论部分是 33 距离惩罚造成的）。现已双轴统一
   flash：单一代码路径，softmax_stats 与 p_big 流量彻底消失，
   wall 107.5 ms（RTF 0.0358），SNR 80.87 dB。
3. 当前分布（每前向）：注意力 38.6 ms（35%）、三 GEMM 57 ms（52%）、
   mask 8.3、rmsnorm 3.3、misc 1.5。

### 第 25 轮补充（两个新证伪）

1. **双行组 warp GEMM（gemm_f16_2rg）**：每 warp 32 行（accA+accB 两个
   [[f32;4];8]），B fragment 复用翻倍、LDS/mma 0.625→0.375。正确但
   **217 ms**（2× 慢）——64 个累加 float 超出该工具链的寄存器保持能力，
   第四次确认 ~32-float 累加上限。
2. **cp.async flash（v6/v6b）**：K/V 经 cp.async 双缓冲流式加载。v6 的
   教训：cp.async 是裸字节拷贝，f32 源（64 f32/行=256B）与 f16 tile
   （32 词=128B）布局不匹配 → 全 NaN。v6b（QKV GEMM 直出 f16x2 词，
   cp.async 8 块×16B/行）正确（SNR 80.97，双重 f16 舍入影响 7e-4）但
   **116.3 ms**：gather 延迟本已被 warp 级 MLP 掩盖，cp.async 逐块
   issue 反而更贵。qkv16 基础设施保留在 git 历史中可复用。

### 第 26 轮补充

- **128 行 q-tile flash（32 字行距）**：114.4 ms（比 64 行差 7 ms）。
  第三次确认 64 行 / 128 线程是该内核在此工具链的最优几何。
- **GEMM 认知校准**：基线 nsys 复核显示 PyTorch 的 cuBLAS SGEMM 在同
  形状上同样只有 ~17 TFLOP/s（58.5 ms GEMM / ~1000 GFLOP）——我们的
  f16 内核（16 TFLOP/s）已处于同工作量 parity，并非工具链短板；该
  工作负载（M=16058 高瘦、K=256/512）在 3080 上就是 L2/布局受限。
  剩余差距在注意力：flash 38.6 ms（1.63 TFLOP/s）vs PyTorch fmha
  24.6 ms（2.56 TFLOP/s）。

### 第 27 轮成果（2026-10-05 深夜）

两个净收益（已合入）：

1. **flash PV 半选**：去掉 SVt 转置 pass（每 warp ~1536 次 shared 操作 →
   128 次 LDS），shared 32→24KB。107.6 → **106.6 ms**。
2. **mask GEMM 重写**：K16 标量加载 → K64 tile + float4（屏障 64→16 次/
   块）。8.3 → ~6.8 ms。106.6 → **105.1 ms**（RTF 0.0351，SNR 80.87）。
   **累计 1.149×。**

ncu（32 距离 flash）：时间轴 2.54 ms/launch，L2 82.9%、DRAM 70.7%
（522GB/s，1.33GB/launch）——K/V 被 5 个 q-tile 重复读且 82MB 工作集
超过 5MB L2。

- **192 行 tile/384 线程**（K/V 重读 5→2、占用率 32→50%）：138.4 ms，
  第四种几何证伪。小块多驻留（2480 块）在这个同步密集内核中胜出，
  64 行/128 线程最终确立。

### 第 27 轮后续：residual/gelu float4（重大）

- **residual + gelu GEMM float4 化**（与 QKV 同法）：105.1 → **99.1 ms**
  （**1.218×**，RTF 0.0331）。至此全部五个 f16 GEMM 内核统一为
  K64 tile + float4 加载。SNR 80.87 不变。
- flash gather float4 向量化：102.3 ms（反亏 3 ms）——warp 跨行拆散
  128B 事务，标量对（每 warp 恰一行 128B）在此内核更优，已回退。
- 192 行 tile：138.4 ms（第四种几何证伪，见上）。

### 第 28 轮补充

- **持久块 flash**（272 块 grid-stride 循环处理 (组,tile)，意图让同组
  K/V 留在 L2 消除 2.6× DRAM 放大）：正确但 **104.7 ms**（差 5.5 ms）
  ——硬件调度器对 2480 个自然块的分配本就优于固定 stride 循环（尾部
  串行 + 无法自适应两种 launch 形态）。L2 放大消除的第五种尝试证伪。
- 现状确认：**99.1 ms / RTF 0.0330 / SNR 80.87（1.22×）**。
  分布：注意力 36.9（37%）、residual 19.7、qkv 18.7、gelu 12.9、
  mask 6.9、rmsnorm 3.3、misc 1.7。

### 第 28 轮补充（两个结构证伪）

- **双组/块 flash**（每块顺序处理 2 个 (b,h) 组，块数减半）：102.1 ms
  （差 3 ms）——2072 块的天然并行度/负载均衡胜过调度开销摊薄。
  flash 结构变体至此六种证伪（64/128✓、128/256、128/128、192/384、
  持久 272、双组/块），**64 行 / 128 线程 / 1 组每块为最终形态**。
- 现状：**99.1 ms / RTF 0.0330 / SNR 80.87（1.22×）**。

### 第 29 轮：全链 f16 激活存储（重大）

三步递进，每步独立验证（SNR 80.99 全程）：

1. **qkv16**：QKV GEMM epilogue 直出 f16x2 词（hout 内核），flash 的
   Q/K/V gather 改读词（unpack+RoPE+pack 在寄存器）。99.1 → **97.5 ms**。
2. **scaled16**：flash epilogue 打包输出（33→16.5 MB/launch），out-proj
   换 gemm_f16_resid_a16（A/W 全 f16x2 词拷贝，无转换）。97.5 → **95.2**。
3. **ff1_16**：GELU epilogue 打包（66→33 MB/launch），ff2 同 a16。
   95.2 → **91.9 ms（1.315×，RTF 0.0306）**。

要点：激活的 f16 量化点与原先"加载时转换"相同（数值等价，除 scaled/ff1
下游即 mma）；写流减半对 DRAM 带宽墙的 flash 与读流减半对 GEMM 加载相
均净收益；GEMM 的 mma/LDS 相不变（gemm_f16_async 早已暗示 GEMM 墙不在
加载侧，收益主要来自 flash 与 epilogue）。

### 第 30 轮：h16 链 + mask hidden_t f16

1. **rmsnorm_gates_h16**：归一化输出经 shuffle_xor 配对打包成 f16x2 词
   （偶 lane 取奇邻居值；shuffle 必须在分支外全收敛执行——首版在
   if 内 shuffle 导致 SNR 0.32，已修复）。QKV GEMM（hout）A 改纯词拷贝。
   91.9 → **89.3 ms**。
2. **mask hidden_t f16**：gemm1 tanh epilogue 打包（394→197 MB 写流），
   gemm2 A 词拷贝（读流减半）。89.3 → **88.9 ms（1.360×，RTF 0.0296）**。

### 第 31 轮：flash XOR swizzle（净收益）+ GEMM swizzle 证伪

- ncu：flash 的 shared 访问 80% 是 bank 冲突浪费（58.9M wavefronts 中
  46.8M excessive）。
- **词级 XOR swizzle**（w' = w ^ (row & 7)，纯 pow2 运算，行距保持 32）：
  flash 88.9 → **87.0 ms（1.388×）**。
- **GEMM chunk 级 swizzle**（c4' = c4 ^ (row&7)，保留 STS.128 块写）：
  **360 ms**——复杂移位/异或地址表达式同样触发代码生成惩罚（与非 2 次幂
  行距同源的 ~4× 惩罚）。GEMM 的冲突优化第三次证伪（33 距离、词级
  swizzle 的复杂形态、chunk swizzle），仅 flash 的简单 XOR 形式可用。

### 第 31 轮终局：GEMM 词级 XOR swizzle —— 1.64× 达成

- **全部 f16 GEMM 内核的 shared 读写改用词级 XOR swizzle**
  （w' = w ^ (row & 7)，纯 pow2 运算、行距保持 32）：
  87.0 → **73.5 ms（1.643×，RTF 0.0245，SNR 80.99）**。
- 关键教训（三次实验的完整故事）：
  1. 33/9 字行距 → ~5× 编译惩罚（第 24 轮）；
  2. chunk 级 swizzle（移位+掩码复合表达式）→ 360 ms，同样惩罚；
  3. **纯 w ^ (r & 7) 简单形式 → 零惩罚，8 路 bank 冲突消除**。
  惩罚的根源是复杂地址表达式破坏代码生成，而非 swizzle 本身。
- 最终分布（每前向）：注意力 33.0（44%）、residual 12.1、qkv 11.1、
  gelu 9.0、mask 4.9、rmsnorm 2.9、misc 1.4。

### 第 32 轮：整曲 demix GPU OLA + 全异步流水线

- **问题**：单 chunk 前向 73.5 ms 但整曲 wall 反超（11.9 s vs pymss 9.7 s）。
  每 chunk 串行地 GPU 前向 → stream sync → pcm 113 MB D2H → host OLA
  0.34 s，GPU 大段空转。
- **改造**（gather 式 OLA 内核 + 全异步流水）：
  1. `ola_demix` 内核：一线程一输出样本 j，重汇聚覆盖 pos=j+1024 的
     <=4 帧 C2R 输出，窗口×istft_inv×fade 一次融合，累加进常驻
     result/counter 显存缓冲（每 launch 内写不相交，chunk 间靠单
     in-order 流排序 —— 无原子、无竞争）。fade/首尾 border 覆盖在内核
     内重算（与 host 公式逐位一致）。
  2. 输入流水线：双 ping-pong x_dev + 双 pinned host stager，H2D 用
     `copy_from_pinned_host_async`（零每 chunk 分配），pinned 复用以
     per-slot CUDA event 保护。
  3. chunk 间零同步：bench_forward(None) 全链无隐式 sync，OLA 内核入队
     后直接进入下一 chunk；整曲只在结尾一次 sync + 384 MB result 一次
     下载（84 ms）。
- **结果**：整曲 180.5 s → **GPU wall 7.02 s（RTF 0.0389，pymss 9.7 s
  的 1.38×）**；14 chunk 完全背压流水（第 4 chunk 起 enqueue 恰等一个
  chunk 的 GPU 时间，GPU 零空转）。正确性：新旧输出逐 stem SNR
  **143+ dB**（max|diff| 4.8e-7，纯 fp32 结合顺序差异）。

### 第 33 轮：注意力结构实验两次证伪（宽 tile / 注意力主序重排）

ncu（T=1151 时间轴 flash，25.8ms/launch）：L2 吞吐 85.3%（墙）、DRAM 76.5% 忙
但数据率仅 ~100GB/s、占用率 33%（smem 限制）、scheduler 77.5% 无 eligible、
65.5% stall 在 L1TEX scoreboard。两个结构性实验均以数据证伪：

1. **256 行宽 q-tile**（512 线程，寄存器结构不变，K/V 流量 2.63GB→0.73GB）：
   T=259 bench 73.5→93.9ms，T=1151 时间轴 25.8→31.9ms/launch。流量不是墙；
   尾 tile 浪费 + 512 线程 barrier + LDS 压力反而更重。
2. **K/V 注意力主序预重排**（新 qkv_kv_attn16 内核，K 的 RoPE 前移消表读；
   重排本身仅 0.46ms/launch，内容逐位验证正确 badk=0/badv=0；修复 K/V 基址
   bug 后 SNR 80.99 不变）：T=1151 时间轴 flash 25.8→33.2ms，整曲 7.02→8.41s。
   教训：折叠布局下 496 个 grp 的 K/V 访问虽然各自 190KB 跨度，但聚合恰好是
   219MB 缓冲的顺序扫描（DRAM 页局部性好、L2 命中 85.9%）；逐 grp 连续布局
   把聚合流打散成 ~30 条相距 4-5MB 的独立流，反而更差。

结论：该 flash 内核对"减流量/改布局"两类手段均不敏感（延迟+占用率受限），
剩余可信杠杆只有 cp.async 双缓冲（但 GEMM 路线已实测零收益）或接受现状。
本轮保留产出：GPU OLA 流水线（第 32 轮）不变，主分支回到 7.02s。

### 第 34 轮：QKV/mask GEMM 权重 f16x2 化（净收益）+ Q 寄存器化证伪

- **发现**：resid/FF GEMM 的 W 早已是打包 f16x2（out_w_h/ff_w1_h/ff_w2_h），
  但 QKV GEMM（hout）与 mask 两级 GEMM 仍在用 f32 W、每次 cooperative load
  时现场 cvt——W 被行块网格反复重读（QKV 每 launch 高达 ~857MB 的 L2 流量）。
- **改造**：hout/mask_gemm1/mask_gemm2 的 W 参数改 `&[u32]`（qkv_w_h 与新增
  mask_w1_h/mask_w2_h，上传时 pack_f16x2 一次性打包），加载循环照 gelu 内核
  的 [u32;4] 16B 直拷（cvt 与流量同时减半）。
- **结果**：bench 73.5 → **71.9ms（1.68×）**；整曲 7.02 → **6.95s（RTF
  0.0385）**；SNR 80.99 不变，整曲输出与改造前逐位一致（pack 时机前移、
  RNE 舍入相同）。
- **证伪**：flash Q 片段寄存器化（删 SQ、Q 不变量驻留寄存器、占用率 33→
  ~50%）：bench 75.5ms、整曲 7.45s，两尺寸均倒退——注意力第 3 次结构实验
  证伪，确认该内核在当前工具链下已处局部最优，停止微调。

### 第 35 轮：A 操作数 f16x2 补全（FF 预归一化 + mask gemm1）——干净 2× 收益

- **改造**：`rmsnorm_h16`（FF 预归一化输出 f16x2，复用 h16 缓冲）+ FF1 GEMM
  A 改打包加载；`transpose_band_major_h16` + mask gemm1 A 改打包加载。
  A 流量与 LDG 指令数同时减半，cvt 全部前移到生产者。
- **中途事故与教训**：首版 `rmsnorm_h16` 写成 `v *= scale * gamma`（右结合），
  而原内核是 `(v * scale) * gamma`（左结合）——**f32 最后一位 ulp 的结合顺序
  差异**经 24 层 transformer 放大后：黄金 SNR 80.99→80.92，整曲 vs pymss
  63.62→61.32 dB（-2.3dB！）。改回左结合后逐位恢复。教训：**移动量化点时
  必须逐位复现上游的浮点结合顺序**，否则 ulp 差异会在深度网络里指数放大。
  （另：本轮一次 git stash 事故丢失过工作区，靠会话记录完整重建。）
- **结果**：bench 71.9 → **69.0 ms（1.75×）**；整曲 6.95 → **6.76 s（1.44×，
  RTF 0.0374）**；输出与基线**逐位一致**（六 stem 全 True，vs pymss 63.62 dB
  不变，黄金 SNR 80.99 不变）。

### 第 36 轮：时间轴 flash 的 cp.async 双缓冲 —— 攻下 65% L1TEX stall（净收益）

- **依据**：ncu 显示时间轴 flash 65.5% stall 在 L1TEX scoreboard（K/V 的
  LDG→STS→sync 串行链），此前四轮结构实验（宽 tile/主序重排/Q 寄存器化）
  均证伪——瓶颈不在流量/布局/占用率，而在加载与计算的串行化。
- **改造**：
  1. `rope_k16`：K 的 RoPE 前移到独立一遍（折叠布局不变，逐位同式）；
  2. `attn_flash_async`：K/V tile 用 cp.async 16B 双缓冲，预取 t+1 与
     t 的 mma/softmax 重叠；shared swizzle 改 **16B chunk 级 XOR
     （chunk ^ (row&7)）**——cp.async 可直写且 mma fragment 读仍无 bank
     冲突（8 group 行 × 4 tig 恰好铺满 32 bank）。V 保持原样裸拷。
- **两个 bug 教训**：首版 issue 闭包目标地址漏 swizzle（写直址/读置换 →
  SNR -2.68dB）；闭包参数勿用 &dyn Fn（设备代码不支持，改捕获）。
- **结果**：bench 69.0 → **67.7 ms（1.78×）**；整曲 6.76 → **6.58 s
  （1.47×，RTF 0.0365）**；输出与基线**逐位一致**（80.99 dB / 63.62 dB）。

### 第 36 轮附：Q 寄存器化 × cp.async 组合二次证伪

在 async 内核中再删 SQ（40→32KB，2→3 blocks/SM，占用率 16.7%→25%）：
bench 67.7→74.4ms、整曲 6.58→7.92s，倒退更甚。Q 寄存器化的散列全局
加载 + 寄存器压力在两种基线上均负收益，彻底关闭该方向。

### 第 37 轮：宽 tile 第三次证伪 + pre2 f16 证伪（精度代价 3.7dB）

1. **attn_flash_async_wide**（128 行 × cp.async，针对 ncu 显示的 L2 108%
   墙，K/V L2 请求减半 + 占用率 16.7%→33%）：bench 67.7→81.8ms、整曲
   6.58→7.85s。宽 tile 家族第三次证伪（sync-wide、async-wide 各败因不同
   但结论一致：该 mma/softmax 体结构在 64 行/128 线程处就是最优）。
2. **pre2 f16x2**（gemm2 epilogue 打包 + glu_scatter 解码，流减半）：
   提速为零（mask 链本就不受带宽限制），黄金 SNR 80.99→**77.30（-3.7dB）**
   ——GLU 前激活的 f16 往返直接进频谱掩码，精度代价远超收益。教训：
   **pre-GLU 激活必须保 f32 存储**。

ncu 补充（async 版）：L2 吞吐 108.7%（墙）、DRAM 52.6%、占用率 2 块/SM。
注意力在当前结构下已连续六轮无法再压，GEMM 受 LDS-fragment 墙与寄存器
约束也已到顶。

### 第 38 轮：cp.async QKV GEMM 证伪 + 双流并发的 GO 信号

1. **gemm_hout_async**（cp.async 双缓冲 + chunk-XOR swizzle + f16x2 A/W，
   从 flash 移植的成功组合）：bench 67.7→**112.2ms** 大倒退。根因：k=256
   只有 4 个 k-tile，双缓冲的 issue/commit/wait/双 sync 开销无法摊销
   （flash 时间轴有 18 个 tile 所以赢）。**cp.async 只适合深 k/长序列循环**。
   （历史上休眠的 gemm_f16_async 失败同源：无 swizzle + 浅 k。）
2. **nsys GPU metrics（demix 全程，10kHz 采样）**：SMs Active 74.9%、
   **SM Issue 仅 18.0%**、活跃 SM 上 **53.5% warp 槽位未分配**、Tensor
   Active 7.8%、DRAM 读 13%/写 26%——流水线整体远未饱和，双流共驻有
   真实空间（GO）。
3. **--dualbench 探测已实现**（make_bench_bufs 抽取 + 双流交替计时，
   提交 b9c96ad）：远程构建被节点离线阻塞（Komari agent 掉线，多次重试
   未恢复），待恢复后先跑探测再决定是否投入完整双流 demix 重构。

### 第 39 轮：双流并发探测完成 —— 边际收益（+3.8%），确认不投入

节点恢复后跑通 --dualbench（golden T=259，双流交替 10 iter）：

```text
BENCH warm aggregate:      67.29 ms/iter   （单流基线）
BENCH dual-stream aggregate: 64.75 ms/iter （双流交替）  → +3.8%
```

尽管 SM Issue 仅 18%/未分配 warp 槽 53%，共驻只回收了 3.8%——共驻块在
LDS/LSU 上相互争抢，注意力内核的 per-SM 瓶颈不因并发而消失。完整双流
demix（OLA 竞争处理 + 双份 scratch + cufft 双 plan）投入远超 4% 收益，
判定不投入。**至此所有结构性优化路径均已探测完毕，流水线到达当前
工具链（cuda-oxide 无 ldmatrix/warp-spec）下的实际天花板。**

### 最终成绩（vs PyTorch 2.14.1 / pymss，RTX 3080，同机同卡同口径）

| 口径 | pymss | 本实现 | 加速比 |
|---|---|---|---|
| 单 chunk 前向（3s 合成输入） | 120.8 ms | **67.3-67.6 ms** | **1.80×** |
| 整曲 demix（3:00 真实歌曲） | 9.7 s | **6.57-6.58 s** | **1.47×** |
| 黄金 SNR（vs fp32 参考） | — | 80.99 dB | 验收线 ≥60 |
| 能量加权 SNR（vs pymss stems） | — | 63.62 dB | 六 stem 逐位可复现 |

### 下一步（按收益排序）

1. **突破 GEMM 的 LDS fragment 读墙**：实测瓶颈是每 mma 约 2.5 次 shared
   标量读（cp.async 流水线已验证：内核时间与 f32 版完全相同，全局加载
   不是瓶颈）。需要 128x128+ 大 tile（B 复用翻倍）但嵌套累加数组会
   spill —— 出路是全展开手写寄存器命名（无循环索引）或 warp-specialized
   producer/consumer 结构；或等 cuda-oxide 支持 ldmatrix 高效布局。
2. 时间轴注意力：QK 单块全行（shared fp16 P 33.8KB）+ 块内 softmax +
   PV 读 fp16 P，消除 softmax_stats 的 10.5 ms 与一半 P 流量。
3. mask GEMM 同样接 cp.async 路线。
