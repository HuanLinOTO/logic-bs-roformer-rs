# 完整波形推理验收（2026-10-07）

本轮完成完整波形基准、按轴后端准备、生产/诊断资源分离、正确的 cuBLASLt 调优和紧凑 Mask 布局。RTX 3080 最终完整块为 **157.89 ms**：相对初始 B0 的 158.97 ms 降低 **0.68%**；在最终五轮交替配对组中，B0 为 159.79 ms，收益为 **1.19%**。**8% 的延迟目标和 70% 标称 MFU 目标未达到。**没有为达到目标降低精度。

确定的主要收益是显存：T=1151 的生产 scratch 从 **9.373 GiB 降至 2.322 GiB**，减少 **7.051 GiB**。GPU 常驻权重从 **996.41 MiB 降至 336.18 MiB**。NVML 按进程采样的 GPU 峰值从 **10.883 GiB 降至 3.164 GiB**。

完整证据：[初始 B0](benchmarks/2026-10-07-baseline.json)、[分层结果、逐 stem 误差、冷启动和 profile](benchmarks/2026-10-07-results.json)。早期不含最终 OLA 的测量保存在[历史实验记录](benchmarks/2026-10-05-history.md)，不能与本报告直接混用。

## 测量口径

- 主设备：RTX 3080 20GB，SM 8.6，驱动 595.91.07；CUDA 13.3、cuDNN 9.10.2、既有 Linux nightly-2026-08-28 编译环境。
- 模型：dim=256、12 个双轴层、8 heads、head_dim=64、62 bands、6 stems，44.1kHz stereo，FFT=2048/hop=512。生产块 588800 samples/T=1151，短 golden 132300 samples/T=259。
- 输入先驻留 GPU；CUDA Event 从 STFT 前开始，到 GPU ISTFT overlap-add、归一化和中心裁剪后的波形结束。cuFFT 绑定同一个实际推理 stream。加载、分配、H2D/D2H、文件 I/O、计划构建和算法调优均在热计时外，另记初始化细项。
- 5 轮交替 A/B；每次进程先独立准备算法，再预热 5 次、计时 20 次。表中为每轮 Event 中位数的中位数，host 时间也保存在 JSON。所有 3080 正式配对组轮间 CV 均小于 2%。未锁频、未改功耗限制或驱动。
- FP16 矩阵输入、FP32 累加和残差，保留原 erf 形式 GELU 及 FP16 舍入边界；未开启 TF32、INT8 或 FP8。
- 完整块有效矩阵运算量 6,069,010,774,016 FLOPs，短块 1,013,639,108,608 FLOPs，FMA=2，不计 padding。3080 的 FP16→FP32 稠密标称峰值固定为 59.53536 TFLOPS。最终完整块标称 MFU **64.56%**；未知设备输出 null 和原因。Boost 遥测与标称分母分开。

## 分层配对结果

| 变更或候选 | 完整块 A → B（ms） | 完整块变化 | 短块变化 | 决定 |
|---|---:|---:|---:|---|
| 仅资源清理及可靠后端准备 | 158.729 → 158.726 | 基本持平 | 慢 0.064% | 采用，满足 ≤1% 回退门槛 |
| 修正 GEMM 调优及描述符缓存 | 159.397 → 159.084 | 快 0.196% | 快 0.048% | 采用，解决错误调优与生命周期问题 |
| 紧凑 Mask 布局 | 159.472 → 157.925 | 快 0.970% | 快 0.889% | 采用，显存收益确定 |
| Mask2 有效 tile 表 | 158.341 → 158.360 | 慢 0.012% | 慢 0.065% | 采用，消除无效 CTA，未宣称速度收益 |
| FF1 handwritten-async | 157.839 → 172.817 | 慢 9.49% | 慢 9.07% | 拒绝作为默认 |
| FF1 cuBLASLt FP32 + erf-GELU + FP16 pack | 158.261 → 165.140 | 慢 4.35% | 慢 4.00% | 拒绝作为默认 |
| Q/K 融合 + gate 广播实验组合 | 158.283 → 157.850 | 快 0.273% | 快 0.162% | 未达门槛，默认保持 split 和原 gate |
| 最终组合 vs 同轮 B0 | 159.787 → 157.890 | **快 1.187%** | **快 1.016%** | 5/5 轮同向，逐位一致 |

FF1 的独立核函数测量包含额外 GELU/pack：M=71362 时 handwritten 0.957 ms、async 1.678 ms、cuBLASLt-erf 1.353 ms；M=16058 时为 0.271/0.408/0.331 ms。两个候选均未达到“局部至少快 10%、完整模型至少快 1%”的采用门槛。显式 CLI 选项保留用于复现实验，auto 在未有合格数据的 SM/shape 上保持 handwritten。

Nsight 测得原 Q/K/gate 周边共 **16.624 ms/完整块**，融合与广播组合为 **16.346 ms**，仅减少约 1.67%，未达 10% 周边门槛。Q/K 自身为 10.701 → 10.339 ms；gate 广播为 5.923 → 6.006 ms，故生产 gate 保留原实现。最终显式 fused 选项只合并 Q/K；广播核函数仅保留在诊断回归中。最终默认仍为 split，表中实验组合的完整块数据不能当成另一个组合的实测。

## 显存和初始化

| 口径 | B0 | 最终 | 说明 |
|---|---:|---:|---|
| 生产 scratch，T=1151 | 9.373 GiB | 2.322 GiB | 先移除 4.898 GiB 旧资源，再减少 2.153 GiB Mask padding；tile 表仅 628B |
| 常驻 GPU 权重 | 996.41 MiB | 336.18 MiB | FP32 parity 矩阵不进入生产集合 |
| cuBLASLt workspace | 32 MiB | 32 MiB | 上限不变 |
| cuDNN workspace | 本次形状 0B | 本次形状 0B | 实现支持非零 workspace，上限 64MiB，单流复用最大需求 |
| 调优临时 D | 长期保留 dummy | 峰值约 209.07 MiB，调优后释放 | 不再计入稳态驻留 |
| NVML 进程 GPU 峰值 | 10.883 GiB | 3.164 GiB | 包含 driver/library 内部占用，与显式分配统计区分 |
| 进程主机峰值 RSS，中位数 | 2.165 GiB | 1.463 GiB | 3 次新进程，减少约 32.4% |

三次进程墙钟分别为 B0 **6.625/4.533/5.491 s**，最终 **5.752/6.188/6.722 s**；中位数为 **5.491 → 6.188 s**。**冷启动没有改善，约慢 12.7%**，且这些新进程复用系统文件/JIT缓存，不代表清空全部缓存后的首次安装启动。

权重转换与上传分项中位数从 **1.234 s 降至 0.984 s**，分配与输入上传约 **17.0 → 9.6 ms**；正确的有界调优与首次准备从 **1.184 s 增至 2.198 s**，抵消了前述收益。调优使用真实只读 W/X/C 和独立 D，最多 8 个候选，每候选预热 3 次并测 3×10 次 CUDA Event；失败候选不参与排名，CUDA 执行/同步失败直接中止。算法只在进程内缓存，没有跨版本持久化 opaque 算法。

## 数值和边界

- 主设备全部九个 fixture 通过：短 golden、音乐开头/中间/结尾完整块、静音、立体声不对称、左右声道冲激、真实 T=62 缓存冲突样本。每个 stem 按相对同一 FP32 参考的门槛检查；静音使用绝对误差。完整块没有误套短 golden 的 SNR≥60dB 门槛。
- 全部正式结构/布局对照和最终组合在 3080 上逐位一致。短 golden 仍超过 80dB。
- GPU OLA 对独立 host OLA 最大差 1.19e-7；非默认 stream 下的 STFT/ISTFT 冲激、静音、声道映射和重复覆写回归通过。
- RoPE/gate 覆盖两个轴及 T=1/37/62/63/64/65/259/1151，Q/K/V 和 gate 输出逐位一致。FF1 覆盖 M=1/37/127/129/16058/71362，检查尾行 zero-fill 及调优不修改生产输入。
- Mask 用可识别 one-hot 权重和 NaN 哨兵验证所有 pre2 槽位、全部 62 band/6 stem 的 GLU 和复数谱映射，覆盖 d=8、d=516、最后列 tile 和最后 stem。
- Windows 原生真实整曲验证：37 samples、8秒、恰好588800 samples、30秒和不足一步的尾块；30秒执行3块，输出长度/6个双声道 stems/finite/有效尾部均正确，空输入明确拒绝。
- 后端 CLI、环境变量、缺库回退和显式失败六种场景通过。10项单元测试覆盖完整缓存键、T=62不同 strides、非零 workspace、分配失败/超限清理、heuristic A→B→FALLBACK、执行失败和析构计数。

WSL 4060 Ti 交叉验证发现旧 host 计时调优会跨进程选择不同算法，新纯速度选择也可能降低个别短 stem 的相对 SNR。因此仅对已验证的 **Linux / RTX 4060 Ti / SM8.9 / cuBLASLt 130600 / M=16058或71362** 约束为保持 B0 算术的候选，仍对候选作合法性和 Event 测量；其他设备/版本不套用这些索引。修复后两种长度均与固定同平台 B0 逐位一致，短 golden 为80.94dB。本机繁忙，配对 CV 为2.9%–6.6%，**不使用本机计时宣称收益**；用户要求后，所有后续 GPU 验收转到3080。WSL读取位置实验只完成部分样本即停止，没有据此宣称I/O收益。

## 最终热点

最后一次默认组合 Nsight Systems：GPU kernel 合计约 **157.05 ms/块**，span **157.26 ms/块**，提交空隙约 **0.133%**。主要热点仍为时间轴 cuDNN SDPA（35.03ms）和 FF1（25.65ms）。没有发现需要用 NCU 解释的新回退，因此未继续扩大 profiler 采集，也没有加入 CUDA Graph、双流、低精度或新推理框架。

## 复现

所有权重、参考和输出路径由调用者传入；运行库环境沿用各平台现有配置。


git 工作区与 WSL 构建目录可以通过 LBRR_SOURCE_DIR、LBRR_WSL_ROOT、LBRR_REMOTE_ROOT、CARGO_TARGET_DIR 覆盖。WSL 复用既有编译器源码时可设置 LBRR_VENDOR_SOURCE；同步先核验任务目录归属，再执行 rsync --delete。


默认与实验候选的 CLI、JSON 检查命令如下：

~~~bash
python tools/benchmark_matrix.py prepare --reference-root /data/dsh/logic-bs-roformer-rs --model-dir assets --audio assets/cyberangel.wav --out /tmp/lbrr-perf-fixtures
./target/release/lbrr --bench --model-dir assets --bench-ref /tmp/lbrr-perf-fixtures/full/ref_output.npz --bench-stage waveform --warmup 5 --iters 20 --attn-time cudnn --attn-freq cudnn --ff1-backend handwritten --qk-rope split --bench-json output/full.json
python tools/check_benchmark.py output/full.json --baseline output/b0-full.json
./target/release/lbrr --kernel-regression --bench-json output/kernels.json
python tools/benchmark_matrix.py run --rust-bin /path/to/b0 --candidate-bin ./target/release/lbrr --model-dir assets --fixtures /tmp/lbrr-perf-fixtures --rounds 5 --warmup 5 --iters 20 --cases full,short --out output/paired
~~~

单元测试通过 cargo oxide build -- --release --tests 构建，再用 [测试入口](../tools/run_unit_tests.py) 执行；GPU与权重可用时 [CI入口](../scripts/ci.sh) 会执行核函数回归、完整波形JSON检查和可选逐 stem B0 门槛。缺权重会明确标记资源型检查 skipped，性能验收数据来自资源齐备的3080。Windows工具链和缓存仍使用 D盘；本轮没有修改 vendor 提交或工具链 pin。
