# BS-RoFormer GPU 推理优化实施计划

> 执行约定：本计划获批后，按下列依赖顺序在本会话实施；每个任务形成可单独验收的变更。当前交付仅为计划。

**Goal：** 以“呵呵”RTX 3080 为主要验收设备、本地 WSL RTX 4060 Ti 为交叉验证设备，降低完整波形推理延迟、生产显存占用和初始化成本，同时保持当前 Rust 实现的数值质量。

**Architecture：** 保留单 CUDA stream 与现有 STFT → BandSplit → Transformer → MaskEstimator → ISTFT/OLA 流水线。先统一基准和后端语义，再分离生产/诊断资源，最后逐项验证 GEMM、RoPE/gate 和掩码布局候选；未通过门槛的候选不进入默认路径。GPU 核函数仍放在现有 cuda_module 内，只提取少量宿主侧配置和基准数据类型，不进行全仓库重构。

**Tech Stack：** Rust 2024、cuda-oxide、Linux CUDA 13.3、cuBLASLt、cuDNN 9.10.2、cuFFT；Linux 既有 nightly-2026-08-28。远程参考环境为 Python 3.12.14 / PyTorch 2.14.1+cu126。本轮不引入新的推理框架或 CUTLASS 依赖；仅允许为结构化测量结果增加宿主侧 serde/serde_json。

**Spec：** [性能复核报告](<D:/Projects/logic-bs-roformer-rs/output/performance-audit/report.md>)、[3080 结果汇总](<D:/Projects/logic-bs-roformer-rs/output/performance-audit/remote-summary.json>)、[cuDNN 热点](<D:/Projects/logic-bs-roformer-rs/output/performance-audit/remote/profile-cudnn.json>)、[手写 Flash 计数器](<D:/Projects/logic-bs-roformer-rs/output/performance-audit/remote/ncu-flash.csv>)。

## 全局约束与成功标准

- 固定当前模型：dim=256、12 个双轴层、8 heads、head_dim=64、62 bands、6 stems、44.1kHz stereo、n_fft=2048、hop=512。生产块长 588800；短参考块长 132300。权重文件、训练 YAML 和输出 WAV 格式保持兼容。
- 保留 FP16 矩阵输入 / FP32 累加、FP32 残差和现有 erf-GELU 定义。不开启 TF32、INT8、FP8，不以降低精度或漏算尾块换速度。
- 当前完整块 Rust + cuDNN 为 158.371ms、标称 MFU 64.37%，手写 Flash 为 366.247ms。旧 Rust 基准缺最终 OLA，**任务 1 建立的完整波形 B0 才是后续正式分母**。
- 完整块有效矩阵运算量固定为 6,069,010,774,016 FLOPs；短块为 1,013,639,108,608 FLOPs，均已由 PyTorch FlopCounterMode 验证。FMA=2，不计 padding 和无用工作；非矩阵算子的时间包含在推理时间里。
- 3080 标称 MFU 用 FP16→FP32 稠密峰值 59.53536 TFLOPS（68 SM、1.710GHz）；4060 Ti 用 44.12928 TFLOPS。频率遥测与标称 MFU 分开报告，不能把某个瞬时 Boost 值当作整段平均值。
- **硬验收：** 全部数值/尾块测试通过；生产旧 scratch 至少减少 4.7GiB；掩码紧凑布局另减少至少 2.0GiB；结构调整完整块/短块延迟回退均不超过 1%。统计内存时区分长期驻留和调优瞬时峰值。
- **收益目标：** 最终完整波形中位数较 B0 降低至少 8%，同口径标称 MFU 向 70% 提升；这是优化目标，不是预先保证。速度候选仅在通过下述收益门槛后采用，未达目标时如实记录，不降低精度补数。
- 远程复用已授权的 Komari 节点、模型和 Python 环境。密码、Cookie 和令牌只来自运行环境，不进入文件或提交。性能测试期间不并发运行其他 GPU 任务；不擅自锁频、修改功耗上限或升级驱动。
- 执行时使用隔离 worktree、WSL ext4 构建目录和远程独立构建目录；保留现有二进制作 B0。Windows Rust 工具链与缓存沿用 D 盘约定。所有临时 profiler/遥测进程在完成或失败时回收。

## 本轮范围及文件分工

| 文件 | 责任 |
|---|---|
| [主程序](<D:/Projects/logic-bs-roformer-rs/src/main.rs>) | CLI 接线、生产/诊断缓冲、现有 GPU 核函数及优化候选、权重上传与推理调度 |
| 新增 src/inference_options.rs | 后端枚举、参数优先级、已验证 shape/SM 的默认选择；不承载运行时调优框架 |
| 新增 src/benchmark.rs | 基准选项、统一结果结构、JSON 输出、数值指标和计时汇总 |
| [cuBLASLt](<D:/Projects/logic-bs-roformer-rs/src/cublaslt.rs>) | 正确的算法测量、描述符/布局缓存、FF1 FP32 输出候选及资源释放 |
| [cuDNN](<D:/Projects/logic-bs-roformer-rs/src/cudnn.rs>) | 按轴预建计划、完整缓存键、workspace、后端失败语义 |
| [cuFFT](<D:/Projects/logic-bs-roformer-rs/src/cufft.rs>)、[STFT](<D:/Projects/logic-bs-roformer-rs/src/stft.rs>) | 明确绑定执行 stream，保证 CUDA Event 边界覆盖 FFT |
| [权重模块](<D:/Projects/logic-bs-roformer-rs/src/weights.rs>) | 减少宿主复制，保持张量布局、形状验证和 checkpoint key 完整性检查 |
| 新增 tools/benchmark_matrix.py | 从已验证的诊断脚本整理 fixture 生成、Rust/PyTorch 对照、配对统计和遥测清理 |
| [PyTorch 基准](<D:/Projects/logic-bs-roformer-rs/tools/bench_ref.py>)、[整曲参考](<D:/Projects/logic-bs-roformer-rs/tools/separate_ref.py>) | 统一精度/实际 SDPA 后端记录及尾块覆盖 |
| [CI](<D:/Projects/logic-bs-roformer-rs/scripts/ci.sh>)、[WSL 同步](<D:/Projects/logic-bs-roformer-rs/scripts/sync_wsl.sh>)、[WSL 构建](<D:/Projects/logic-bs-roformer-rs/scripts/build_wsl.sh>)、[远程构建](<D:/Projects/logic-bs-roformer-rs/scripts/build_remote.sh>) | 可覆盖隔离目录、构建与有意义的 GPU 回归入口 |
| [Cargo 清单](<D:/Projects/logic-bs-roformer-rs/Cargo.toml>)及 [锁文件](<D:/Projects/logic-bs-roformer-rs/Cargo.lock>) | 仅在引入结构化输出依赖时更新 |
| [基准文档](<D:/Projects/logic-bs-roformer-rs/docs/BENCHMARK.md>) | 最终方法、完整波形基准、误差/显存/冷启动和被拒绝的候选 |

将已有诊断结果的必要摘要归档为新增 docs/benchmarks/2026-10-07-baseline.json；完整音频、权重、Nsight 二进制及海量日志继续留在实验目录，不纳入源码交付。

## 公共接口与数据契约

### CLI

保留已有参数，新增以下参数；具体名称固定，避免实施时另行设计：

| 参数 | 默认值与语义 |
|---|---|
| --bench-ref PATH | 默认 model-dir/ref_output.npz；读取 inp/out，模型权重仍来自 model-dir |
| --bench-stage waveform|frames | 默认 waveform；frames 仅用于和历史核函数基准复核，JSON 必须明确标注 |
| --warmup N | 默认 5；仅控制基准预热，iters 继续使用现有参数 |
| --bench-json PATH | 可选；写 schema_version=1 的单次基准结果，失败返回非零 |
| --attn-time auto|cudnn|handwritten | 默认 auto，独立控制时间轴 |
| --attn-freq auto|cudnn|handwritten | 默认 auto，独立控制频率轴 |
| --ff1-backend auto|handwritten|cublaslt-erf|handwritten-async | 默认 auto；初始等价于当前 handwritten，任务 4 后仅对已验证 SM/shape 切换默认 |
| --qk-rope split|fused | 仅用于可复现 A/B；初始 split，任务 5 通过门槛后默认 fused |

注意力优先级：显式 CLI > LBRR_NO_CUDNN > auto。保持旧环境变量“存在即禁用”的兼容语义；显式 cudnn 失败即报错，不静默切换。auto 优先 cuDNN，只在预建失败或不支持时按轴回退并输出原因。训练 YAML 的 flash_attn 不映射为 Rust 的手写/库实现选择，Rust 后端以这些参数和实际日志为准。

### 测量结果 schema_version=1

- identity：source/binary/model/config/reference 源码哈希、设备名/SM、驱动、CUDA/cuBLASLt/cuDNN 版本、fixture id。
- shape：batch、channels、stems、samples、T、bands；pipeline 为 frames 或 waveform。
- backend：每轴 requested/selected/fallback_reason、FF1 候选、RoPE 路径、cuBLASLt 候选标识；不序列化裸指针。
- precision：矩阵输入、累加、残差 dtype、TF32 状态；输入/输出是否驻留 GPU。
- timing_ms：初始化细项、预热次数、各轮 host/CUDA Event 耗时；RTF 分母由 samples/sample_rate 推导。
- memory_bytes：weights、scratch、workspace、调优临时量及峰值；同时保存外部遥测，避免与 PyTorch reserved memory 混淆。
- correctness：finite、输出形状、每 stem 的 SNR、max_abs_error、相对 B0 的差值、尾部覆盖与对齐结果。
- mfu：matrix_flops、nominal_peak_tflops、nominal_mfu；未知设备未配置理论峰值时输出 null 和原因，禁止猜测。

正常运行继续输出简短的人类可读状态，并保留现有 BENCH final SNR 行供旧 CI 解析。机器比较使用 JSON，不再靠截断的终端文本推断后端。

## 任务 1：建立完整波形基准和数值 B0

**依赖：** 无。**交付：** Rust/PyTorch 可比较的固定输入、完整波形计时、B0 记录和可重复构建入口。

- [ ] 为构建/同步脚本增加 LBRR_SOURCE_DIR、LBRR_WSL_ROOT、LBRR_REMOTE_ROOT 和可覆盖 CARGO_TARGET_DIR，保留当前默认。调用 rsync --delete 前核验解析后的源/目标是指定任务目录；不可指向项目根的父级、磁盘根或其他工作的目录。
- [ ] 在 cuFFT 加载 cufftSetStream，提供 CufftPlan::set_stream；STFT/C2R 计划绑定实际推理 stream。此任务仍只跑单流，先确保 Event 覆盖全部 GPU 工作。
- [ ] 保留 bench_forward 的“到 C2R frames”内部语义，新增 istft_ola GPU kernel：复用 ola_demix 的 gather、window-energy reciprocal 和中心裁剪公式，直接覆写 [stem,channel,sample] 波形，不使用跨迭代 +=。整曲仍调用现有 ola_demix，避免额外产生整块波形中间副本。
- [ ] waveform 基准计时从驻留 GPU 输入开始，到归一化后的驻留 GPU 波形结束；模型加载、plan 构建、调优、分配、H2D/D2H 和文件 I/O 均单独记录。每轮同步只在测量边界，正确性下载放在计时后；输出必须实际消费校验。
- [ ] 整理 tools/benchmark_matrix.py：以现有短参考、音频开头/中间/结尾三个完整块、静音和立体声不对称输入生成未压缩 NPZ 与 manifest。PyTorch 模块位置用 --reference-root 指定；复用远程现有依赖，WSL 无 PyTorch 时执行 Rust 的 B0/候选回归和现有短 golden。
- [ ] 修复 PyTorch --full 起点算法为覆盖最后一个有效样本的网格，并按所需最后块扩大反射填充；与当前 Rust 整曲算法一致。单块模式 L 必须满足 STFT 反射约束，空输入明确报错；整曲短音频走固定块填充。
- [ ] 输出 schema_version=1；测量 B0，归档 source/binary/model 哈希和必要原始结果。初始库/内核算法不在此任务改变。

**验收：** 同输入两侧样本数、6 stems、双声道、采样率一致；新增 OLA 的输出与既有 host OLA 数值一致。整曲覆盖短于块长、恰好一块、30 秒、最后不足一步的音频；30 秒必须覆盖 3 块，尾部不能为未计算区域。用单/双声道冲激检查 STFT/ISTFT 裁剪与通道对应，使用静音检查 NaN/除零。

**性能数据契约：** 5 轮配对、每轮预热 5 次并计时 20 次；A/B 顺序交替。host 与 CUDA Event 同报，记录时钟/功耗/温度。初始化与冷启动另做 3 次进程计时，不与热推理混合。若轮间变异系数 >2%，该轮组不用于宣称收益；检查遥测、排除外部负载后至多补测一组并标注情况。

## 任务 2：明确后端选择并提前准备 cuDNN 计划

**依赖：** 任务 1。**修改：** 主程序、inference_options、cuDNN 封装；现有 C++ wrapper 的 build/exec/workspace 参数足够，本轮保持其 ABI。

- [ ] 定义 AttentionBackendRequest { Auto, Cudnn, Handwritten }，每轴保存请求值、实际值和回退原因；解析前述 CLI/环境变量优先级。
- [ ] 将 CudnnSdpa 分为 prepare 与 execute：在修改 Q/K 之前准备好两个轴的 plan。候选仅按 heuristic A → B → FALLBACK 顺序寻找首个可用方案；本轮不做新一轮 SDPA 在线速度搜索。
- [ ] 将 plans 的缓存键从 (b,s) 扩展为 axis、B/H/S/D、dtype、Q/K/V/O strides 和 attention scale；context/设备由 CudnnSdpa 实例拥有，计划不得跨 context 使用。
- [ ] 接入真实 workspace，以当前 stream 分配，单实例单流复用最大需求；本轮上限 64MiB。超过上限、不支持或分配失败：auto 在执行前回退，显式 cudnn 返回错误。去掉 ws==0 的 panic，并检查 set_stream 的返回码。
- [ ] 正式 execute 失败时中止该次推理并保留 CUDA 错误，不在 Q 已原地 RoPE 后直接调用手写内核，避免重复旋转或继续使用已损坏的 context。
- [ ] 补齐 graph/handle/workspace 生命周期；启动日志与 JSON 必须一致地说明每个轴的选用路径。

**验收：** CLI 覆盖环境变量、库缺失、单轴不支持、非零 workspace、显式选择失败、正常自动回退均有确定结果。使用 T=62 验证两个轴虽然 B/S 相同、strides 不同也不会复用错误计划。测试失败路径后没有泄漏或二次释放；准备期间不修改生产 QKV。

## 任务 3：移除生产无用缓冲区与重复权重副本

**依赖：** 任务 2。**修改：** 主程序的 E2eScratch、BenchBufs、make_bench_bufs、separate、forward_only、e2e_test、upload_weights，以及权重模块。

- [ ] 将 E2eScratch 分为 RuntimeScratch 与 ParityScratch；生产构造函数只创建 transformer_step/bench_forward 实际需要的资源，诊断路径显式申请旧参考资源。将重复初始化集中到同一生产构造函数，保持 GPU 核函数代码所在模块不变。
- [ ] 从生产构造中移除 h、qkv、qkv_rope、qkv_attn、v_flat、scaled、oproj、ffpre、ff1、ff2、p_big、attn_out_long 等旧 FP32 scratch；legacy parity 用到时在 ParityScratch 分配。不得用零长 buffer 冒充仍被调用的内核输入。
- [ ] 分开运行与 parity 权重上传。保留生产需要的 norm、bias、BandSplit FP32 权重和 FP16 打包矩阵；仅 parity 使用的 QKV/输出投影/FFN/Mask FP32 矩阵不进入生产 GPU 常驻集合。
- [ ] 上传时借用 TransformerLayer，消除两轮 layer.clone；每个 mask stem 只整理/打包一次。GPU 上传与异步复制完成后释放 SafeTensors、宿主模型和临时拼接副本，明确最后一次读与释放的 stream 顺序。
- [ ] 保持 safetensors/YAML 格式不变，不新增磁盘预打包格式、服务进程或批量任务 API。分别测宿主读取、解析、转换、H2D；WSL ext4 权重与 /mnt/d 权重只作为受控 I/O 对照，不自动移动用户文件。

**验收：** T=1151 生产 scratch 较 B0 下降至少 4.7GiB；显式 cuDNN/手写后端都能运行；诊断 flags 仍能取得所需旧 buffer。shape/dtype/key 校验不放宽。固定后端和相同 GEMM 算法时，纯资源重排应保持输出逐位一致；算法选择变更时适用全局数值门槛。冷启动与宿主峰值内存有分项记录，稳态延迟回退不超过 1%。

## 任务 4：修正 cuBLASLt 调优并验证 FF1 候选

**依赖：** 任务 3。**修改：** cuBLASLt 封装、lt_ready、FF1 路由与核函数。

当前源码有两个明确问题：[调优](<D:/Projects/logic-bs-roformer-rs/src/cublaslt.rs#L346>)将 A/B/C/D 全部指向同一 dummy；同时 D 始终按 M×N×4 估算，而 [dummy 分配](<D:/Projects/logic-bs-roformer-rs/src/main.rs#L5827>)为 M×3072，导致 QKV FP16 输出的调优被容量检查跳过。

- [ ] 使用真实只读 W/X/C 和独立 D_scratch 计时，不修改残差或生产输出；beta=0 不读取 C。按各矩阵 dtype/stride 计算实际容量，删除错误的统一 4 字节输出估计。
- [ ] 保留最多 8 个 heuristic 候选。每候选预热 3 次，CUDA Event 测 3 组×10 次；任何调用失败都淘汰该候选，不能因提前 break 计得更短而胜出。CUDA 执行/同步错误直接终止，不能当作普通“不支持”。
- [ ] 缓存 operation/layout/算法，键覆盖 M/N/K、dtype、转置/stride、epilogue、beta 语义和 workspace 上限。每次正确更新 bias 指针；描述符/handle 用 RAII 释放。调优临时输出测完释放；workspace 保持 32MiB。算法缓存本轮仅驻留进程，不把 opaque Algo 跨版本写盘。
- [ ] 只验证三个 FF1 候选：现有 handwritten；cuBLASLt FP32 输出+独立精确 erf-GELU+FP16 打包；复用现有 swizzle/ldmatrix 的 handwritten-async（Mtile=128、Ntile=64、Ktile=64、256 threads、两级 cp.async，48KiB shared tile，保持计算与舍入顺序）。不引入其他 tile 搜索或第三方 GEMM 库。
- [ ] FF1 的额外中间输出、GELU、转换都包含在候选时间和峰值内存中。覆盖 M=16058/71362 及非整 tile 尾行；异步加载的越界行显式 zero-fill。
- [ ] auto 默认只采用在对应 SM/shape 通过完整模型收益与数值门槛的候选；未知 SM/shape 保持 handwritten。显式选择缺失库时报错。若候选都未获益，保留原 FF1，仅交付正确调优和测量结论。

**验收：** QKV 不再因为 FP16 容量误算跳过候选；调优前后生产输入/残差不变。FF1 核函数候选至少快 10%，且完整波形至少快 1%、5 组中至少 4 组同向，才进入默认组合；短块不得回退 >1%。本任务不采用 tanh-GELU 近似。FF1 即使两倍提速，按原占比整模型上限收益约 8.8%，报告不得混淆局部与整体。

## 任务 5：合并 Q/K RoPE，减少 gate 重复计算

**依赖：** 任务 4。**修改：** transformer_step、rope_k16/rope_q16_inplace/sdpa_gate 周边核函数。

- [ ] 新增 rope_qk16：一条线程处理同一 token/head 的 Q/K f16x2，共享位置/cos/sin 读取；Q 原地写回 qkv16，K 写入 k16r。保留当前旋转后的 FP16 舍入边界；scale=0.125 仍只由 cuDNN 应用。
- [ ] 保留 split 模式供 A/B；手写路径继续使用其原有 Q RoPE 语义，禁止给手写内核传已经旋转的 Q。
- [ ] sdpa_gate 按每 warp 对应一个 head 的现有布局，由一个 lane 计算 sigmoid 再广播；保留独立输出 pass 和原始 gates buffer 语义，避免破坏 parity/手写路径。若广播无净收益，保留原 gate 实现。
- [ ] 本轮不新增轴连续 pack 缓冲，也不改写 cuDNN attention epilogue；这些属于后述独立实验。现有 RMSNorm+gate 线性投影已融合，不重复实现。

**验收：** Q/K 单独比较通过现有 RoPE parity，全模型通过数值门槛；覆盖 axis=0/1、T=1/37/62/63/64/65/259/1151 的内核边界和尾线程。合并后的周边处理至少快 10%、完整波形至少快 1% 才切默认；否则保留 split。结果必须包含 Q/K/gate 整段成本。

## 任务 6：紧凑化 MaskEstimator 的输出布局

**依赖：** 任务 3；性能验收基于任务 5 已接受的组合。**修改：** 缓冲构造、mask_gemm2_f16、glu_scatter、mask_apply 和相关 parity/dump。

定义 d_b=4×freqs_per_band[b]，offset[b]=前 b 个 d 的前缀和，D_total=4100：

- pre2 使用 [stem][band][T][2d_b]，地址为 stem×T×2D_total + T×2offset[b] + t×2d_b + d。
- glu 使用 [stem][band][T][d_b]，地址为 stem×T×D_total + T×offset[b] + t×d_b + d。
- pre2_all 长度为 6×T×8200 个 f32；glu_all 为 6×T×4100 个 f32。权重布局不变，消费者按同一 offset 计算。

- [ ] 首先只改 compact 写入/读取和分配，保留每 stem 的已有 launch 顺序，核对全部 band/stem/channel/frequency 的映射。
- [ ] 再加入静态 mask2 tile 表 (band,col_tile)：按 band 升序、有效 ceil(2d_b/64) 列 tile 枚举，复用于六个 stem，减少无效 CTA；如果表寻址造成 >1% 整体回退，则保留原 grid，只保留紧凑布局。
- [ ] GLU 的两半读取位置和 complex mask 的频点/声道顺序不变。后续融合 GLU+mask_apply 的实现不包含在本轮，先完成紧凑布局的确定性收益。

**验收：** 两个缓冲区合计减少至少 2.0GiB（理论约 2.15GiB）；每个 band 的 GLU 输出和最终谱图与原布局一致。专测最窄 d=8、最宽 d=516、非整 64 列 tile、T=259/1151、最后 band 和最后 stem。旧 padded 槽位不能再被读，所有有效槽位都有唯一写者；完整块与短块回退不超过 1%。

## 任务 7：组合回归、MFU 与交付

**依赖：** 任务 1–6。

- [ ] 对“原 B0、仅资源清理、逐项接受候选、最终组合”保存分层对照，避免多个优化互相掩盖回退。主设备 3080 做配对矩阵；WSL 4060 Ti 做同输入回归和代表性完整块测量。
- [ ] 更新 CI：沿用 self-test 与既有 golden 检查，补充完整波形基准、每 stem 数值门槛、尾块覆盖、backend failure/cache 键及 compact mask 测试。缺少权重时只允许明确标记资源型测试 skipped，最终性能验收必须在资源齐备的远程完成。
- [ ] 使用既有 Windows 原生构建入口做兼容性 build/self-test/短 golden smoke，保留 D 盘工具链路径；本轮不改 vendor 编译器、toolchain pin 或无关 Windows 代码。
- [ ] 对最终组合做一次 Nsight Systems，只有新的热点或回退需要解释时再做定点 NCU。NCU 若改变时钟或导致库回退，只作为单核诊断，不进入延迟排名。
- [ ] 更新基准文档，记录同口径 B0、新延迟/MFU/内存/冷启动、实际后端和数值误差、未获益候选及原因；保留复现命令和源码/模型哈希。关闭本任务启动的遥测及 profiler 进程，收集全部结果后交付。

### 统一数值门槛

1. 所有输出 finite，shape/sample_rate/样本长度与参考一致；尾部覆盖、中心裁剪和声道映射正确。
2. 对非静音 stem，候选相对同一 PyTorch FP32 参考的 SNR 不得低于 B0 超过 0.1dB；最大绝对误差不得高于 max(1.1×B0 最大绝对误差, 1e-6)。对参考 RMS<1e-8 的 stem 使用 max_abs_error≤1e-6，避免零信号 SNR 失真。
3. 在既有短 golden 上继续满足原 CI 的 SNR≥60dB，并执行更严格的相对 B0 门槛。完整音乐块 B0 总体约 59.35dB，不能错误套用短 fixture 的绝对阈值；必须记录每个 fixture/每个 stem 的 B0。
4. 资源/布局调整在相同后端和算法下要求逐位一致；更换数值核函数则按以上容差验收。既有 kernel parity 门槛保留，不能只通过总体 SNR 就掩盖某个 stem 或边界错误。

### 复现命令契约

以下命令为任务 1 完成后的固定工具接口；路径按隔离构建目录传入，任何秘密均不在命令参数中：

```bash
python tools/benchmark_matrix.py prepare   --reference-root /data/dsh/logic-bs-roformer-rs   --model-dir assets --audio assets/cyberangel.wav   --out /tmp/lbrr-perf-fixtures

./target/release/lbrr --bench --model-dir assets   --bench-ref /tmp/lbrr-perf-fixtures/full/ref_output.npz   --bench-stage waveform --warmup 5 --iters 20   --attn-time cudnn --attn-freq cudnn   --bench-json output/bench-full.json

python tools/benchmark_matrix.py run   --rust-bin ./target/release/lbrr   --reference-root /data/dsh/logic-bs-roformer-rs   --fixtures /tmp/lbrr-perf-fixtures   --rounds 5 --iters 20 --warmup 5   --out output/performance-run
```

## 明确后置的工作与触发条件

- **手写 Flash 重写 / 轴连续 pack / 新 attention tile：** 本轮只保留现有后端作回退与对照。若生产目标确实无法使用 cuDNN，或完成前述工作后 attention 周边仍是主要瓶颈，再单独比较完整 pack+RoPE+SDPA+gate 成本。现有 16.5% occupancy、52.8% L1TEX 等待可作诊断依据，不能预先承诺某种 tile 获胜。
- **cuDNN 每轴在线速度调优及 gate epilogue 融合：** 本轮只实现可靠的按轴选择/准备/workspace，库内核形态保留。只有频率轴或周边成本仍值得投入且能证明完整块 ≥3% 净收益时，才新增 bounded 候选测量。
- **CUDA Graph / 双流 / FFT 合并：** 当前完整块提交间隙仅 0.127%，C2R 约 0.38ms，后置。若优化后 host 提交空洞稳定超过 5%，再评估 Graph；双流必须在 cuFFT 绑定、独立 scratch 与有序 OLA 成立后另行设计。
- **磁盘算法缓存、离线预打包权重、常驻服务/批量 CLI：** 先完成复制/上传清理并量化剩余冷启动；只有加载仍占目标使用场景墙钟 20% 以上时，再为该场景提出独立格式或生命周期方案，避免本轮增加无必要的持久化协议。

本轮成功交付是可重复的完整波形基准、明确后端、显著降低的生产内存、通过数值和收益门槛的算子优化，以及如实记录的延迟/MFU；不把所有实验候选都变成长期维护的默认实现。
