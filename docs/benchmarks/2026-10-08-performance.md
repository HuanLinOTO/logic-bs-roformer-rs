# 性能复核与优化：Windows RTX 4060 Ti（2026-10-08）

## 结论

本轮确定收益在 **CPU 权重转换、音频 I/O 和端到端启动**，没有宣称 GPU 热推理加速。

| 同条件测量 | B0 | 优化后 | 变化 |
|---|---:|---:|---:|
| 生产 FP16 权重转换，短块进程，5轮中位数 | 343.17 ms | 78.48 ms | 减少 77.13% |
| 生产 FP16 权重转换，完整块进程，5轮中位数 | 380.32 ms | 89.94 ms | 减少 76.35% |
| 转换+上传+释放 host 权重，短块 | 520.75 ms | 248.74 ms | 减少 52.23% |
| 转换+上传+释放 host 权重，完整块 | 575.43 ms | 284.91 ms | 减少 50.49% |
| 全 checkpoint payload CPU pack，7轮 | 393.56 ms | 93.22 ms | 4.22×，减少 76.31% |
| 单个 63.67 MB F32 WAV 写入，7轮 | 63.39 ms | 38.32 ms | 减少 39.55% |
| 同一 WAV 读取，7轮 | 55.41 ms | 41.47 ms | 减少 25.17% |
| 180.48秒整曲进程墙钟，固定 GEMM，3轮 | 6.888 s | 6.417 s | 减少 6.85%，3/3轮同向 |
| 同一整曲 GPU 阶段（host wall） | 4.68 s | 4.68 s | 中位数持平 |
| 默认后端短块 GPU Event，5轮 | 58.028 ms | 57.971 ms | 约0.10%，不计为收益 |
| 默认后端完整块 GPU Event，5轮 | 291.241 ms | 290.793 ms | 约0.15%，不计为收益 |

**限制：整曲的6.85%来自双方都设置 `LBRR_NO_LT=1` 的固定手写GEMM路径，不是默认cuBLASLt整曲性能结论。** 该设置用于隔离本次CPU/I/O变更，默认后端没有修改。

可审阅的汇总证据在 [2026-10-08-results.json](<D:/Projects/logic-bs-roformer-rs/docs/benchmarks/2026-10-08-results.json>)；其包含每轮值、全部身份哈希、输出哈希和测量局限。

## 为什么不直接相信旧 benchmark

1. **现成二进制不是当前HEAD构建。** 初始二进制嵌入revision `3b3befd`，当前HEAD为`b3c78c8`。本轮先不修改代码重编译，保存B0，再测候选。两个基线的模型相关source digest相同，说明后续提交未改变该digest覆盖的模型源码，但不能只凭文件名假定对应。
2. **不同平台/库版本不能共用性能基线。** 旧报告3080/Linux数据不是当前4060Ti/Windows结果。当前cuDNN=92700，cuBLASLt=130800；运行时报告13030，尽管toolkit目录叫v13.4，不能把目录名写成实际runtime版本。
3. **历史墙钟不能替代当前复测。** 旧Windows验证在当前WSL复测与历史不符时仍采用历史值判通过，本轮不沿用这个gate。
4. **完整块参考不是独立FP32 golden。** 本轮所用 [full-b0-ref.npz](<D:/Projects/logic-bs-roformer-rs/output/performance-optimization/resource-checks/full-b0-ref.npz>) 是旧B0输出，只能检查回归。其逐位相同时SNR会显示约3045dB；这是`noise.max(1e-300)`造成的数值表示，不是模型有3045dB精度，应读`bit_equal`和max_abs_error。短块使用既有FP32参考；整体SNR为80.934–80.961dB，不等于每个近静音stem也有80dB。
5. **进程内cuBLASLt速度调优会改变跨进程算术。** B0短块五轮已经产生两种输出哈希，候选恰好是这同两种；完整块的一个候选产生另一种输出，但与此次初始B0复测逐位相同。因此默认A/B并非每对都逐位相同，不能一概声称“全部bit-exact”。本轮以实际checkpoint转换逐位相同、固定算术整曲逐位一致及已观察B0输出哈希验证无新增算术变化。没有跨版本持久化或硬编码算法索引。

## 热点：实际Nsight证据

完整块最后一次热forward的GPU kernel合计 **292.689 ms**，span **293.118 ms**，空隙 **0.428 ms / 0.146%**。这份profile解释热点，不用于最终收益百分比。

| 热点 | ms / block | 占kernel时间 |
|---|---:|---:|
| cuDNN双轴SDPA | 75.62 | 25.84% |
| QKV GEMM | 36.46 | 12.46% |
| FF1 + erf GELU | 36.92 | 12.62% |
| FF2 GEMM | 29.97 | 10.24% |
| 输出投影GEMM | 22.18 | 7.58% |
| 两个RoPE pass | 29.96 | 10.24% |
| SDPA gate | 13.53 | 4.62% |
| 两个norm系列 | 22.86 | 7.81% |

提交空隙很小，CUDA Graph只能覆盖其中一小部分，不是当前优先项。SDPA、GEMM和memory passes仍是GPU主要瓶颈；本轮没有改precision、GELU、归约顺序、stream或vendor。

既有Q/K fused做了三轮交替筛选：短块58.196→58.271ms（慢0.129%），完整块291.302→291.152ms（约快0.051%），均不具备采用价值，保留split默认。跨进程bit不一致与调优选择混杂，不据此认定RoPE核函数本身错误；独立RoPE/gate回归是逐位一致的。FF1诊断中M=71362原核1.457ms、async1.755ms、cuBLASLt-erf含pack3.182ms，也不切换默认。

## 已落地变更

- [fp16.rs](<D:/Projects/logic-bs-roformer-rs/src/fp16.rs>)：host FP32→FP16采用运行时AVX/F16C检测，每次转换8个f32；保持nearest-even、sign、subnormal、NaN旧位约定和f16x2顺序。对不支持的CPU使用原标量转换。输出直接初始化Vec spare capacity，不额外zero-fill。
- [main.rs](<D:/Projects/logic-bs-roformer-rs/src/main.rs>)：生产upload_weights和诊断pack共同使用该模块，不再保留重复标量packing。GPU内核未改变。
- [audio.rs](<D:/Projects/logic-bs-roformer-rs/src/audio.rs>)：read_wav借用data切片，不复制整份payload；little-endian写header后直接写f32 bytes，消除一个与WAV同大小的序列化Vec。big-endian保留16KiB有界转换；创建文件前校验RIFF长度、声道、采样率和完整frame。

该整曲每个stem为63,673,388字节；新写入不再为每个stem另分配约60.72MiB的字节副本。这是由分配路径确定的削减，不冒充已实测进程RSS峰值。输入读取也不再复制data chunk。GPU scratch和权重驻留量未变。

## 正确性与构建

- current HEAD B0、候选、最终格式化源码均通过Windows release构建；最终源码重新计算digest与嵌入digest一致。最终binary SHA256 `d9a544e6c27ceb9a73ac04476c2e5fa6a17888bd68e33ab4c0671c4bf6f2f956`。
- **16项unit tests全部通过**，包括新增6项：IEEE float WAV byte compatibility/roundtrip、PCM16/24、非法metadata；FP16所有normal half值及tie附近、subnormal、20万随机bit、非对齐切片/尾部、多parts、奇数长度拒绝。
- 对checkpoint payload共 **174,702,068个f32** 比较旧标量和新pack，全部u32 words一致；此计数包括checkpoint中的重复bias，并不等于去重的模型参数数。
- GPU `--kernel-regression`通过，含RoPE/gate T=1/37/62/63/64/65/259/1151，非默认stream STFT/OLA，compact Mask全部band/stem，FF1 M=1/37/127/129/16058/71362和调优输入保护。
- 固定GEMM下短块、完整块B0/候选waveform逐位一致；最终格式化构建完整块也与固定B0逐位一致。
- 180.48秒真实整曲三轮配对，每轮六个双声道F32 WAV **整文件逐位相同**，每个15918336个samples、均finite，总输出382040328字节，输入长度和尾部保留。
- 默认后端最终构建短golden整体SNR80.961dB；没有新算法或降低精度。其他CPU架构和big-endian分支未在本机执行，保留明确fallback但不宣称跨平台实测。

## 测量方法和波动

CPU Intel Core i5-12400F（6C/12T），GPU RTX4060Ti16GB SM8.9，Windows11 WDDM，driver610.88，stable Rust1.99.0。不锁频、不改功耗，桌面程序保留。

默认后端short/full各5轮交替A/B，每次独立进程prepare，warmup5，Event20次；输入驻留GPU，计时STFT→GPU ISTFT waveform，不含H2D/D2H和文件I/O。轮间Event CV：short0.50%/0.37%，full1.53%/1.10%。零点几个百分点不做收益宣称。

默认benchmark进程墙钟short中位4.168→3.859s、full11.804→11.549s；short候选首进程6.328s离群（五轮CV22.5%，包含file/JIT/hash等启动干扰），因此不把7.42%作为稳定的冷启动承诺。整曲固定算术三轮的进程墙钟CV：B0约0.38%，candidate约0.65%，3/3轮同向，可支持这组6.85%的端到端结论。

Host微基准7轮交替，包含分配和文件写入，不调用fsync、不清OS cache。不是durable-write磁盘带宽或首次安装冷启动。最终格式化构建仅做smoke/数值复核，没有用其单次56.13ms替换五轮统计。

## 复现

保持现有环境，先构建和保存未优化B0，再构建候选。全部新实测文件在output/perf-2026-10-08，不覆盖用户已有输出。

```powershell
# Windows构建，复用已准备的shim与D盘工具链
powershell scripts/build_win.ps1 -SkipShim

# host微基准旧I/O源码快照；需在本次变更之前的revision上生成
New-Item -ItemType Directory -Force output/perf-2026-10-08 | Out-Null
git show b3c78c8:src/audio.rs | Set-Content -Encoding utf8 output/perf-2026-10-08/audio_baseline.rs
D:/cargo/bin/rustc.exe --edition=2024 -O tools/bench_host.rs -o output/perf-2026-10-08/bench-host.exe
./output/perf-2026-10-08/bench-host.exe

# 不需要CUDA的新增单元回归
D:/cargo/bin/rustc.exe --edition=2024 -O --test src/fp16.rs -o output/perf-2026-10-08/fp16-tests.exe
./output/perf-2026-10-08/fp16-tests.exe
D:/cargo/bin/rustc.exe --edition=2024 -O --test src/audio.rs -o output/perf-2026-10-08/audio-tests.exe
./output/perf-2026-10-08/audio-tests.exe

# 完整模型独立short golden + GPU边界回归
powershell scripts/lbrr.ps1 --bench --model-dir assets --warmup 5 --iters 20 --bench-json output/perf-2026-10-08/recheck-short.json
powershell scripts/lbrr.ps1 --kernel-regression --bench-json output/perf-2026-10-08/recheck-kernels.json

# 与本轮固定算术整曲口径一致；基线和候选双方都设置
$env:LBRR_NO_LT = "1"
powershell scripts/lbrr.ps1 --separate --model-dir assets --input assets/cyberangel.wav --outdir output/perf-2026-10-08/recheck-song
Remove-Item Env:LBRR_NO_LT
```

原始证据：[五轮配对脚本](<D:/Projects/logic-bs-roformer-rs/output/perf-2026-10-08/paired.py>)、[Q/K筛选脚本](<D:/Projects/logic-bs-roformer-rs/output/perf-2026-10-08/screen_rope.py>)、[固定算术整曲脚本](<D:/Projects/logic-bs-roformer-rs/output/perf-2026-10-08/separate_pair.py>)、[GPU profile汇总](<D:/Projects/logic-bs-roformer-rs/output/perf-2026-10-08/profile-summary.json>)、[Nsight SQLite时间线](<D:/Projects/logic-bs-roformer-rs/output/perf-2026-10-08/profile-baseline.sqlite>)、[host微基准](<D:/Projects/logic-bs-roformer-rs/tools/bench_host.rs>)。汇总JSON随文档保存，不要求读者相信旧报告或只看一个漂亮的百分比。
