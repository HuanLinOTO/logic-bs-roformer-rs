# logic-bs-roformer-rs

BS-RoFormer 六 stem 音源分离的 Rust + cuda-oxide 实现，支持 Windows 原生和 Linux CUDA。模型权重、训练 YAML 与六个双声道 WAV 的输出格式保持兼容。

## 当前结果

RTX 3080 20GB，完整波形（包括 GPU ISTFT overlap-add），5轮交替配对、每轮预热5次/计时20次：

| 指标 | 同轮 B0 | 最终默认组合 |
|---|---:|---:|
| 588800 samples，T=1151 | 159.79 ms | **157.89 ms** |
| 132300 samples，T=259 | 34.16 ms | **33.81 ms** |
| 完整块生产 scratch | 9.373 GiB | **2.322 GiB** |
| 常驻 GPU 权重 | 996.41 MiB | **336.18 MiB** |
| 进程 GPU 峰值（NVML） | 10.883 GiB | **3.164 GiB** |

完整块配对收益为 **1.19%**；对初始 B0 158.97 ms 的收益为 **0.68%**。原定8%延迟目标未达到。输出在3080正式对照中逐位一致，九个固定 fixture 通过逐 stem 误差门槛。更充分的 GEMM 调优增加了冷启动时间，进程墙钟中位数约5.49→6.19秒；它与热推理延迟分别报告。

方法、候选拒绝原因、冷启动、MFU和原始结果见[性能验收报告](docs/BENCHMARK.md)。Windows及WSL已完成编译、短golden和边界检查；本地繁忙时仅作数值验证，正式性能结论来自3080。

## Windows 原生构建与运行

[编译器子模块](vendor/cuda-rust) 使用 [cuda-rust-windows fork](https://github.com/ansidium/cuda-rust-windows)，现有 stable 1.99 工具链与构建入口保持不变。

~~~powershell
powershell scripts/build_win.ps1
powershell scripts/lbrr.ps1 --input song.mp3 --outdir separated   # 分离（--input 不带模式标志时默认进入）
powershell scripts/lbrr.ps1 --separate --model-dir assets --input song.wav --outdir separated
powershell scripts/lbrr.ps1 --bench --model-dir assets --warmup 5 --iters 20 --bench-json output/short.json
~~~

输入接受 wav 及任意 ffmpeg 可解码格式（mp3/flac/ogg 等）：非 44.1kHz/stereo 的输入自动经 ffmpeg 转成临时 wav，读完即删（ffmpeg 取自 PATH 或 LBRR_FFMPEG）。

运行库布局与 shim 编译说明在[Windows 构建脚本](scripts/build_win.ps1)。现有本机 cuDNN/cfe/shim 位于 D:/Projects/lbrr-win-libs。

依赖与缓存沿用 D 盘约定：CARGO_HOME=D:/cargo、RUSTUP_HOME=D:/rustup。禁止把新增依赖安装到 C 盘；构建脚本会拒绝指向 C 盘的 Rust 工具链/缓存路径。

## 可复现基准与后端选择

~~~bash
lbrr --bench --model-dir assets --bench-ref fixtures/full/ref_output.npz --bench-stage waveform --warmup 5 --iters 20 --attn-time cudnn --attn-freq cudnn --ff1-backend handwritten --qk-rope split --bench-json output/full.json
python tools/check_benchmark.py output/full.json --baseline output/b0-full.json
lbrr --kernel-regression --bench-json output/kernels.json
~~~

- 默认测完整 GPU 波形；frames 模式只用于历史 C2R 帧边界复核，JSON明确标注计时终点。RTF根据真实样本数/采样率计算。
- 时间轴和频率轴独立选择 auto/cudnn/handwritten。优先级是显式CLI > LBRR_NO_CUDNN > auto；环境变量保留“存在即禁用”的旧语义。auto在准备阶段按轴回退并记录原因；显式cuDNN失败返回错误。执行错误会中止该次推理。
- FF1支持 auto/handwritten/cublaslt-erf/handwritten-async。实测后默认保留 handwritten；候选没有通过收益门槛。原有erf形式GELU和FP16舍入边界不变。
- Q/K RoPE支持 split/fused。默认split；广播gate没有收益，生产路径保留原gate pass。手写注意力保持原有Q旋转语义。
- 结果记录源码/二进制/模型/配置/参考哈希、实际后端、初始化细项、CUDA Event/host时间、内存和逐stem误差。算法只在进程内缓存，不跨库版本保存opaque数据。

固定样本生成和五轮配对入口在[benchmark_matrix.py](tools/benchmark_matrix.py)：

~~~bash
python tools/benchmark_matrix.py prepare --reference-root /path/to/pymss --model-dir assets --audio assets/cyberangel.wav --out fixtures
python tools/benchmark_matrix.py run --rust-bin /path/to/b0 --candidate-bin ./target/release/lbrr --model-dir assets --fixtures fixtures --rounds 5 --warmup 5 --iters 20 --cases full,short --out output/paired
~~~

## Linux、WSL与隔离目录

[WSL同步脚本](scripts/sync_wsl.sh) 支持 LBRR_SOURCE_DIR/LBRR_WSL_ROOT，[构建脚本](scripts/build_wsl.sh)支持 CARGO_TARGET_DIR。同步前验证目标是指定任务目录，保留真实的编译器源码目录，排除构建缓存。既有 Linux nightly 编译环境可用 LBRR_VENDOR_SOURCE 指定原编译器源码；这不修改 Windows fork 或工具链 pin。

~~~bash
export LBRR_SOURCE_DIR=/mnt/d/Projects/logic-bs-roformer-rs
export LBRR_WSL_ROOT=/root/work/lbrr-isolated
export CARGO_TARGET_DIR=/root/work/target-lbrr-isolated
bash scripts/sync_wsl.sh
LBRR_CARGO_ACTION=build bash scripts/build_wsl.sh -- --release
~~~

[远程构建脚本](scripts/build_remote.sh)支持 LBRR_REMOTE_ROOT/CARGO_TARGET_DIR，默认复用既有 CUDA13.3、LLVM19 与 nightly-2026-08-28 环境。运行时通过 LBRR_CUDNN_DIR、LBRR_NVRTC_DIR、LBRR_SDPA_WRAP 指向本机准备好的库；不把训练配置的 flash_attn 映射成 Rust 后端选择。

## 模型与实现

模型固定为174.66M参数、1989个checkpoint key、62频带、12个双轴层、8heads×64、dim256/FF1024、6stems。加载保留dtype/shape/key完整性校验。

| 文件 | 职责 |
|---|---|
| [main.rs](src/main.rs) | GPU内核、生产流水线、独立parity资源、CLI接线 |
| [inference_options.rs](src/inference_options.rs) | 后端请求、优先级与保守默认 |
| [benchmark.rs](src/benchmark.rs) | 完整波形测量契约、数值指标、MFU和结果身份 |
| [cublaslt.rs](src/cublaslt.rs) | 非别名调优、描述符缓存、资源生命周期 |
| [cudnn.rs](src/cudnn.rs) | 按轴准备、完整缓存键、workspace与明确失败语义 |
| [stft.rs](src/stft.rs)、[cufft.rs](src/cufft.rs) | 单流FFT计划和执行 |
| [weights.rs](src/weights.rs) | 权重加载与完整校验 |
| [bench_ref.py](tools/bench_ref.py)、[separate_ref.py](tools/separate_ref.py) | PyTorch完整波形与整曲参考 |

生产 scratch 与诊断资源分离，GPU只保留实际使用的矩阵表示。MaskEstimator使用真实 band 宽度，pre2/GLU总宽为8200/4100；157项静态tile表复用于6个stem。

## 验证与CI

[本地CI](scripts/ci.sh)运行GPU smoke、核函数边界回归、后端生命周期单元测试以及完整波形JSON门槛。LBRR_MODEL_DIR可指定模型目录，LBRR_CI_BASELINE_JSON可启用相对B0逐stem门槛。

[GitHub workflow](.github/workflows/ci.yml)在托管 Linux/Windows runner验证构建；GPU步骤通过设备探测启用。缺GPU或权重会明确说明资源型检查被跳过。最终性能验收必须使用资源齐备的机器。

已有 --e2e-test、--stft-test、--attn-test 等诊断入口保留，所需参考可用[dump_refs.py](tools/dump_refs.py)生成。新增 --kernel-regression 不依赖模型权重，覆盖RoPE、FFT/OLA、Mask映射、FF1尾行和调优输入保护。
