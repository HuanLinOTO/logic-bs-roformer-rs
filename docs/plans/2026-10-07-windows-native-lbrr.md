# Windows 原生运行 lbrr（保证最优速度）实施计划

日期：2026-10-07 ｜ 状态：待批准

## 目标与成功标准

lbrr.exe 在 Windows 11 原生运行（无 WSL 依赖），且速度不低于现有 WSL 基线：

| 指标 | 基线（WSL, nightly-2026-08-28） | 成功标准 |
|---|---|---|
| 整曲 GPU wall（cyberangel.wav 180.5s） | 5.92s | ≤ 5.92s |
| golden bench SNR | 80.83 dB（gate 60） | ≥ 60 dB，记录新基线 |
| 整曲输出 vs WSL 产物 SNR | —（跨运行噪声 ~70.4dB） | ≥ 70 dB |
| 短输入（8s） | 42.1% 非零 | 非零，无静音尾 |

## 调研结论（已完成，全部实测/一手验证）

1. **障碍真相**：README 记录的 LNK1189（>65535 导出）只挡"Windows 上构建 rustc_codegen_cuda backend"，不挡运行。运行时链路（驱动 API dlopen、PTX 驱动 JIT）平台中立。
2. **Windows fork 已就绪**：[ansidium/cuda-rust-windows](https://github.com/ansidium/cuda-rust-windows)，与上游同步至 `a0cc6cc`——**恰为我们 vendor/cuda-rust 当前基线 commit**（drift = 0），fork 在此之上 +101 commits 纯 Windows 移植。
3. **fork 修复覆盖我们实测的全部编译障碍**：本机 plain `cargo check` 卡在 cuda-core 18 处 u32/i32 类型错误（api.rs / simt/{context,event,module,stream,mod}.rs）——正是 fork patch 的文件清单。
4. **运行时 embedded 加载已平台中立**：fork 的 `cuda-core/src/simt/embedded.rs` `artifact_bundles_from_current_exe()` = 读 exe 文件字节 + 对象解析，零平台 cfg；宏生成的 load 在非 Linux 分支自动走 `load_embedded_module`（cuda-macros/src/cuda_module/mod.rs:178 有测试断言）。
5. **toolchain 变化**：fork 将 backend 适配到 **stable Rust 1.99**（摆脱 nightly pin）。我们业务代码（src/）零 nightly feature 声明，edition 2024 stable 兼容。代价：MIR importer 变化 → PTX 生成链变化 → golden bench 必须重验并记录新基线。
6. 本机环境：MSVC（F:\vs）、CUDA Toolkit 12.5 + 13.4、driver 610.88（4060 Ti sm_89）、ffmpeg.exe 在 PATH、vcpkg 在 D:\Projects\vcpkg。
7. 依赖项：fork Windows CI 需 `libffi:x64-windows`（vcpkg 装）；fork release（windows-v0.2.1，6 月）比 main 旧，**工具链从 fork main 自建**。

## 架构决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| vendor 来源 | 切 ansidium/cuda-rust-windows @ main | 基线相同 drift=0；自建 Windows 移植成本远高于复用 |
| Rust toolchain | stable（按 fork pin） | fork 要求；backend 已适配 stable |
| 构建宿主 | Windows 本机全量构建（cargo oxide build --release） | 不再依赖 WSL 构建；WSL 侧保持现状作为 A/B 基准（暂不切） |
| 设备代码形态 | PTX embedded（驱动 JIT sm_89）为默认 | 现状架构不变；若首次加载 JIT 耗时显著，评估 `--materialize-cubin --arch sm_89`（fork CI 已产 cubin） |
| 运行库 | cublasLt/cufft 用本机 CUDA 13.4；cuDNN 用 pip `nvidia-cudnn-cu13` win wheel 提取；SDPA shim 用 MSVC 重编 .dll | 就地取材，零下载成本 |

## 实施阶段

### 阶段 0：工具链冒烟（Gate：vecadd 实跑）～1h
1. clone fork 到 `D:\Projects\cuda-rust-win`（repo 外，不进 git）
2. `rustup toolchain install` 按 fork rust-toolchain.toml；`vcpkg install libffi:x64-windows`
3. 按 fork 的 `cuda-oxide/cuda-oxide-book/getting-started/windows.md`：release profile 构建 codegen backend + `cargo oxide build vecadd --arch sm_89`
4. **实跑 vecadd**（fork CI 是 no-GPU canary，我们有 4060 Ti，验证 COFF .oxart 解析 → 驱动 JIT → launch 全链路）
5. 失败则查 fork issues / 定位；这是全计划唯一的未知深水区

### 阶段 1：vendor 切换 + lbrr 编译
1. `vendor/cuda-rust` 加 remote 指向 fork，checkout fork/main，记录 commit 到 `docs/vendor-pins.md`
2. repo `rust-toolchain.toml`：nightly-2026-08-28 → fork 要求的 stable
3. `cargo oxide build --release`（Windows），修少量编译差异（预期：fork Windows 改动引入的 API 微调；基线相同应量小）
4. Gate：lbrr.exe 产出

### 阶段 2：运行时库贯通
1. dlopen 候选名单加 Windows（`src/cublaslt.rs`、`src/cufft.rs`、`src/cudnn.rs`）：`cublasLt64_*.dll`、`cufft64_*.dll`、`cudnn64_9.dll` + 现有 CUDA_HOME 遍历逻辑加 `bin` 子目录（~30 行）
2. cuDNN：`pip download nvidia-cudnn-cu13 --platform win_amd64` 解出 DLL → `D:\Projects\lbrr-win-libs\cudnn\bin`，`LBRR_CUDNN_DIR` 指向
3. SDPA shim：取 WSL `/opt/lbrr-cudnn/cfe/` 源码 → MSVC (`cl /LD`) 编 `cudnn_sdpa_wrap.dll`；若系 cudnn frontend C++ 且受阻，先无 shim 跑通并量化 fallback 损失，再决定是否重写为纯 C graph API
4. NVRTC：`nvrtc64_*.dll`（CUDA 13.4 bin 已有）
5. Gate：`lbrr.exe` golden 子命令跑通

### 阶段 3：正确性 + 性能验证
1. golden bench：SNR ≥ 60 dB，记录 stable 工具链新基线
2. 整曲 A/B：Windows 产物 vs WSL 产物 SNR ≥ 70 dB；短输入（8s）无静音尾
3. 性能：整曲 GPU wall vs 5.92s；分解项：SDPA shim 有/无、模型冷/热加载、（必要时）PTX JIT 一次性耗时 → 决定是否 materialize-cubin
4. 不达标则 nsys/ncu（Windows 版）定位；回退基准：WSL 侧未动，随时可退

### 阶段 4（验证通过后）：工程化
- `scripts/build_win.ps1`（环境 + 构建一条龙）；`lbrr.ps1` 用户入口（路径友好）
- CI 加 windows job（gh runner no-GPU compile canary + `lbrr-win-x64` artifact）
- WSL 侧统一切 fork（Linux 上游兼容）——统一 vendor 后删 sync 分叉逻辑

## 风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| fork stable backend 在我们 kernels 上有 bug | 中 | golden bench 全量验证；WSL 基线不动可随时回退 |
| COFF .oxart bundle 解析缺 case | 低 | 阶段 0 vecadd 实跑即暴露 |
| SDPA shim MSVC 编译（C++ frontend） | 中 | 分级：先量化无 shim 损失再决定投入；最坏重写纯 C |
| stable MIR importer 致 PTX 质量变化 | 低-中 | bench 数字说话；SNR gate 60dB 留有余量 |
| libffi/MSVC 链接环境 | 低 | vcpkg 已就位；fork CI 同路径已验证 |

## 假设

- 本机 CUDA 13.4 + MSVC + driver 610.88 满足 fork 要求（fork 声明支持 Toolkit 12.x/13.x）
- lbrr 业务代码无隐藏平台依赖（已 grep 验证：零 unix-only 调用、零 nightly feature）
- ffmpeg 查找逻辑已原生支持 Windows（find_ffmpeg 含 ffmpeg.exe 分支）

## 交付物

`lbrr.exe`（Windows x64）+ `docs/vendor-pins.md` + 验证数据表（SNR/perf 对比）+ 可选 CI windows job
