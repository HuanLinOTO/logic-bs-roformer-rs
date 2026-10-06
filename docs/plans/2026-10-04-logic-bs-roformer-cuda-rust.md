# logic-bs-roformer-rs — BS-RoFormer 六轨分离模型的纯 Rust + cuda-oxide GPU 实现

## 1. 目标

为 pymss 生态中的 `logic_bs_roformer` 模型（六轨音乐分离：bass/drums/other/vocals/guitar/piano）编写独立的 Rust 推理实现：

- **底层库**：NVIDIA 官方 [cuda-rust](https://github.com/NVIDIA/cuda-rust)（cuda-oxide：纯 Rust 单源 CUDA kernel，`#[kernel]` → rustc 后端 → NVVM → PTX）
- **专门算子**：针对该模型的固定形状（62 频带、dim 256、12 层双轴 transformer、6 stems）手写融合 kernel，目标是超过 PyTorch fp32 通用路径
- **双机验证**：本机 RTX 4060 Ti 16G（sm_89，经 WSL2）+ 远程节点 <REMOTE_NODE_NAME>（RTX 3080 20G，sm_86，Debian 13）
- **数值正确性**：与 PyTorch 参考实现逐算子对拍 + 端到端对拍

### 成功标准（验收）

1. `cargo test` 全绿：每个算子 vs PyTorch 参考输出 rel-err < 1e-4（融合算子 < 5e-4，fp32）
2. 端到端：同一 3s 合成测试音频，6 个 stem 输出与 PyTorch fp32 参考 SNR ≥ 60 dB；真实音乐听感无退化
3. 性能：两机端到端 RTF 均优于同机 PyTorch fp32 基线 ≥ 1.5×（目标 3×）。基线已测：4060Ti + torch 2.6 cu124 fp32（math SDPA）= 205 ms / 3s 音频，RTF 0.068
4. CLI 可用：`lbrr --model-dir assets --input song.wav --outdir out/` 产出 6 个 stem wav

## 2. 已验证的环境事实（2026-10-04 实测）

### 本机（Windows 11）
- RTX 4060 Ti 16G（sm_89），CUDA **12.5** toolkit（`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5`），driver 596.49
- rustc 1.97.1 stable；**nightly-2026-08-28 已装**（rsproxy 镜像可用，`RUSTUP_DIST_SERVER=https://rsproxy.cn`）
- ⚠️ **Windows 原生无法构建 cuda-oxide 编译器栈**：MSVC 链接器 `LNK1189`（COFF 库 65535 对象上限，rustc_codegen_cuda 超限）。本机 GPU 开发一律走 WSL
- WSL2 Ubuntu-22.04 已修复（unregister 后 `wsl --install -d Ubuntu-22.04 --no-launch`，root 直入），**4060Ti CUDA 直通正常**；**CUDA 12.6 toolkit（/usr/local/cuda-12.6）+ rustup（stable + nightly-2026-08-28）+ libclang 已装好并验证**（脚本 `D:\Projects\.wsl-setup\setup.sh`）
- uv + PyTorch 2.6.0+cu124 参考环境就绪：`.venv2`（`D:\Projects\.model-research`），已跑通参考推理
- 权重/YAML/参考输出已下载：`D:\Projects\.model-research\{logic_bs_roformer.ckpt, logic_bs_roformer.yaml, ref_output.npz, state_dict_shapes.json}`

### 远程测试节点（Komari 节点 <REMOTE_NODE_NAME>，uuid <KOMARI_NODE_ID>）
- Debian 13，64 核 E5，62G RAM，RTX 3080 **20G**（sm_86），driver 595.91，CUDA **13.3** 完整 toolkit（/usr/local/cuda-13.3）
- rustc 1.98.0 stable + rustup；**nightly-2026-08-28 已装**；已补装 libclang-dev、libcurand-dev-13-3
- **cuda-oxide 全链路已验证成功**：`cargo oxide run vecadd` 在 3080 上正确执行（cuda-rust 已 clone 在 /data/dsh/cuda-rust）
- ⚠️ 只能通过 Komari 面板远程执行（无直连 SSH）：封装工具 `<SSH_TOOLS_DIR>/kexec.js`（登录 → 面板 exec API → 轮询结果，root 权限）。长任务模式：`nohup cmd > log 2>&1 &` + 轮询 `cat log`
- 磁盘 /data 261G 可用；miniconda3 在 /data/dsh/miniconda3（可用于远程装 torch 参考基准）

### 其他
- hf-mirror.com 双机可达（权重已下过一次，699MB）
- pymss-core 源码（参考实现）：本地 `D:\Projects\.pymss-core-research`
- cuda-rust 源码（研读用）：本地 `D:\Projects\.cuda-rust-research`

## 3. 模型规格（logic_bs_roformer，已从 YAML + 源码 + 权重三重确认）

BS-RoFormer（bandsplit roformer），174.66M 参数，state_dict 1989 键，stereo，6 stems。

### 超参
| 项 | 值 |
|---|---|
| dim / inner | 256 / 512（heads 8 × dim_head 64） |
| depth | 12（每层 = time transformer(depth 1) + freq transformer(depth 1)） |
| 频带 | 62 bands：2×24, 4×12, 12×8, 24×8, 48×8, 128, 129（和 = 1025 = n_fft/2+1） |
| STFT | n_fft 2048, hop 512, win 2048, hann, center, 非 normalized |
| MaskEstimator | 每 stem 62 个 per-band MLP：Linear(256→1024)+Tanh+Linear(1024→dim_in·2)+GLU |
| BandSplit | 每 band RMSNorm + Linear(dim_in→256)，dim_in = 2·freqs·2(channels) |
| 共享偏置 | use_shared_bias=True：`linear_62_bias_0`[1536]（qkv）、`linear_64_bias_0`[256]（out proj），全部 24 个 transformer 共享 |
| 推理分块 | chunk C=588800，overlap 29440，step=559360，border/fade=29440，reflect pad |

### 前向数据流（batch=1，T=floor(588800/512)+1=1151）
```
raw (1,2,588800)
→ STFT/hann            → (1, 2050, 1151, 2)   # 2050 = 1025 freq × 2 ch，interleave (f,s)
→ permute/reshape      → (1, 1151, 4100)      # 实虚交错
→ BandSplit ×62        → (1, 1151, 62, 256)   # 每 band RMSNorm(F.normalize·√d·γ)+Linear，62 组 GEMM
→ ×12 层:
    time transformer:  (62, 1151, 256) 折叠 batch → RMSNorm→QKV(256→1536,+shared bias)
                        →RoPE(交错布局,cos/sin 成对)→SDPA(8×64,scale=8⁻¹ᐟ²)→gates(sigmoid·to_gates)
                        →out proj(512→256,+shared bias)→残差→FF(RMSNorm→256→1024→GELU→1024→256)→残差→RMSNorm
    freq transformer:  同上，序列轴为 62（batch 折叠 1151）
→ final RMSNorm        → (1, 1151, 62, 256)
→ MaskEstimator ×6 stems（可打包 grouped GEMM）→ (1, 6, 1151, 2050)
→ reshape              → (1, 6, 1025, 1151, 2) 复数掩码
→ 复数乘 mix STFT      → 6 个掩码谱
→ ISTFT ×6（hann 窗）  → (1, 6, 2, 588800)
```

### 推理编排（overlap-add，与 pymss demix_track 语义一致）
- mix 前后 reflect-pad border=29440；starts=range(0, padded, step)；首 chunk fade-in 置 1，末 chunk fade-out 置 1
- 窗口 = linspace fade-in(前 29440) + 1(中) + fade-out(后 29440)；`result += out·w`，`counter += w²`，最终 `result/counter` 去 pad
- 输出音质：训练 config `normalize: false`，不做响度归一（保持与 pymss 默认一致）

### state_dict 键映射（Rust 侧加载表）
```
linear_62_bias_0 [1536] / linear_64_bias_0 [256]            → 共享 qkv/out 偏置
layers.{L}.{A}.layers.0.0.{norm.gamma,to_qkv.weight[1536,256],to_qkv.bias,to_gates.weight[8,256],to_gates.bias,to_out.0.weight[256,512],to_out.0.bias,rotary_embed.freqs[32]}
  L∈[0,12), A∈{0=time,1=freq}；to_qkv.bias 与 to_out.0.bias 即共享偏置的引用（值相同）
layers.{L}.{A}.layers.0.1.net.{0.gamma[256],1.weight[1024,256],1.bias,4.weight[256,1024],4.bias}   → FF
layers.{L}.{A}.norm.gamma[256]                                                              → Transformer 尾 RMSNorm
final_norm.gamma[256]
band_split.to_features.{b}.0.gamma[dim_in] / .1.weight[256,dim_in] / .1.bias[256]
mask_estimators.{s}.to_freqs.{b}.0.{0.weight[1024,256],0.bias,2.weight[dim_in*2,1024],2.bias}
```
RoPE 常量 freqs[32] 在设备侧重建（1/10000^(2i/64)），pos=arange(seq)，成对 repeat_interleave 后 cos/sin 取偶数位；旋转为交错布局 `(even·cos−odd·sin, odd·cos+even·sin)`。

## 4. 架构与算子设计

### 4.1 工程结构（standalone crate，随 cuda-oxide examples 模式）
```
logic-bs-roformer-rs/            # git 仓库 = 工作目录 D:\Projects\logic-bs-roformer-rs
├── Cargo.toml                    # [workspace]=standalone；path 依赖 vendor/ 下 cuda-rust crates
├── rust-toolchain.toml           # nightly-2026-08-28 + rust-src/rustc-dev/llvm-tools
├── vendor/cuda-rust/             # git submodule，锁 commit（两机 git submodule update --init）
├── src/
│   ├── main.rs                   # CLI：--model-dir/--input/--outdir/--device/--bench/--stems
│   ├── config.rs                 # 解析 logic_bs_roformer.yaml（serde_yaml，容忍 python/tuple 标签）
│   ├── weights.rs                # safetensors → 平铺 GPU 权重缓冲（一次 H2D，常驻显存）
│   ├── model.rs                  # 前向编排（层循环、tensor 布局、stream 同步）
│   ├── stft.rs                   # STFT/ISTFT 编排：frame/window kernel + cufft 计划复用
│   ├── audio.rs                  # wav 读写（hound）+ chunk/overlap-add 编排（GPU 上）
│   └── kernels/mod.rs            # #[cuda_module] 全部 kernel 的宿主
├── kernels/                      # device 代码（被 kernels/mod.rs include）
│   ├── frame.rs                  # 加 hann 窗 + 分帧装载（STFT 前处理） / overlap-add 缩放（ISTFT 后处理）
│   ├── bandsplit.rs              # 62-band fused RMSNorm+Linear（grouped GEMM，一次 kernel）
│   ├── gemm.rs                   # 通用 GEMM：Phase A SIMT tiled fp32 → Phase B mma.sync m16n8k16 + ldmatrix + swizzle smem
│   ├── rope_qkv.rs               # fused QKV-GEMM + RoPE + scale（time/freq 两个变体）
│   ├── attn.rs                   # freq 轴：短序列(62)直接 materialize SDPA；time 轴：flash 风格 tile online-softmax（1151）
│   ├── gate_out.rs               # fused gates-sigmoid + out-proj + 残差
│   ├── ff.rs                     # fused FF GEMM（GELU 融合进 epilogue）
│   ├── maskest.rs                # 6 stems × 62 bands 打包 grouped GEMM（Tanh+GLU 融合）
│   ├── cmask.rs                  # 复数掩码乘（stft × mask）+ reshape
│   └── rmsnorm.rs                # RMSNorm（fp32 F.normalize 等价路径）
├── sys/cufft/                    # cufft-sys：dlopen libcufft（模仿 libnvvm-sys 的 RAII 绑定写法）
│   └── lib.rs                    # cufftPlanMany/cufftExecC2R/R2C/cufftDestroy，最小函数集
├── tools/
│   ├── export_weights.py         # ckpt → safetensors（本机 .venv2 跑一次）
│   ├── dump_refs.py              # 每算子固定随机输入 → PyTorch 输出 npz（parity 金标准）
│   └── bench_ref.py              # PyTorch 基线计时（chunk 模式，两机跑）
├── tests/                        # 集成测试：parity_*.rs（读 tools 生成的 npz 对拍）
└── benches/                      # 逐算子 + 端到端计时（CUDA event）
```

### 4.2 算子清单与融合策略（核心：减少 kernel 数与全局存取）

| # | PyTorch 路径 kernel 数 | Rust 融合算子 | 说明 |
|---|---|---|---|
| 1 | stft：帧窗+FFT(内部) | `frame_hann` + cuFFT R2C plan | 分帧+乘窗一 kernel；cufftPlanMany 复用（形状固定） |
| 2 | bandsplit：62×(norm+linear)+cat | `bandsplit_fused` | 1 kernel：62 组按 dim_in 分组（4/8/16/48/96/192/512/516 五类宽度），组内 RMSNorm+F.normalize+GEMM，warp-per-band-row |
| 3 | 每层 time attn：norm+qkv+rope+sdpa+gates+out+resid ≈ 8 | `qkv_rope`(1) + `flash_attn`(1) + `gate_out_residual`(1) | QKV GEMM 的 epilogue 写出时即做 RoPE 旋转与 √64 缩放；flash：Q-tile 64 行×K/V-tile 64 列、online softmax、fp32 累加 |
| 4 | 每层 FF：norm+2×GEMM+GELU | `ff1_gelu`(1)+`ff2`(1)（或 Phase B 单 kernel 双 GEMM 流水） | GELU tanh 近似（PyTorch GELU 默认 erf——用 erf 版保证 parity，libdevice erff） |
| 5 | freq attn 同 3 | 同 3（序列 62，flash 退化为单 tile；直接 materialize 更快，走 attn.rs 的短序列路径） | |
| 6 | mask：6×62×(MLP+GLU)+cat+view | `maskest_packed` | 1 kernel：6 stems×62 bands 打包（每线程组处理一个 (stem,band)），Tanh/GLU 融合 epilogue |
| 7 | 复数乘 + view_as_complex | `cmul_mask` | 1 kernel，交错实虚直接乘 |
| 8 | istft ×6 | cuFFT C2R ×6 + `ola_scale` | 窗平方归一（torch istft 的 window_sum 语义：除以 Σw²）融合在回写 kernel |
| 9 | overlap-add 编排 | `chunk_window_acc` | chunk 输出 × 窗口累加 + counter，GPU 上完成 |

每 chunk 前向 kernel 数：1(frames)+1(bandsplit)+12×(2×3+2×2)(双轴 transformer)+1(maskest)+1(cmul)+6×2(istft)+2(ola) ≈ **130 kernels**（PyTorch eager ≈ 1500+，还有大量 cat/permute/copy）。

### 4.3 GEMM 引擎（分两阶段）
本模型 GEMM 形状集（fp32，M=71,342 tokens=1151×62 折叠后）：
- QKV: (71342×256)·(256×1536)；Out: (71342×512)·(512×256)；FF1: (71342×256)·(256×1024)；FF2: (71342×1024)·(1024×256)
- freq 轴：M=1151 折叠 62 → 同形状不同 M（M=71342 同一集合；实际 freq 轴 token=1151×62 相同——time/freq 折叠后 M 相同 71342）

**Phase A（正确性基线）**：SIMT tiled fp32 GEMM（shared memory 32×32 tile + K 循环 + 向量化 float4 装载），参考 examples/tiled_gemm 加强版（32×32→64×64 tile + 寄存器分块 4×4）。预期 ~3-6 TFLOPS（4060Ti fp32 峰值 34），已可与 PyTorch math-SDPA 路径一搏。
**Phase B（性能达标）**：mma.sync m16n8k16 f32·tf32 tensor core（cuda-device 的 `wmma`/`mma_frag`/`generated_ldmatrix` 原语已具备），ldmatrix 装载 + XOR swizzle smem + 双缓冲。参考 examples/{gemm, swizzle_smem, generated_ldmatrix}。TF32 精度：parity 门限放宽到 rel-err < 3e-3 单算子 / 端到端 SNR ≥ 40 dB（单独 `--tf32` 开关，默认 fp32 路径保 60 dB）。
> GEMM 单一形状集合极小（K∈{256,512,1024}，N∈{256,1024,1536}），不做 autotune，每种手调一份 policy（cuda-device `config` 模块的编译期 policy 模式）。

### 4.4 Attention
- time 轴（B=62, H=8, N=1151, D=64，fp32）：flash 风格——每 block 处理 64 行 Q tile，K/V 64 列 tile 迭代，smem 缓存 K/V tile，online rescale，fp32 累加。无 mask、无 dropout（推理）、scale=8⁻¹ᐟ²。
- freq 轴（B=1151, H=8, N=62, D=64）：单 tile materialize（62×62 scores 32KB/warp-group），直接 softmax·V。
- parity 注意：PyTorch 参考本身是 fp32 math SDPA；flash 重归约顺序差异 → 算子对拍门限 1e-4 起步，实测后按 ulp 水平定稿（预期 ~1e-5）。

### 4.5 STFT/ISTFT
- cuFFT via `sys/cufft` dlopen 绑定：`cufftPlanMany(2ch, 1151+pad 帧, R2C 2048)`、C2R ×6 stems；plan 创建一次常驻。
- STFT 前的 hann 窗乘 + 分帧 gathering 一 kernel（每线程一帧一元素，窗常量 smem）；ISTFT 后的 window_sum 归一（∑w²[h]）+ 定长裁剪一 kernel。
- 备选（若 cufft FFI 受阻）：手写 radix-2 Stockham 迭代 FFT（2048 固定，11 级，双 buffer），r2c/c2r 打包布局——工作量 1-2 天，性能可达 cuFFT 的 60-80%（数据量小，不是瓶颈）。

### 4.6 精度与显存
- 主路径全 fp32（与 PyTorch 参考同精度域）。中间张量常驻显存 ≈ x(73MB)+qkv(165MB)+attn out(73MB)+mask(56MB)+stft(9.4MB) ≈ 400MB 级，16G/20G 充裕。
- 可选 `--tf32`（GEMM tensor core）与 `--amp`（fp16 计存分离）后续迭代，验收以 fp32 为准。

### 4.7 构建/运行矩阵
| 场景 | 位置 | 命令要点 |
|---|---|---|
| 开发+本机测试 | WSL2 Ubuntu-22.04（4060Ti） | `cd /mnt/d/Projects/logic-bs-roformer-rs && CARGO_TARGET_DIR=$HOME/target-lbrr cargo oxide run --release lbrr -- --bench`；CUDA 12.6 toolkit（安装中）|
| 远程测试 | <REMOTE_NODE_NAME>（3080, CUDA 13.3） | 经 kexec.js：代码 tar→base64 传输或 git（若项目推远端）；`export PATH=/root/.cargo/bin:$PATH RUSTUP_TOOLCHAIN=nightly-2026-08-28 LIBCLANG_PATH=/usr/lib/llvm-19/lib CUDA_HOME=/usr/local/cuda-13.3`（此组合已验证 vecadd 成功）|
| 参考基准 | 本机 .venv2 / 远程 miniconda | tools/bench_ref.py |

构建兜底：若 WSL CUDA 12.6 的 libnvvm 与 LLVM23 NVVM IR 不兼容（vecadd 失败），则本机产物从远程构建：远程 `cargo oxide build --release` 出 PTX → 拷回 WSL/Windows，host 侧（cuda-core stable 构建）直接 cuModuleLoad PTX 运行（4060Ti driver 596 支持 sm_89 PTX JIT）。此路径在计划中作为 P1 回退。

## 5. 实施步骤（PDCA 循环）

### Phase 0 — 脚手架与构建管线（0.5 天）
1. ~~确认 WSL 安装~~（已完成：CUDA 12.6 + nightly-2026-08-28 就绪）；在 WSL clone cuda-rust 到 `$HOME/work/cuda-rust`，跑通 vecadd（复刻远程已验证命令；LIBCLANG_PATH 需按 WSL 实际 llvm 版本设置）
2. git init + submodule vendor/cuda-rust；Cargo.toml（standalone + path 依赖）；rust-toolchain.toml；`hello kernel` 在两机 GPU 上跑通
3. **P：两机 vecadd 输出正确；C：对比远程已知成功输出；A：如 WSL 失败走 PTX 分发回退路径**
4. CLI 骨架（clap）+ wav 读写（hound）+ YAML 解析

### Phase 1 — 数据层与 STFT（1 天）
1. tools/export_weights.py：ckpt → safetensors（键名原样保留）；weights.rs 加载映射表（§3）+ 一次性上传 GPU
2. sys/cufft dlopen 绑定 + 单测（R2C/C2R roundtrip vs numpy.fft）
3. frame_hann / ola_scale kernel；`stft.rs` 编排；parity：与 torch.stft 同输入 rel-err < 1e-5（STFT 是确定点积，门限从严）
4. audio.rs chunk/overlap 计划（窗口常量在 Rust 侧重建，与 §3 语义一致），CPU 参考实现先行（对拍 _getWindowingArray 数值）

### Phase 2 — 核心算子逐个对拍（2-3 天，每算子一个 PDCA）
顺序：rmsnorm → bandsplit（含 RMSNorm 融合）→ GEMM Phase A（4 形状）→ qkv_rope → 短序列 attn → flash attn → gate_out → ff(gelu) → transformer 单层组装 → maskest_packed → cmul
- tools/dump_refs.py 为每算子生成 (输入, PyTorch输出) npz（固定种子、真实形状）
- 每算子：实现 → 对拍 → 计时（CUDA event, 100 次均值）→ 记录到 benches/RESULTS.md
- 单层 transformer 组装后与 `layers.0.0` PyTorch 输出对拍

### Phase 3 — 端到端组装（1 天）
1. model.rs 全 12 层循环 + final_norm + maskest + cmul + istft：单 chunk 前向
2. E2E parity：ref_output.npz 对拍，6 stems SNR ≥ 60 dB
3. audio.rs 接通 overlap-add 全曲推理；CLI 完整流程
4. 两机 E2E bench（3s/30s/整曲），对比 torch 基线（远程 miniconda 装 torch 跑 tools/bench_ref.py）

### Phase 4 — 性能优化循环（2-3 天，量测驱动）
1. nsys/ncu（WSL 可用；远程经 exec 命令行版）定位热点 kernel
2. GEMM Phase B（mma.sync+ldmatrix+swizzle）：先 QKV（占时最大），逐形状替换并回归 parity/性能
3. 融合深化：gate_out 与 FF1 合并装载、maskest 与 final_norm 融合、ISTFT C2R 后处理合并
4. 每 优化的验收：算子级 bench 提升记录 + E2E RTF 不回退 + parity 不破

### Phase 5 — 交付（0.5 天）
1. README：模型结构图、算子表、两机构建/运行指南（含全部环境坑：LIBCLANG_PATH/CUDA_HOME/rsproxy/LNK1189 原委）
2. 基准报告（两机，vs PyTorch）：逐算子表 + E2E RTF + 加速比
3. 清理：研究用临时目录归档说明（D:\Projects\.{pymss,pymss-core,cuda-rust}-research 保留供查证）

## 6. 测试与对拍资产

- 金标准生成（tools/dump_refs.py，本机 .venv2）：
  `stft / bandsplit / qkv_rope(incl. rope) / attn_short / attn_flash / gate_out / ff / transformer_layer0_time / transformer_layer0_freq / maskest / cmul / e2e_chunk`
- 对拍框架：tests/parity.rs 读 npz（ndarray-npy crate），逐算子 rel-err = max|a-b|/(max|b|+ε)
- 计时：CUDA event 环路（warmup 10 + 100 iter），benches/ 输出 markdown 表
- 端到端音频验收：合成 3s（已有 ref_output.npz）+ 一段真实音乐（用户提供或 MUSDB18-AI 样本，HF 下载）听感 + SDR 粗检（可选 muskit 类工具，非必须）

## 7. 风险与对策

| 风险 | 概率 | 对策 |
|---|---|---|
| cuda-oxide alpha 期编译器 bug（特定 Rust 写法触发） | 中 | 算子小步快跑；改写法（数组化/去闭包/去递归）；`ptx_asm!` 内联 PTX 逃生舱；examples 里已有同类算子的验证写法可抄 |
| WSL CUDA 12.6 libnvvm 不接受 LLVM23 IR | 中 | 已验证的远程 13.3 构建 → PTX 产物拷回本机运行（§4.7 兜底） |
| cufft dlopen 绑定工作量大 | 低 | 函数集极小（5 个）；备选 Stockham kernel（§4.5） |
| flash attn 数值/性能不达标 | 低 | freq 轴已用短序列路径兜底；time 轴可退化为 2-pass materialize（2.6GB fp32 显存 16G 可容纳，慢但正确） |
| Komari exec 长任务中断/无输出 | 中 | 一律 nohup+日志文件模式；kexec.js 已扩轮询到 10 分钟；构建命令幂等（cargo 增量） |
| GEMM Phase B tensor core 调优超期 | 中 | Phase A SIMT 已可交付正确实现；Phase B 逐形状推进，每形状独立合入；TF32 parity 单独门限+开关 |
| 权重键遗漏/形状错位 | 低 | §3 映射表已从 state_dict_shapes.json 全量核对（1989 键分 6 组，枚举无遗漏）；加载时全键覆盖断言 |

## 8. 明确不做（YAGNI）
- 训练/反传/梯度（仅推理）
- 其他模型架构（mel_band/conformer/hyperace 等）——但算子保持参数化形状，留扩展余地
- fp8/int8 量化、CUDA Graphs、多 GPU（后续可加）
- pymss 的 TTA/ensemble/响度归一（CLI 不暴露，输出与 pymss 默认一致）
- Windows 原生构建（LNK1189 硬限制，永远走 WSL/远程）

## 9. 立即执行清单（Phase 0 首日）
1. ~~WSL 环境确认~~（已完成）；WSL 内 clone cuda-rust 并跑通 vecadd 于 4060Ti
2. WSL: git init 项目 + submodule cuda-rust + hello kernel 跑通 4060Ti
3. 远程: mkdir /data/dsh/logic-bs-roformer-rs（kexec），同步验证远程构建命令模板
4. tools/export_weights.py 产出 safetensors（~700MB，gitignore）
