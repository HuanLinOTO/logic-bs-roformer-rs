# logic-bs-roformer-rs

BS-RoFormer 六 stem 音源分离模型的纯 Rust + cuda-oxide 推理实现，
在 RTX 3080 上与 PyTorch/pymss 基线对比。

## 结果速览（RTX 3080 20GB，3 秒单 chunk，warm）

| 指标 | PyTorch 2.14.1 基线 | 本实现 | 对比 |
|---|---|---|---|
| wall（warm，10 次均值） | 120.8 ms | **73.5 ms** | **1.64×** |
| GPU kernel 合计（nsys） | ~119.6 ms | ~73 ms | 1.63× |
| RTF | 0.0403 | **0.0245** | — |
| 六 stem 输出 SNR | —（同一模型） | **80.87 dB** | 验收线 ≥60 dB |

完整剖析、逐内核时间线与全部优化实验记录见
[docs/BENCHMARK.md](docs/BENCHMARK.md)。

## 模型

- 174.66M 参数（1989 个 state-dict key 全量校验，逐 key 消费恰好一次）
- 62 频带 BandSplit → 12 层 ×（时间轴 + 频率轴）线性注意力 transformer
  → MaskEstimator → 复数域 mask → ISTFT 六 stem
- dim 256 / FF 1024 / 8 头（head dim 64）/ 每 8 头门控

## 代码结构

| 文件 | 内容 |
|---|---|
| src/main.rs | host 编排、全部 CUDA 内核（gpu_kernels 模块）、warm bench |
| src/weights.rs | Safetensors 加载与全量 key 校验 |
| src/config.rs | 零依赖 YAML 解析 |
| src/stft.rs, src/cufft.rs | STFT/ISTFT 与 dlopen cuFFT 绑定 |
| src/audio.rs, src/npz.rs | WAV 与 NPZ 读写 |
| tools/bench_ref.py | PyTorch 基线（同输入、同计时口径） |
| tools/dump_refs.py | parity 参考生成 |
| scripts/build_remote.sh | 远程构建入口（cargo oxide run） |

## 关键内核

- **attn_flash_tc**：tensor-core flash attention。RoPE 融合进 Q/K 加载、
  在线 softmax 在寄存器内、C fragment 原地转 PV 的 A fragment、门控
  sigmoid 融合进 epilogue；分数矩阵零落显存。两轴统一使用。
- **gemm_f16_128x64 / residual / gelu**：f16x2 tensor-core GEMM，float4
  向量化加载，残差/GELU 融合进 epilogue。
- **rmsnorm_gates**：RMSNorm 与 8 头门控投影融合（lane-major 合并访问）。
- **mask_gemm1/2 + glu_scatter**：逐 stem 分组 MaskEstimator。

## 本机（Windows + WSL）构建

Windows 原生 MSVC 无法编译：cuda-rust 的 rustc_codegen_cuda 后端 DLL 需要
导出 ~10 万个符号（上游 Windows 适配的转发 thunk），超过 PE 导出表 65535
硬上限（link.exe LNK1189 / lld-link 同报 too many exported symbols），
已双链接器实测证伪。本机编译走 WSL：

```bash
# 一次性环境：WSL Ubuntu-22.04 + rustup nightly-2026-08-28 +
#   CUDA 13.3 toolkit（.cn ubuntu2204 apt 源，cuda-toolkit-13-3）
#   + llvm-14（libclang，bindgen 用）

# 环境已持久化到 WSL /etc/profile.d/99-rust-cuda.sh（PATH 含
# /root/.cargo/bin 与 /usr/local/cuda-13.3/bin、CUDA_HOME、LIBCLANG_PATH、
# CARGO_TARGET_DIR），登录 shell（bash -l）自动生效，无需手动 export。
# rustup 工具链由 rust-toolchain.toml 钉定 nightly-2026-08-28。

# 同步源码到 ext4 并构建（rsync exclude 已修复为锚定 /target，
# 不再误删 cuda-oxide-codegen/src/target/ 源码目录）
wsl -d Ubuntu-22.04 -u root -- bash -lc \
  'bash /mnt/d/Projects/logic-bs-roformer-rs/scripts/sync_wsl.sh && \
   cd /root/work/lbrr && cargo oxide build -- --release'
# 产物: /root/work/target-lbrr/release/lbrr（GPU 直通可用）
```

注意：cuda-oxide-codegen 的 src/target/ 目录被上游 gitignore 规则吞掉
（不入库），只能靠文件系统同步——sync_wsl.sh 的 exclude 必须保持锚定写法。

本机 cudnn fused SDPA（可选，提速 ~20%）：部署在 /opt/lbrr-cudnn/（pip wheel
cudnn 9.10.2.21 + nvrtc 12.6.85 的 lib/include + 本机重编的
libcudnn_sdpa_wrap.so——远程编译版需要 GLIBC 2.38，Ubuntu 22.04 只有 2.35，
须用本机 g++ 11 重编：frontend 头在 cfe/cudnn-frontend-v1212/include）。
LBRR_CUDNN_DIR/LBRR_NVRTC_DIR/LBRR_SDPA_WRAP 已入 profile.d。坑：frontend
的 load_cudart_so() 要求进程可 dlopen 的 libcudart 唯一——WSL 装过 CUDA
12.6 时 ldconfig 缓存同时有 .12/.13 会报 "Multiple libcudart"，需禁用
ld.so.conf.d 里 12.x 的条目（988_cuda-12.conf、gds-12-6.conf）后 ldconfig。
验证：golden SNR 80.94dB，全曲 GPU wall 6.67s(回退) -> 5.45s(cudnn)。

## 远程构建与运行（基准环境：Komari 节点 RTX 3080）

```bash
# 上传源码（MD5 校验的可靠通道）
node <SSH_TOOLS_DIR>/upload_model.js src/main.rs \
     /data/dsh/logic-bs-roformer-rs/src/main.rs 1

# 构建并跑端到端正确性
cd /data/dsh/logic-bs-roformer-rs
nohup bash scripts/build_remote.sh -- --e2e-test > build.log 2>&1 &

# warm 基准（与 tools/bench_ref.py 同口径）
/data/dsh/target-lbrr/release/lbrr --bench --iters 10
```

环境要求（build_remote.sh 已内置）：RUSTUP_TOOLCHAIN=nightly-2026-08-28、
CUDA_HOME=/usr/local/cuda-13.3、LIBCLANG_PATH=/usr/lib/llvm-19/lib、
CARGO_TARGET_DIR=/data/dsh/target-lbrr。

## 已知边界（详见 BENCHMARK.md）

1. f16 GEMM 停在 ~15 TFLOP/s：LDS fragment 读是墙；非 2 次幂行距触发
   ~5× 编译惩罚（bank 冲突修复两轮证伪）、>32 个累加 float 的寄存器
   数组必 spill（四轮证伪）、cp.async 对已 L2 命中的加载无收益。
2. flash 注意力与三段链持平（~2.6 ms/launch 时间轴）：gather 模式由
   折叠 qkv 布局决定，warp 级 MLP 已足够掩盖延迟。
3. cuBLAS/cuBLASLt 与 cuda-oxide 的上下文/流不兼容（719/参数错误）。

## 正确性工作流

```bash
lbrr --e2e-test      # 黄金输入全链路：阶段级 parity + 六 stem SNR（80.87 dB）
lbrr --stft-test     # STFT vs torch.stft parity
lbrr --attn-test     # 注意力算子级 parity
# ...其余 --*-test 覆盖每个算子
```

参考数据在 assets/ref_output.npz 与 parity/（由 tools/dump_refs.py 生成）。
