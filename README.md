# logic-bs-roformer-rs

BS-RoFormer 六 stem 音源分离模型的纯 Rust + cuda-oxide 推理实现，
在 RTX 3080 上与 PyTorch/pymss 基线对比。

## 结果速览（RTX 3080 20GB，3 秒单 chunk，warm）

| 指标 | PyTorch 2.14.1 基线 | 本实现 | 对比 |
|---|---|---|---|
| wall（warm，10 次均值） | 120.8 ms | **91.9 ms** | **1.32×** |
| GPU kernel 合计（nsys） | ~119.6 ms | ~92 ms | 1.30× |
| RTF | 0.0403 | **0.0306** | — |
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

## 远程构建与运行（唯一测试环境：Komari 节点 RTX 3080）

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
