# Measured Performance Implementation Plan

> **For agentic workers:** 单会话按任务执行；用户明确不提问，不启用 delegation 或额外批准流程。

**Goal:** 降低已测得的启动转换和音频复制开销，仅采用实测通过的热推理变更。

**Architecture:** 将host FP16 packing隔离成可单独测试的模块；运行时检测AVX/F16C并保留标量fallback。WAV边界使用borrow/direct little-endian I/O，GPU数学保持不变。

**Tech Stack:** Rust std、现有cuda-oxide/cuBLASLt/cuDNN、Python NumPy、Nsight Systems。

**Spec:** [实测设计](<D:/Projects/logic-bs-roformer-rs/docs/plans/2026-10-08-measured-performance-design.md>)。

**验收:** [实测报告](<D:/Projects/logic-bs-roformer-rs/docs/benchmarks/2026-10-08-performance.md>)；GPU默认和精度不变，Q/K fused候选未采用。

## Global Constraints
- 无新增依赖、不改vendor/toolchain、模型权重或精度。
- 不修改已有output，实测文件置于output/perf-2026-10-08。
- 原始二进制不是current HEAD重编译基线；最终A/B必须同环境、同参数、同fixture。
- 不用历史3080/WSL墙钟作为本机性能gate。

### Task 1: 建立当前源码B0并筛选现有选项
- [x] cargo oxide build -- --release，保存二进制并核验embedded source digest。
- [x] short/full各做split/fused交替三轮20次，逐stem及bit比较，不稳定则不更改默认。

### Task 2: CPU精确FP16 packing
**Files:** src/fp16.rs（新）、src/main.rs（host转换与upload_weights接线）。
**Interfaces:** f32_to_f16_bits(f32)->u16；pack_f16x2(&[f32])->Vec<u32>；pack_parts(&[&[f32]])->Vec<u32>。
- [x] 写标量兼容模块和测试，先运行`rustc --edition=2024 --test src/fp16.rs -o output/perf-2026-10-08/fp16-tests.exe`。
- [x] 添加F16C批量路径，保持nearest-even、NaN bit和低位顺序，尾长回退标量。
- [x] 测试所有有限half值、tie、tiny、随机bit和实际权重parts的逐位一致。
- [x] 测量标量与批量同一输入的pack时间；无收益则保留标量。

### Task 3: WAV复制消除
**Files:** src/audio.rs（读data借用、写header/body与测试）。
- [x] 独立`rustc --test src/audio.rs`运行roundtrip/byte compatibility测试。
- [x] little-endian直接写f32 bytes，big-endian保持逐值bounded conversion；长度溢出和无效声道在创建文件前报错。
- [x] 借用read_wav data payload，保持PCM16/24/F32解码。
- [x] 交替测量同一大音频的读写，输出bytes一致。

### Task 4: 验收及证据报告
- [x] 重编译候选，运行unit、GPU kernel regression及short/full waveform校验。
- [x] 至少5轮A/B测初始化、热Event、进程墙钟；逐stem保留golden指标并比较B0bytes。
- [x] 验证真实整曲或30秒输出、边界和finite，记录端到端而不是仅GPUwall。
- [x] 新报告写docs/benchmarks/2026-10-08-performance.md，引用原始JSON并明确环境、抖动、参考局限、拒绝候选。
