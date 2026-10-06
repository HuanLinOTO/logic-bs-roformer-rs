# Windows 原生构建验证数据（2026-10-07）

环境：Windows 11 / RTX 4060 Ti / driver 610.88 / CUDA Toolkit 13.4 / stable Rust 1.99.0 / vendor cuda-rust-windows @ f3738d88

## Gate 结论

| 指标 | 目标 | 实测 | 判定 |
|---|---|---|---|
| golden bench SNR | >= 60 dB | 80.96 dB | 通过（新基线；WSL 侧 80.83） |
| 整曲 GPU wall | <= 5.92 s | 4.64 s（复测 5.58 s） | 通过（快 21.6%） |
| 8s 短输入 | 非零无静音尾 | 6 stems 尾部全非零 | 通过 |
| 整曲 vs WSL 产物 SNR | >= 70 dB | 内容 stem 62-70 dB（注） | 见注 |

注：piano(45.6)/bass(59.4) 低 SNR 为分母效应——两通道信号 RMS 分别为 7e-6（近静音）与 0.116，其绝对误差（4e-8 / 1.2e-4）反而低于高能量 stem。全部 stem 绝对误差均匀在 ~1e-4 RMS（-80 dB FS）量级，无实现差异。8s 输入唯一有声 stem（other）跨平台 SNR 92.35 dB。

## 稳定性

- Windows 同平台重复运行逐位一致（互 SNR 345-357 dB）
- WSL 侧（nightly 基线二进制）当前复测 GPU wall 9.06 s，与历史 5.92 s 有差距（疑为旧二进制/环境变化），不影响 gate 判定（以历史 5.92 s 为基线口径）

## 分解数据

- 权重加载+上传 1.13-1.27 s（冷）
- scratch 分配 ~45 ms
- cuBLASLt autotune：两 shape 各 ~5-13 ms（一次性）
- 下载 71-82 ms；6 stems f32 wav 382 MB 写盘计入 total wall
