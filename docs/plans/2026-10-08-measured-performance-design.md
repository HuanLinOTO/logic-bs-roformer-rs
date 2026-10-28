# 实测性能优化设计

## 目标和约束
以当前 Windows RTX 4060 Ti 实测为依据，不以旧 benchmark 的百分比或 gate 判定为验收依据。用户要求全程不提问，采用最小改动、单会话执行，在当前工作区落地；保留用户已有 output/ 和所有模型、音频。无新增依赖、不改 vendor、精度、GELU、矩阵归约或模型语义。

## 实际发现
保存的原始二进制 SHA256 b29789399adec5d8f840c43a84e5a01580c2ce601df3b3bb802150995c668aa4，嵌入源码 revision 3b3befd，与当前 HEAD b3c78c8 不同，必须重新构建当前源码基线。短块 58.30ms，完整块 290.16ms；Nsight 最后完整块 kernel 292.69ms/span293.12ms，空隙0.146%。SDPA75.62ms，FF1 36.92ms，RoPE两核29.96ms，gate13.53ms。权重 CPU FP16 pack347.8/400.5ms。这些为初筛，不是最终收益。完整块已有参考含旧同平台输出，不能冒充 FP32 golden。

## 方案取舍
1. **采用：CPU 精确转换与复制消除。** x86/x86_64 运行时检测 AVX+F16C，批量转换有限值，非有限值维持旧 NaN bit 约定，其他 CPU 保持标量路径。WAV 输出直接写合法的小端 float 字节，输入不再复制整个 data chunk。逐位相同、低风险。
2. **筛选：现有 Q/K fused。** 先按当前平台配对筛选，仅数值逐位相同且完整波形可重复获益才改变默认，不把旧3080结论泛化。
3. **不采用：CUDA Graph/多流/低精度/算法重写。** 提交空隙极小，这些方案收益没有依据，且扩大生命周期和数值风险。

## 验证
独立 rustc host 测试覆盖所有有限 half 值、FP32 tie、subnormal、随机bit、NaN、尾长、split parts；使用现有真实权重验证 pack 字节。WAV bytes与旧序列化逐位比较，mono/stereo、PCM16/24、IEEE F32、异常长度校验。当前源码重新构建并保存 B0，候选重复交替配对，记录CUDA Event、初始化分项、进程墙钟、CPU pack、内存、真实短golden与同平台B0逐stem误差。Nsight只解释热点，最终收益使用无profiler数据。全曲输出逐个WAV比较；本机有桌面负载，CV超2%则如实报告不稳定。
