# ADR: 默认流语义（`NULL` = legacy 默认流，不引入严格模式开关）

## Status

Accepted

## Context

`docs/ARCHITECTURE.md` 的 Open Decisions 长期挂着一行：

| Decision | Options | Owner | Due |
| --- | --- | --- | --- |
| `COOL_STRICT_DEFAULT_STREAM` 严格模式是否默认开启 | 默认关 / 默认开 | maintainers | P2 |

该项自 P2 建立以来从未被细化。全仓检索确认：`COOL_STRICT_DEFAULT_STREAM` **只存在于这张表格**，
代码中没有对应开关，也没有任何文档定义「严格模式」究竟指什么。同时 hacc 侧的
`docs/TODO.md` B4 一直等这个决策 —— 因为 hacc 生成的 `<<<>>>` 启动在未显式指定流时传 `NULL`，
其行为是否符合预期取决于此。

需要澄清的是两层不同的事：

1. **基础事实**（已由 [ADR 0003](0003-stream-event-handle-registry-placement.md) 决定）：
   `coolStream_t NULL` 表示默认流、不登记入句柄注册表，
   `streamSynchronize` / `streamQuery` / `streamWaitEvent` 对 `NULL` 直接透传驱动。
2. **本 ADR 要回答的**：`NULL` 映射到 CUDA 的**哪种**默认流语义，以及是否需要引入开关在两者间切换。

CUDA 提供两种默认流语义：

- **legacy（传统）**：`NULL` 流与所有 *blocking* 流（创建时未带 `CU_STREAM_NON_BLOCKING` 的流）
  之间存在隐式同步 —— 向默认流提交的工作会等待此前所有 blocking 流的工作，反之亦然。
- **per-thread（每线程）**：`NULL` 流是每线程私有的独立流，不与其他流隐式同步。
  由 nvcc 的 `--default-stream per-thread` 在**编译期**选择。

## Decision

**`coolStream_t NULL` 映射到 CUDA 的 legacy 默认流，且不引入 `COOL_STRICT_DEFAULT_STREAM`
或任何等价的运行期 / 编译期开关。**

| API | `stream == NULL` 的行为 |
| --- | --- |
| `coolLaunchKernel` | 提交到 legacy 默认流 |
| `coolStreamSynchronize` | 同步 legacy 默认流 |
| `coolStreamQuery` | 查询 legacy 默认流 |
| `coolStreamWaitEvent` | 令默认流等待指定事件 |
| `coolEventRecord` | 在默认流上记录事件 |
| `coolStreamDestroy` | 返回 `coolErrorInvalidValue`（默认流不可销毁） |

实现依据：cuda 后端把 `NULL` 直接 `reinterpret_cast<CUstream>` 传给 `cuStream*` 系调用，
`CUstream 0` 在 Driver API 中即 legacy 默认流；所有路径统一使用
`if (stream != nullptr && !handles().hasStream(...))` 的卫语句（`cuda_backend.cpp` 中
`cudaLaunchKernel` / `cudaStreamSynchronize` / `cudaStreamQuery` / `cudaStreamWaitEvent` /
`cudaEventRecord` 五处），语义自洽。

**不支持 per-thread 默认流**，也不为其预留开关。

## Consequences

Positive:

- **与 cudart 默认行为一致**：hacc 生成的 `<<<>>>`（`stream = NULL`）预期明确，等同 `nvcc` 默认编译结果。
  跨项目的 B4 待确认项可关闭。
- **行为可预测**：不存在「同一份二进制因环境变量而在两种流语义间漂移」的风险，复现与调试成本低。
- **不增加 API 表面积**：公开头文件无新开关，C ABI 不变。
- **本 ADR 固化的就是既有行为**：无代码变更、无回归风险。

Negative / trade-offs:

- **legacy 隐式同步会削弱并发**：默认流与 blocking 流之间的隐式同步可能造成意外串行化。
  需要完整并发的用户必须显式创建并全程使用自己的流 —— 这是 CUDA 本身的取舍，coolrt 不做规避。
- **若将来确需 per-thread**：须另开 ADR，并评估其对既有 `<<<>>>` 语义的破坏性（届时属破坏性变更）。

## Alternatives Considered

1. **用 `COOL_STRICT_DEFAULT_STREAM=1` 运行期切到 per-thread**（对该条目的一种可能解读）
   拒绝。该开关一旦开启，同一份二进制的语义随环境变化，违背可预测性；且 Driver API 的
   `cuLaunchKernel` 在 `CUstream 0` 上无法按线程区分默认流 —— CUDA 把该语义做成**编译期**选项
   （`-default-stream`），运行期模拟需在每次调用前替换 stream 句柄，代价与收益不匹配。

2. **严格模式 = 禁止使用默认流的检查**（另一种可能解读）
   拒绝。cudart 无对应 API；且会破坏不带流参数的 `<<<>>>` 既有写法（包括 hacc 生成的代码），
   与「对齐 cudart 行为」的目标直接冲突。

3. **默认流与所有流（含 non-blocking）都隐式同步**
   拒绝。比 CUDA legacy 语义更强，引入 cudart 没有的串行化，属语义偏离。

## References

- [ADR 0003](0003-stream-event-handle-registry-placement.md) —— 句柄注册表归属，含「`NULL` 不登记」的初步约定
- `docs/ARCHITECTURE.md` §Data Model
- hacc `docs/TODO.md` B4 —— 跨项目待确认项，本 ADR 关闭它
