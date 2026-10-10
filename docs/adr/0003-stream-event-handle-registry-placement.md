# ADR: 流/事件 API 实现位置与句柄注册表归属

## Status

Accepted

## Context

P2 阶段需要实现流管理（`coolStreamCreate` / `coolStreamCreateWithFlags` / `coolStreamCreateWithPriority` / `coolStreamDestroy` / `coolStreamSynchronize` / `coolStreamQuery` / `coolStreamWaitEvent`）与事件管理（`coolEventCreate` / `coolEventCreateWithFlags` / `coolEventDestroy` / `coolEventRecord` / `coolEventSynchronize` / `coolEventQuery` / `coolEventElapsedTime`）。两个问题需要明确：

1. **流/事件 API 实现在哪一层**：`docs/ARCHITECTURE.md` 的模块边界表将 `src/exec` 列为 P2 的「流、事件、调度器」；但 P0 的设备接口与 P1 的内存接口（见 [ADR 0002](0002-memory-metadata-registry-placement.md)）都直接在 A 层实现、透传 BAL。
2. **句柄校验状态归属**：为拦截未知句柄与识别重复销毁，运行时必须跟踪后端创建的流/事件句柄。这类注册表天然需要**并发保护**（宿主互斥量），而 AGENTS.md 硬约束规定 **R 层禁止**使用 `std::thread`/`std::mutex`/`std::atomic`/`std::condition_variable` 等宿主并发原语，这些只允许出现在 `src/backend/**`。

本决策沿用 ADR 0002 已确立的原则：「需要并发保护的状态一律下沉后端；A/R 层只做校验与透传，不持有宿主锁；避免空层」。

## Decision

- **流/事件 API 直接在 A 层实现**：新增 `src/api/stream_api.cpp` 与 `src/api/event_api.cpp`，沿用 `src/api/device_api.cpp` / `src/api/mem_api.cpp` 的卫语句模式（参数校验 → `ensureInitialized` → 取 `activeBackend()` → 调用 BAL 函数指针）。`coolStreamCreate` 归一化到 `coolStreamCreateWithFlags`，再归一化到 `coolStreamCreateWithPriority`；`coolEventCreate` 归一化到 `coolEventCreateWithFlags`。
- **不引入 `src/exec`**：P2 的流/事件仅需薄透传，R 层无独立职责；调度器属后续阶段（P3 内核）。此刻创建 `src/exec` 属过度设计，故本阶段不创建该目录。
- **句柄注册表归属后端**：流/事件句柄的跟踪表（`HandleRegistry`，内部使用 `std::mutex` + 两个 `std::unordered_set<std::uintptr_t>`）放在 `src/backend/cuda/cuda_backend.cpp`。`cudaStreamCreate`/`cudaEventCreate` 登记、`cudaStreamDestroy`/`cudaEventDestroy` 校验后注销、`cudaFinalize` / `cudaDeviceReset` 清空。
- **默认流语义**：`NULL` 表示默认流，不登记入注册表；`streamSynchronize`/`streamQuery`/`streamWaitEvent` 对 `NULL` 直接透传驱动（`cuStream*` 接受 0），而 `streamDestroy(NULL)` 返回 `coolErrorInvalidValue`。
- **A 层与 R 层保持无并发原语**：两层只通过 BAL 函数指针交互，不直接持有任何宿主锁；并发保护全部由后端实现承担，符合 AGENTS.md 硬约束。

## Consequences

Positive:

- **不触碰 R 层红线**：并发原语只出现在允许的后端目录，架构约束无需放宽。
- **与既有先例一致**：流/事件接口与设备、内存接口一样在 A 层薄透传，边界清晰、可读性高（ADR 0002 同一原则）。
- **校验集中**：句柄校验与重复销毁识别集中在后端，注册表与真实的 `cuStreamDestroy`/`cuEventDestroy` 调用在同一模块，减少跨层状态同步。
- **避免空层**：不创建无职责的 `src/exec`，符合最小复杂度原则。

Negative / trade-offs:

- **与 `docs/ARCHITECTURE.md` 现有模块表不符**：表中 `src/exec` (P2) 一行描述为「流、事件、调度器」，与本决策（流/事件在 `src/api`、句柄状态在后端、本阶段无 `src/exec`）存在偏差，`ARCHITECTURE.md` 需相应更新或指向本 ADR。
- **后端职责变重**：cuda 后端除硬件映射外还承担了流/事件句柄的生命周期校验，需在后端内部维持注册表与驱动状态的一致（如 `cudaFinalize` 清理）。
- **未来若引入调度器**（如多流调度、优先级调度策略，见路线图 P3+）：需重新评估本决策，届时可能引入 `src/exec` 并重新划定句柄/调度状态归属。

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| 句柄注册表放 R 层 `src/exec`（初版 ARCHITECTURE 设想） | 与模块表一致；R 层统一管理执行状态 | 需要 `std::mutex`，直接违反 R 层无并发原语硬约束；或需放宽约束、破坏架构红线 |
| 句柄注册表放 A 层（公开 API 实现处） | 无新增层 | A 层同样属 R 层约束范围，仍禁用并发原语；且 A 层职责应仅为校验与透传 |
| 句柄注册表放后端 + A 层薄透传（本决策） | 满足硬约束；与 ADR 0002 / device_api 先例一致；校验集中 | 需更新 ARCHITECTURE 模块表；后端职责变重 |
