# ADR: 内存 API 实现位置与元数据注册表归属

## Status

Accepted

## Context

P1 阶段需要实现内存管理接口（`coolMalloc` / `coolFree` / `coolMemcpy` / `coolMemset` / `coolMemGetInfo` / `coolMallocHost` / `coolFreeHost`）。两个问题需要明确：

1. **内存 API 实现在哪一层**：`docs/ARCHITECTURE.md` 的模块边界表将 `src/mem` 列为 P1 的「分配器、内存元数据、拷贝/填充」；但 P0 阶段的设备管理接口（`src/api/device_api.cpp`）是直接在 A 层实现、透传 BAL 的。
2. **内存元数据注册表归属**：为支持释放校验（拦截未知指针）与重复释放识别，运行时必须跟踪已分配的设备/主机指针。这类注册表天然需要**并发保护**（宿主互斥量），而 AGENTS.md 硬约束规定 **R 层禁止**使用 `std::thread`/`std::mutex`/`std::atomic`/`std::condition_variable` 等宿主并发原语，这些只允许出现在 `src/backend/**`。

因此注册表的物理位置与「R 层无并发原语」的红线存在直接冲突，必须做出取舍。

## Decision

- **内存 API 直接在 A 层实现**：新建 `src/api/mem_api.cpp`，沿用 `src/api/device_api.cpp` 的卫语句模式（参数校验 → `ensureInitialized` → 取 `activeBackend()` → 调用 BAL 函数指针）。P1 **不创建 `src/mem`**：由于注册表下沉到后端（见下条），R 层在该阶段无独立职责，空层属过度设计。
- **内存元数据注册表归属后端**：设备/主机指针的跟踪表（`AllocationRegistry`，内部使用 `std::mutex` + `std::unordered_map`）放在 `src/backend/cuda/cuda_backend.cpp`。`cudaMalloc`/`cudaMallocHost` 登记、`cudaFree`/`cudaFreeHost` 校验后注销、`cudaFinalize` 清空。
- **A 层与 R 层保持无并发原语**：两层只通过 BAL 函数指针交互，不直接持有任何宿主锁；并发保护全部由后端实现承担，符合 AGENTS.md 硬约束。

## Consequences

Positive:

- **不触碰 R 层红线**：并发原语只出现在允许的后端目录，架构约束无需放宽。
- **与既有先例一致**：内存接口与设备接口一样在 A 层薄透传，边界清晰、可读性高。
- **校验集中**：释放校验与重复释放识别集中在后端，注册表与真实的 `cuMemFree`/`cuMemFreeHost` 调用在同一模块，减少跨层状态同步。
- **避免空层**：不创建无职责的 `src/mem`，符合最小复杂度原则。

Negative / trade-offs:

- **与 `docs/ARCHITECTURE.md` 现有模块表不符**：表中 `src/mem` (P1) 一行描述为「分配器、内存元数据、拷贝/填充」，与本决策（内存 API 在 `src/api`、元数据在后端、无 `src/mem`）存在偏差，`ARCHITECTURE.md` 需相应更新或指向本 ADR。
- **后端职责变重**：cuda 后端除硬件映射外还承担了分配元数据管理，需在后端内部维持注册表与驱动状态的一致（如 `cudaFinalize` 清理）。
- **未来若需 R 层内存策略**（如内存池、池化分配，见 ARCHITECTURE.md Non-Goals 的后续阶段）：需重新评估本决策，届时可能引入 `src/mem` 并重新划定注册表归属。

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| 注册表放 R 层 `src/mem`（初版 ARCHITECTURE 设想） | 与模块表一致；R 层统一管理内存元数据 | 需要 `std::mutex`，直接违反 R 层无并发原语硬约束；或需放宽约束、破坏架构红线 |
| 注册表放 A 层（公开 API 实现处） | 无新增层 | A 层同样属 R 层约束范围，仍禁用并发原语；且 A 层职责应仅为校验与透传 |
| 注册表放后端 + A 层薄透传（本决策） | 满足硬约束；与 device_api 先例一致；校验集中 | 需更新 ARCHITECTURE 模块表；后端职责变重 |
