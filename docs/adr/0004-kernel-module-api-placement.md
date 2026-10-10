# ADR: 内核/模块 API 形态与句柄注册表归属

## Status

Accepted

## Context

P3 阶段需要实现内核（kernel）相关能力。与 P1/P2 不同，此处存在一个**前置的形态问题**：`docs/ARCHITECTURE.md` 的 Non-Goals 明确「**不做 CUDA C++ 编译前端**（PTX/cubin 由 nvcc 或外部工具产出）」，因此运行时无法接收 `__global__` 函数指针，必须有一种方式把**已编译好的**内核映像交给运行时。两个问题需要明确：

1. **内核如何获取、公开 API 采用何种形态**：可选路线包括 Driver 风格的显式模块 API（对应 `cuModuleLoad*` / `cuModuleGetFunction` / `cuLaunchKernel`），或运行时自管 PTX 缓存的简化入口，或仅提供最小 `coolLaunchKernel` 由调用方注入句柄。
2. **代码归属哪一层**：`docs/ARCHITECTURE.md` 的模块边界表将 `src/kernel` 列为 P3 的「内核注册表、启动跳板、占用率」；但 P0/P1/P2 的设备、内存、流/事件接口都直接在 A 层实现、透传 BAL（见 [ADR 0002](0002-memory-metadata-registry-placement.md) 与 [ADR 0003](0003-stream-event-handle-registry-placement.md)）。

补充约束：

- BAL 已预留内核能力位 `launchKernel(const coolKernel* k, dim3 grid, dim3 block, void** args, size_t sharedMem, coolStream_t s)`（见 `src/backend/cool_backend.h`）。
- 数据模型已定义：`coolKernel_t` 为不透明指针，cuda 后端内部即 `CUfunction`（见 `include/cool/cool_types.h`）。
- AGENTS.md 硬约束：**R 层禁止** `std::thread`/`std::mutex`/`std::atomic`/`std::condition_variable`，这些只允许出现在 `src/backend/**`。模块/函数句柄的生存期校验需要并发保护，故注册表的物理位置与「R 层无并发原语」红线直接冲突。

## Decision

### 1. 公开 API：Driver 风格显式模块 API

采用与 CUDA Driver API 一一对应的显式模块模型，内核映像由外部工具产出、运行时只负责载入与启动：

| COOLRuntime API | 对应 Driver API | 说明 |
| --- | --- | --- |
| `coolModuleLoadData(coolModule_t* module, const void* image)` | `cuModuleLoadData` | 从内存载入模块（PTX 文本或 cubin 映像），驱动负责 JIT/加载 |
| `coolModuleLoadFromFile(coolModule_t* module, const char* path)` | `cuModuleLoad` | 从文件载入模块 |
| `coolModuleGetFunction(coolKernel_t* kernel, coolModule_t module, const char* name)` | `cuModuleGetFunction` | 按符号名取内核句柄 |
| `coolModuleUnload(coolModule_t module)` | `cuModuleUnload` | 卸载模块 |
| `coolLaunchKernel(coolKernel_t kernel, dim3 grid, dim3 block, void** args, size_t sharedMem, coolStream_t stream)` | `cuLaunchKernel` | **启动跳板**：直接转发 `void** args`（与 `cuLaunchKernel` 的 `kernelParams` 语义一致，不做参数类型推导），`stream` 沿用「NULL 表示默认流」语义 |
| `coolOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, coolKernel_t kernel, int blockSize, size_t dynamicSMemSize)` | `cuOccupancyMaxActiveBlocksPerMultiprocessor` | **占用率**查询 |
| `coolOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, coolKernel_t kernel, size_t dynamicSMemSize, int blockSizeLimit)` | `cuOccupancyMaxPotentialBlockSize` | **占用率**查询（推荐 block 大小） |

- 新增不透明句柄 `coolModule_t`（`struct coolModule_st*`）到 `include/cool/cool_types.h`。
- **不做编译前端**：运行时不解析 CUDA C++、不生成 PTX/cubin，内核映像始终由 nvcc/外部工具提供，与 Non-Goals 一致。
- **不做隐式注册**：不提供「按名字启动」的简化入口，调用方必须显式持有 `coolKernel_t`，保持与 Driver API 的对照关系（便于教学）。

### 2. 代码归属：沿用 ADR 0002/0003 模式（A 层薄透传 + 后端注册表，不引入 `src/kernel`）

- **内核 API 直接在 A 层实现**：新建 `src/api/kernel_api.cpp`（含模块、函数、启动跳板、占用率），沿用 `src/api/device_api.cpp` / `mem_api.cpp` / `stream_api.cpp` 的卫语句模式（参数校验 → `ensureInitialized` → `activeBackend()` → 调用 BAL 函数指针）。`coolLaunchKernel` 即**启动跳板**：校验 `kernel`/`stream` 合法性后直接转发 `void** args` 到 BAL。
- **BAL 扩展**：在 `coolBackend` 中新增 `moduleLoadData` / `moduleLoadFromFile` / `moduleGetFunction` / `moduleUnload` / `occupancyMaxActiveBlocksPerMultiprocessor` / `occupancyMaxPotentialBlockSize`；`launchKernel` 复用已预留的能力位。返回 NULL 表示未实现，转发前检查并返回 `coolErrorNotSupported`。
- **模块/函数句柄注册表归属后端**：跟踪 `CUmodule` 与 `CUfunction` 的生存期（拦截未知句柄、重复卸载、卸载仍在使用中的模块）。该注册表复用后端已有的 `HandleRegistry`（`std::mutex` + `std::unordered_set<std::uintptr_t>`）机制，登记于 `cudaModuleLoad*` / `cudaModuleGetFunction`，注销于 `cudaModuleUnload`，`cudaFinalize` / `cudaDeviceReset` 统一清理。
- **不引入 `src/kernel`**：本阶段「内核注册表、启动跳板、占用率」三项职责中，注册表下沉后端（因并发约束），启动跳板与占用率均为无状态薄透传（在 A 层），故 R 层无独立职责，创建 `src/kernel` 属空层/过度设计。
- **A 层与 R 层保持无并发原语**：并发保护全部由后端承担，符合 AGENTS.md 硬约束。

### 3. 可测试性（无 CUDA Toolkit 亦可验证）

`coolModuleLoadData` 接受 PTX 文本，驱动在加载时做 JIT 编译，因此单元测试可在源码中**内嵌一段最小 PTX** 字符串（如一个空的 `.entry`）来加载模块、取函数、启动并校验，无需 nvcc/ptxas。测试对**不支持在运行时 JIT PTX、或缺少可用设备**的环境返回 `coolErrorNotSupported`/跳过，避免测试不稳定。

## Consequences

Positive:

- **契合 Non-Goals**：明确引入「内核映像由外部工具产出」的边界，不在运行时做编译。
- **可教学对照**：公开 API 与 Driver API 模块模型一一对应，读者可直接对照 `cuModule*` / `cuLaunchKernel`。
- **不触碰 R 层红线**：并发原语仍只出现在后端目录，架构约束无需放宽（与 ADR 0002/0003 同一原则）。
- **无 Toolkit 可测**：借驱动 PTX JIT，单元测试不需要 CUDA Toolkit。
- **避免空层**：不创建无职责的 `src/kernel`。

Negative / trade-offs:

- **与 `docs/ARCHITECTURE.md` 现有模块表不符**：表中 `src/kernel` (P3) 一行描述为「内核注册表、启动跳板、占用率」，与本决策（API 在 `src/api`、句柄状态在后端、无 `src/kernel`）存在偏差，`ARCHITECTURE.md` 需相应更新或指向本 ADR。
- **BAL 契约变更**：`coolBackend` 新增多个能力位，属对内接口扩展（`COOL_BACKEND_API_VERSION` 语义上应随之审视）。
- **参数编组由调用方负责**：`void** args` 只做透传，运行时不做类型/大小校验，误用会在驱动层报错，错误定位相对靠后。
- **后端职责进一步变重**：cuda 后端需额外维护模块/函数生存期与注册表一致性。

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| 运行时托管 PTX 缓存 + `coolLaunchKernelByName` | 调用方接口更简单；可内建缓存 | 运行时需自研模块缓存/命名注册表，复杂度高；与「不做编译前端」的边界模糊；难以对照 Driver API |
| 最小 `coolLaunchKernel`（注入已有 `CUfunction`） | 接口最少 | 无模块生命周期管理，调用方需自行持有并保证 `CUfunction` 有效；不利于教学对照与句柄校验 |
| 本决策：Driver 风格显式模块 API + A 层薄透传/后端注册表 | 契合 Non-Goals；与 Driver API 一一对应；与 ADR 0002/0003 原则一致；无 Toolkit 可测 | 需更新 ARCHITECTURE 模块表；BAL 扩展；后端职责变重 |
