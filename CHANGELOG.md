# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses semantic versioning when releases are created.

## [Unreleased]

### Added

- **`docs/TODO.md`**：把设备内建（原子补完 / 数学函数 / warp 原语 / 位操作 / `__ldg` 等）、
  主机异步拷贝与查询类 API 的待办，连同 Non-Goals 集中到一处。
  此前这些只知道「缺什么」，散落在 `ARCHITECTURE.md` 的 3 行 Open Decisions 里，没有可跟踪的清单。
  内容基于对两个仓的实测盘点：例如数学函数区分「零依赖」与「需 libdevice」两档，
  避免把近似指令误当精确语义。
- `README` 「已实现」表补上**设备符号四件套**（`coolMemcpyToSymbol` / `coolMemcpyFromSymbol` /
  `coolGetSymbolAddress` / `coolGetSymbolSize`）—— 代码早已实现并有 `test_symbol` 覆盖，表格漏列。

- **R 层转发逻辑可在无 GPU 下测试**：新增 `cool::installBackendForTesting()`（内部头
  `core/lifecycle.hpp`，测试专用），可注入假后端替换真实 cuda 后端。这一点能成立是因为
  BAL 用函数指针结构体表达能力 —— 转发逻辑本就与后端实现无关。
  新增 `test_api_dispatch`，**不需要 GPU**（跑在 0.00 s）即覆盖：参数校验先于初始化、
  懒初始化只发生一次、能力位缺失 → `coolErrorNotSupported`、后端错误码透传到返回值与
  TLS、实参按值转发（`size` / `coolMemcpyKind` 等）。此前所有测试都依赖真实驱动，
  R 层的这层逻辑等于没有回归保护。
- **`dim3` 补齐 CUDA 风格的 C++ 构造函数**（跨项目对齐，hacc TODO 的 B5）：新增
  `dim3(unsigned x = 1, unsigned y = 1, unsigned z = 1)`，支持 `dim3 block(256)` /
  `dim3 grid((n + 255) / 256)` 这类 CUDA 惯用写法（此前只能 `dim3 g = {a,b,c}`）。
  默认构造为 `(1,1,1)`，与 CUDA 一致（此前是未初始化）。
  **ABI 不变**：仍是 12 字节、对齐 4、可平凡复制（`coolLaunchKernel` 按值传 grid/block 的前提），
  C 下仍是纯聚合 struct（`#ifdef __cplusplus` 包住构造函数）。
  副作用：`dim3` 在 C++ 下不再是聚合类型，`dim3{x,y,z}` 改走构造函数（语义等价；hacc 的
  `to_dim3` 依赖这一点，已在其注释里写明）。
  新增 `test_dim3`：构造语义、默认值、`sizeof`/`alignof`/`is_trivially_copyable`、非聚合性。

- 共享库产物 `libcoolrt.so`（与静态库同源；经 OBJECT 库复用，不重复编译）。
- **Debian 打包**：`make package` 生成 `libcoolrt0_<ver>_<arch>.deb`（运行时）与
  `libcoolrt-dev_<ver>_<arch>.deb`（开发包）。按 Debian 惯例拆包：运行时包只带 soname
  （`libcoolrt.so.0`）；开发包带头文件、`libcoolrt_static.a`、裸名链接与 `pkg-config`
  文件 `coolrt.pc`。共享库新增 `VERSION` / `SOVERSION`（对齐 `libcudart.so.12`，
  soname 为 `libcoolrt.so.0`）。安装布局对齐 CUDA：前缀 **`/usr/local/cool`**，
  头在 `include/`、库在 `lib64/`（`CPACK_PACKAGING_INSTALL_PREFIX` 必须显式设置，
  CPack 默认会截成 `/usr`）。`coolrt.pc` 的 `Libs.private` 含 `-lcuda -lstdc++`，
  保证 C 程序用 `pkg-config --static` 链静态库时能成功（库内部为 C++ 实现）。
- 初始项目骨架（P0）：CMake 构建、CI、pre-commit。
- 公开头文件 `cool_version.h` / `cool_types.h` / `cool_runtime_api.h`（C ABI）。
- 统一入口头 `cool_runtime.h`（聚合 `cool_types.h` / `cool_version.h` / `cool_runtime_api.h`）。
- 后端抽象层 `cool_backend.h` 与后端访问器。
- `cuda`（Driver API）后端骨架。
- 版本管理、错误处理、设备管理接口。
- 内存管理接口（P1）：`coolMalloc` / `coolFree` / `coolMemcpy` / `coolMemset` / `coolMemGetInfo` / `coolMallocHost` / `coolFreeHost`。
- `cuda` 后端内存实现，以及后端侧内存元数据注册表（跟踪设备/主机指针，用于释放校验与重复释放识别）。
- 内存单元测试 `test_mem`。
- 流管理接口（P2）：`coolStreamCreate` / `coolStreamCreateWithFlags` / `coolStreamCreateWithPriority` / `coolStreamDestroy` / `coolStreamSynchronize` / `coolStreamQuery` / `coolStreamWaitEvent`。
- 事件管理接口（P2）：`coolEventCreate` / `coolEventCreateWithFlags` / `coolEventDestroy` / `coolEventRecord` / `coolEventSynchronize` / `coolEventQuery` / `coolEventElapsedTime`。
- `cuda` 后端流/事件实现，以及后端侧句柄注册表（跟踪流/事件句柄，用于销毁校验与重复销毁识别）。
- 流/事件单元测试 `test_stream` / `test_event`。
- 模块与内核接口（P3）：`coolModuleLoadData` / `coolModuleLoadFromFile` / `coolModuleGetFunction` / `coolModuleUnload` / `coolLaunchKernel` / `coolOccupancyMaxActiveBlocksPerMultiprocessor` / `coolOccupancyMaxPotentialBlockSize`。
- `cool_types.h` 新增模块句柄 `coolModule_t`。
- `cuda` 后端模块/函数/启动/占用率实现，并扩展后端侧句柄注册表（跟踪模块与函数地址，卸载模块时级联失效其函数）。
- 内核单元测试 `test_kernel`（内嵌最小 PTX，经驱动 JIT 加载，无需 nvcc/ptxas）。
- 单元测试（version / error / device）与 C ABI 编译校验、`samples/hello` 示例。
- 设备端内建头 `device_builtins.h`：执行/存储限定符宏（`__global__` / `__device__` /
  `__host__` / `__shared__` / `__constant__`，映射为 `__attribute__((cool_*))`）、内建向量类型
  （`uint3` / `int3`）、内建变量（`threadIdx` / `blockIdx` / `blockDim` / `gridDim` / `warpSize`）
  与内建函数（`__syncthreads`）。内建变量/函数仅在设备编译（`__COOLCC__`）下声明；
  由 `cool_runtime.h` 聚合，并纳入 C ABI 校验。
- `device_builtins.h` 新增原子内建声明（仅 `__COOLCC__` 下）：`atomicAdd` / `atomicSub` /
  `atomicExch` / `atomicMin` / `atomicMax` / `atomicCAS`，覆盖 `int` / `unsigned int` / `float`
  常用重载（`atomicAdd` 与 `atomicExch` 含 unsigned 重载；`atomicMin`/`atomicMax` 靠有符号与
  无符号重载区分语义）。由 `coolicc` 按名字映射到 LLVM `atomicrmw` / `cmpxchg`；返回运算前旧值。
- 设备符号拷入接口 `coolMemcpyToSymbol`（API 对齐 cudart 的 `cudaMemcpyToSymbol`）：
  `(symbol, src, count, offset, kind)`，把主机 / 设备内存写入设备符号；`kind` 支持
  `coolMemcpyHostToDevice`（默认语义）与 `coolMemcpyDeviceToDevice`，其余方向返回
  `coolErrorInvalidValue`，`offset + count` 越界亦拒绝。`symbol` 传 `coolSymbolDesc`
  描述符地址（替代 cudart 的「主机侧符号地址」编译期魔法，由 hacc 主机桩构造）。
  后端函数表新增 `memcpyToSymbol`，`cuda` 后端经 `cuModuleGetGlobal` 解析符号地址后
  走 `cuMemcpyHtoD` / `cuMemcpyDtoD`。同时提供 `cudaMemcpyToSymbol` 兼容别名（宏）。
- 设备符号拷出接口 `coolMemcpyFromSymbol`（API 对齐 cudart 的 `cudaMemcpyFromSymbol`）：
  与既有的 `coolMemcpyToSymbol` 对称，覆盖「设备符号 → 主机 / 设备内存」方向。
  `kind` 只接受 `coolMemcpyDeviceToHost`（默认语义）与 `coolMemcpyDeviceToDevice`，
  其余方向与越界均返回 `coolErrorInvalidValue`；同时提供 `cudaMemcpyFromSymbol` 兼容别名。
  后端函数表新增 `memcpyFromSymbol`，`cuda` 后端以 `cuMemcpyDtoH` / `cuMemcpyDtoD` 实现。
  补齐 `test_symbol` 覆盖（往返一致、偏移、越界、非法方向、未初始化描述符、别名可用）。

### Changed

- **默认流语义明确为 CUDA legacy 默认流**（[ADR 0005](docs/adr/0005-default-stream-semantics.md)）：
  `coolStream_t NULL` 与所有 blocking 流隐式同步，不支持 per-thread 默认流，也不为其预留开关。
  `ARCHITECTURE.md` 的 Open Decision「`COOL_STRICT_DEFAULT_STREAM` 严格模式是否默认开启」一并关闭 ——
  该项自 P2 起只有一行表格、代码中从未存在该开关，本次决定不引入（理由见 ADR，含两种可能解读的否定论证）。
  **无行为变更**：cuda 后端一直把 `NULL` 直接交给 `cuStream*` 系调用（`CUstream 0` 即 legacy 默认流），
  本条目只是把既有事实固化为契约，并让 hacc 的 `<<<>>>`（`stream = NULL`）语义有据可依。
  `cool_types.h` 的句柄注释同步补明语义。

- **`cuda_backend.cpp` 按职责拆分**（994 行 → 4 个模块）：原文件同时装着结果码映射、两个
  资源注册表、上下文生命周期和 40 个 BAL 适配函数。现拆为：
  `cuda_result_map`（`CUresult` → `coolError_t`）、`cuda_registry`（内存与句柄登记）、
  `cuda_context`（上下文与生命周期）、`cuda_backend`（BAL 适配层 + 后端描述符）。
  纯结构调整、零行为变化。命名同时澄清：`cuResultToCool` → `cuda::resultToError`、
  `registry()` → `cuda::allocations()`、`handleRegistry()` → `cuda::handles()`。
- **`lastError` 现在在成功调用后也刷新（对齐 cudart）**：`error.hpp` 一直声明「每次 API
  调用成功/失败都会刷新 TLS」，但实现只在失败路径写入，成功路径直接返回 `coolSuccess`。
  于是「失败 → 紧接着成功 → `coolPeekAtLastError()`」仍返回**上一次**的错误码，容易把
  调用方带偏。现由 `COOL_API_ENTER` 进入时先把 TLS 置为 `coolSuccess`（成功时就是这个
  值），失败再覆盖具体错误码。
  **可观测行为变更**：成功调用后 `coolPeekAtLastError()` 返回 `coolSuccess`，此前返回上一次
  的错误。`coolGetLastError` 的「读取即清空」语义不变；`coolGetErrorName/String` 这类纯
  查询接口仍不改变 TLS。由 `test_api_dispatch` 第 7 组断言锁住。
- **生命周期状态机下移到后端层**：新增 `src/backend/bootstrap.{hpp,cpp}`，负责选定后端、
  线程安全的一次性初始化与进程退出时的 finalize。`core/lifecycle.cpp` 从 84 行降到 30 行，
  只剩对 bootstrap 的转发，R 层保持无状态（符合 AGENTS.md「宿主并发原语只允许出现在
  后端层」）。
- **A 层转发样板收敛为一个宏**：44 个 `extern "C"` 中 35 个都在重复同一段「懒初始化 →
  取后端 → 检查空指针 → 检查能力位」的 6 行代码（约 210 行）。新增
  `COOL_API_ENTER(member)`（`core/lifecycle.hpp`），每个转发函数降到 2 行。
  `src/api/` 从 666 行降到 410 行。
  这不仅是行数问题：35 份手写检查里漏写一处就是静默的空指针解引用，收敛后「检查能力位」
  只有一个实现，新增 API 也不会忘。
- **库产物重命名**，对齐 CUDA（`libcudart_static.a` / `libcudart.so`）：
  `libcoolruntime.a` → **`libcoolrt_static.a`**（静态带 `_static` 后缀），并新增 **`libcoolrt.so`**。
  CMake target 名仍为 `coolruntime`（共享库为 `coolruntime_shared`，别名
  `COOLRuntime::coolruntime` / `COOLRuntime::coolruntime_shared`），测试与 samples 无需改动。
  下游 hacc 的默认链接路径已同步为 `libcoolrt_static.a`。

### Fixed

- **驱动错误码映射从 7 项扩到 15 项**：`cuda::resultToError` 此前只认识 7 个 `CUresult`，
  其余一律收敛成 `coolErrorUnknown`，用户只看到 `"unknown error"`。CUDA 13.4 共定义 105 个
  驱动错误码，这意味着除分配失败外的多数失败都拿不到有意义的分类。
  现补齐语义明确等价的：`NOT_INITIALIZED` / `DEINITIALIZED` → `coolErrorNotReady`、
  `NO_DEVICE` → `coolErrorInvalidDevice`、`INVALID_CONTEXT` / `INVALID_IMAGE` /
  `INVALID_SOURCE` / `INVALID_PTX` / `FILE_NOT_FOUND` → `coolErrorInvalidValue`。
  **有意不映射**内核运行期错误（`LAUNCH_FAILED` / `ILLEGAL_ADDRESS` / `LAUNCH_TIMEOUT`
  等）：把它们报成 `coolErrorInvalidValue` 会把「kernel 跑炸了」误导成「参数传错了」，
  比报 unknown 更糟。这条决策由 `test_result_map` 的反向断言锁住。
  实现同时从 switch 改为查表（16 个 case 的 switch 圈复杂度 16，撞上项目 ccn 门限 10；
  查表后 CCN = 2，加映射只需加一行数据）。
- **首次并发调用会重复初始化后端（重复 `cuCtxCreate` 泄漏 context）**：懒初始化原先在
  `core/lifecycle.cpp` 用函数局部静态量 + 普通 `bool` 实现，读写都没有同步。多个线程首次
  并发调用任意 `cool*` API 时都会读到「未初始化」，各自执行 `backend->init()`；在 cuda
  后端下就是两次 `cuCtxCreate` —— 第一个 context 被覆盖且永不销毁，`std::atexit` 也可能
  被重复注册。CUDA 运行时 API 本身是线程安全的，用户从线程池里调 `coolMalloc` 是正常
  用法，所以这不是理论风险。
  现在由 `bootstrap` 用 `std::mutex` 串行化「初始化 / 复位 / atexit 清理」，
  `std::atomic<bool>` 做无锁快路径（`release`/`acquire` 配对保证状态字段可见），
  避免每次 API 调用都进临界区。
  新增 `test_lifecycle_concurrency`：8 线程同时冲进 `coolMalloc`，假后端 `init` 内
  sleep 50 ms 放大竞态窗口，断言 init 恰好一次。**该测试已实测有效**——临时去掉锁后
  它会失败（`g_initCalls == 1` 不成立）。
- **初始化状态从两处收敛为一处**：此前 `core/lifecycle` 的 `LifecycleState` 与 cuda 后端的
  `g_initialized` 各记一份「是否已初始化」，且互不通知 —— `cudaFinalize()` 只清后端那份，
  于是 atexit 之后运行时仍声称「初始化成功」却没有后端（调用返回 `coolErrorNotReady`）。
  现在以 `bootstrap` 的状态为唯一权威，cuda 后端改用 `g_ctx != nullptr` 表示就绪，
  `g_initialized` 已删除。
- **4 处错误码不符合 BAL 契约**：`coolFree` / `coolMemcpy` / `coolMemset` / `coolFreeHost`
  在「后端能力位为 `NULL`」时返回 `coolErrorNotReady`，而 `cool_backend.h` 明确约定
  「未实现该能力则返回 `coolErrorNotSupported`」（其余 API 也都这么做的）。
  现把「后端未就绪」与「能力未实现」拆成两个判断，分别返回正确错误码。
  由新增的 `test_api_dispatch` 锁住。

### Removed

- 移除 `cpu` 软件后端及其自动回退逻辑；后端不可用/初始化失败时直接返回错误码。
- 移除 `null` 测试后端与 `COOL_BACKEND` 后端选择逻辑；CUDA Driver API（`cuda.h` + `libcuda`）成为唯一后端与构建硬依赖。
