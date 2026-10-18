# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses semantic versioning when releases are created.

## [Unreleased]

### Added

- 共享库产物 `libcoolrt.so`（与静态库同源；经 OBJECT 库复用，不重复编译）。
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
- 设备符号拷出接口 `coolMemcpyFromSymbol`（API 对齐 cudart 的 `cudaMemcpyFromSymbol`）：
  与既有的 `coolMemcpyToSymbol` 对称，覆盖「设备符号 → 主机 / 设备内存」方向。
  `kind` 只接受 `coolMemcpyDeviceToHost`（默认语义）与 `coolMemcpyDeviceToDevice`，
  其余方向与越界均返回 `coolErrorInvalidValue`；同时提供 `cudaMemcpyFromSymbol` 兼容别名。
  后端函数表新增 `memcpyFromSymbol`，`cuda` 后端以 `cuMemcpyDtoH` / `cuMemcpyDtoD` 实现。
  补齐 `test_symbol` 覆盖（往返一致、偏移、越界、非法方向、未初始化描述符、别名可用）。

### Changed

- **库产物重命名**，对齐 CUDA（`libcudart_static.a` / `libcudart.so`）：
  `libcoolruntime.a` → **`libcoolrt_static.a`**（静态带 `_static` 后缀），并新增 **`libcoolrt.so`**。
  CMake target 名仍为 `coolruntime`（共享库为 `coolruntime_shared`，别名
  `COOLRuntime::coolruntime` / `COOLRuntime::coolruntime_shared`），测试与 samples 无需改动。
  下游 hacc 的默认链接路径已同步为 `libcoolrt_static.a`。

### Fixed

### Removed

- 移除 `cpu` 软件后端及其自动回退逻辑；后端不可用/初始化失败时直接返回错误码。
- 移除 `null` 测试后端与 `COOL_BACKEND` 后端选择逻辑；CUDA Driver API（`cuda.h` + `libcuda`）成为唯一后端与构建硬依赖。
