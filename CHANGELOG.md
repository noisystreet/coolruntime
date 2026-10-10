# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses semantic versioning when releases are created.

## [Unreleased]

### Added

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

### Changed

### Fixed

### Removed

- 移除 `cpu` 软件后端及其自动回退逻辑；后端不可用/初始化失败时直接返回错误码。
- 移除 `null` 测试后端与 `COOL_BACKEND` 后端选择逻辑；CUDA Driver API（`cuda.h` + `libcuda`）成为唯一后端与构建硬依赖。
