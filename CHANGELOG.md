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
- 单元测试（version / error / device）与 C ABI 编译校验、`samples/hello` 示例。

### Changed

### Fixed

### Removed

- 移除 `cpu` 软件后端及其自动回退逻辑；后端不可用/初始化失败时直接返回错误码。
- 移除 `null` 测试后端与 `COOL_BACKEND` 后端选择逻辑；CUDA Driver API（`cuda.h` + `libcuda`）成为唯一后端与构建硬依赖。
