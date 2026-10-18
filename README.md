# COOLRuntime

COOLRuntime 是一个**教学 / 研究用**的参考实现，提供一套**类 CUDA（CUDA-like）**的运行时 API，公开接口前缀为 `cool`（如 `cudaMalloc` → `coolMalloc`）。它在设计上借鉴 NVIDIA CUDA Runtime API（cudart），但**并非 cudart 的复刻或兼容实现**。

目标不是复刻 CUDA 的性能，而是：

1. 用可读的源码实现一套**类 CUDA** 的运行时，命名与用法上接近 cudart，便于对照学习；
2. 通过**后端抽象层（BAL）**把「运行时语义」与「具体硬件」解耦；当前提供 **CUDA Driver API 后端**（唯一后端）；
3. 作为教学与研究平台，讲解/验证内存模型、流式异步执行、事件同步、内核调度等概念。

架构概览见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 架构分层

```text
应用 / 示例 (samples, tests)
        |
        v
A 层：公开 API  (include/cool/cool_runtime.h → cool_runtime_api.h)   —— C ABI
        |
        v
R 层：运行时核心 (src/core, mem, exec, kernel)      —— C++20
        |
        v
B 层：后端抽象层 BAL (src/backend/cool_backend.h)
        |
        v
后端实现：cuda (CUDA Driver API)
```

层间依赖单向：上层依赖下层，下层不反向依赖。**R 层禁止**出现任何平台/宿主专有调用（`std::thread`/`std::mutex`/`std::atomic`/`malloc`/`pthread`），这些只允许出现在后端实现中。

## 语言边界

- **公开层（C ABI）**：`include/cool/*.h` 保持 C 兼容（`extern "C"` 声明，只用 C 类型），可被 C、C++ 及 CUDA 示例直接 `#include`。
- **实现层（C++20）**：R/BAL/各后端使用现代 C++（`-std=c++20`）。

## 当前状态（阶段 P0 骨架 + P1 内存 + P2 流/事件 + P3 模块/内核）

已交付：仓库结构、CMake 构建、公开头文件、BAL 接口、cuda 后端（含 `cuInit` + 上下文、内存、流、事件、模块与内核），版本/错误/设备/内存/流/事件/内核的单元测试、`samples/hello`。

| 已实现 | 说明 |
| --- | --- |
| 版本管理 | `coolRuntimeGetVersion` / `coolDriverGetVersion` |
| 错误处理 | `coolGetErrorString` / `coolGetErrorName` / `coolGetLastError` / `coolPeekAtLastError` |
| 设备管理 | `coolGetDeviceCount` / `coolSetDevice` / `coolGetDevice` / `coolGetDeviceProperties` / `coolDeviceSynchronize` / `coolDeviceReset` |
| 内存管理 | `coolMalloc` / `coolFree` / `coolMemcpy` / `coolMemset` / `coolMemGetInfo` / `coolMallocHost` / `coolFreeHost` |
| 流管理 | `coolStreamCreate` / `coolStreamCreateWithFlags` / `coolStreamCreateWithPriority` / `coolStreamDestroy` / `coolStreamSynchronize` / `coolStreamQuery` / `coolStreamWaitEvent` |
| 事件管理 | `coolEventCreate` / `coolEventCreateWithFlags` / `coolEventDestroy` / `coolEventRecord` / `coolEventSynchronize` / `coolEventQuery` / `coolEventElapsedTime` |
| 模块与内核 | `coolModuleLoadData` / `coolModuleLoadFromFile` / `coolModuleGetFunction` / `coolModuleUnload` / `coolLaunchKernel` / `coolOccupancyMaxActiveBlocksPerMultiprocessor` / `coolOccupancyMaxPotentialBlockSize` |
| 设备符号 | `coolMemcpyToSymbol` / `coolMemcpyFromSymbol` / `coolGetSymbolAddress` / `coolGetSymbolSize`（含 `cudaMemcpy*Symbol` 兼容别名） |

内存分配的设备/主机指针、流/事件句柄、以及模块/函数句柄，均由 cuda 后端内的注册表跟踪，用于校验释放/销毁并识别重复操作（宿主并发原语只允许出现在后端，见 [ADR 0002](docs/adr/0002-memory-metadata-registry-placement.md)、[ADR 0003](docs/adr/0003-stream-event-handle-registry-placement.md) 与 [ADR 0004](docs/adr/0004-kernel-module-api-placement.md)）。

## 构建与测试

依赖：CMake ≥ 3.15、支持 C++20 的 GCC/Clang，以及 CUDA Driver API（`cuda.h` 与 `libcuda`，随 NVIDIA 驱动安装，**不需要 CUDA Toolkit**）。cuda 后端是唯一后端，也是构建的硬依赖；若检测不到 `cuda.h` 或 `libcuda`，CMake 配置阶段会直接报错。

```bash
make build      # 配置并编译
make test       # 构建并运行 CTest
make fmt        # clang-format 格式化
make check      # fmt-check + build + test（pre-commit / CI 使用）
```

或直接使用 CMake：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### 打包为 Debian 包

```bash
make package          # 生成 .deb 到 build-package/
make clean-package    # 清理打包目录
```

`make package` 用 **Release** 构建（不污染 `build/`）。安装布局对齐 CUDA
（`/usr/local/cuda/{include,lib64}`），前缀为 **`/usr/local/cool`**：

| 包 | 内容 |
| --- | --- |
| `libcoolrt0` | `/usr/local/cool/lib64/libcoolrt.so.0` 与实体 `libcoolrt.so.0.1.0` |
| `libcoolrt-dev` | `/usr/local/cool/include/cool/*.h`、`lib64/libcoolrt_static.a`、`lib64/libcoolrt.so` 链接、`lib64/pkgconfig/coolrt.pc` |

> 共享库带 soname（`libcoolrt.so.0`），与 CUDA 的 `libcudart.so.12` 同构；裸名 `libcoolrt.so`
> 只由开发包提供，供 `-lcoolrt` 链接使用。

安装与使用：

```bash
sudo dpkg -i libcoolrt0_0.1.0_amd64.deb libcoolrt-dev_0.1.0_amd64.deb

export PKG_CONFIG_PATH=/usr/local/cool/lib64/pkgconfig
pkg-config --libs coolrt                     # -L/usr/local/cool/lib64 -lcoolrt
gcc app.c $(pkg-config --cflags --libs coolrt) -o app

# 静态链接：需补 -lstdc++（库内部是 C++ 实现，与 libcudart_static.a 同理）
gcc app.c -I/usr/local/cool/include /usr/local/cool/lib64/libcoolrt_static.a \
    -lcuda -lstdc++ -o app
```

运行时需要把 `/usr/local/cool/lib64` 加入动态库搜索路径（`LD_LIBRARY_PATH`，或写进
`/etc/ld.so.conf.d/coolrt.conf` 后 `ldconfig`）——与 CUDA 的 `/usr/local/cuda/lib64` 同理。

包依赖：运行时包 `Depends: libc6`、`Recommends: nvidia-driver | cuda-drivers`
（`libcuda.so.1` 由 NVIDIA 驱动提供、不在 Debian 仓库里，因此不写成硬依赖，与 cudart 一致）。

## 后端

cuda（CUDA Driver API）是唯一后端，无需选择。初始化失败时运行时**不做自动回退**，直接返回错误码。其它配置项见 [.env.example](.env.example)。

## 许可证

本项目基于 [MIT](LICENSE) 许可证发布。
