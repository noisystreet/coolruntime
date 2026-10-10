# Architecture

## Purpose

CoolRuntime 是面向异构并行编程教学的**类 CUDA** 运行时 API 参考实现，服务对象是学习/研究者与参与实现的开发者。它以可读源码提供一套命名与用法接近 cudart 的运行时，并通过后端抽象层（BAL）把运行时语义与具体硬件解耦，当前提供 CUDA Driver API 后端（唯一后端）。

## Goals

- G1 类 CUDA 语义：核心子集函数在命名与用法上与 cudart 保持可对照的近似，便于教学对照。
- G2 后端解耦：硬件相关行为全部收敛到 BAL，运行时核心不含硬件假设。
- G3 真实设备可落地：后端基于 CUDA Driver API，可在真实 GPU 执行。
- G4 可观测：提供 trace 开关，打印 API 调用、调度与内核派发。
- G5 可测试：模块化单元测试框架。

## Non-Goals

- 不追求性能指标（带宽、延迟、占用率）。
- 不做 CUDA C++ 编译前端（PTX/cubin 由 `nvcc`/外部工具产出）。
- 阶段一不实现：图、纹理/表面、外部内存/信号量、内存池、peer 访问、驱动入口点。
- 不承诺与 cudart 的二进制 ABI 或行为完全兼容（仅提供类 CUDA 的命名与用法）。

## System Overview

```text
应用 / 示例 (samples, tests)
        |
        v
A 层：公开 API  (include/cool/cool_runtime_api.h)   —— C ABI
        |
        v
R 层：运行时核心 (src/core, mem, exec, kernel)      —— C++20
        |
        v
B 层：后端抽象层 BAL (src/backend/cool_backend.h)
        |
        v
后端实现：cuda (Driver API)
```

## Module Boundaries

| Module | Responsibility | May Depend On |
|--------|----------------|---------------|
| `src/api` | 公开函数实现、参数校验、错误落地 | `core`、`backend`（仅接口） |
| `src/core` | 错误状态、生命周期、设备/句柄注册表、追踪 | `backend`（仅接口） |
| `src/mem` (P1) | 分配器、内存元数据、拷贝/填充 | `core`、`backend` |
| `src/exec` (P2) | 流、事件、调度器 | `core`、`backend` |
| `src/kernel` (P3) | 内核注册表、启动跳板、占用率 | `core`、`backend` |
| `src/backend` | BAL 契约与后端访问器 | — |
| `src/backend/cuda` | CUDA Driver API 后端（`cuXxx`） | `backend`（BAL 契约） |

依赖方向单向：`api` → `core` → `backend`(BAL) → 后端实现。R 层禁止使用宿主并发/专有调用（见 AGENTS.md 硬约束）。

## Data Model

- 设备：`int` 逻辑设备 id；阶段一暴露逻辑设备 0，保留多设备接口。
- 上下文：每设备一个隐式主上下文，运行时自动创建；cuda 后端内部即 `CUcontext`。
- 流：`coolStream_t`（不透明指针），`NULL` = 默认流。
- 事件：`coolEvent_t`（不透明指针）。
- 内存块：`void*` 设备指针，内部记录分配元数据。
- 内核：`coolKernel_t`（不透明指针）；cuda 后端内部即 `CUfunction`。
- 句柄统一为不透明指针，结构体定义仅在内部头可见。

## Configuration

配置优先级：

1. 命令行参数（示例程序自定义）
2. 环境变量（`COOL_TRACE`）
3. 配置文件（暂无）
4. 默认值

## Security Notes

- Trust boundaries：本库为进程内库，不处理网络/外部输入；公开 API 的参数校验是第一道边界。
- Sensitive data：无持久化敏感数据；`.env` 不入库。
- Authentication/authorization：不适用。

## Open Decisions

| Decision | Options | Owner | Due |
|----------|---------|-------|-----|
| 一致性对照测试框架形态 | 独立 target / 编译期双实现 | maintainers | P4 |
| `COOL_STRICT_DEFAULT_STREAM` 严格模式是否默认开启 | 默认关 / 默认开 | maintainers | P2 |
