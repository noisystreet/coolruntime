# ADR: 实现语言与 ABI 边界（公开 C + 内部 C++20）

## Status

Accepted

## Context

CoolRuntime 提供一套**类 CUDA** 的运行时 API（命名与 cudart 的 `cudaXxx` 约定相接近）。cudart 的公开接口是纯 C ABI（`cudaXxx`），可被 C、C++ 及 CUDA 示例直接调用；而实现其语义（上下文、流队列、错误状态、后端起落）用现代 C++ 能显著提升可读性与安全性。

需要明确：公开头文件的 ABI 形态、内部实现语言，以及两者的边界。

## Decision

- **公开层**（`include/cool/cool_version.h`、`cool_types.h`、`cool_runtime_api.h`）保持 **C ABI**：只用 C 类型，以 `extern "C"` 声明，禁止 STL/模板/引用，可被 C 编译器（`-std=c11`）独立编译。
- **内部实现**（R 层、BAL、各后端）使用 **C++20**（`-std=c++20`）。
- 两层仅通过公开 C 符号边界交互；句柄用不透明指针（`typedef struct coolStream_st* coolStream_t;`）。

## Consequences

Positive:

- 上层示例可用 C 编译，与 cudart 用法一致，教学对照成本低。
- 内部可用 RAII、`std::atomic`、`std::chrono`、`std::jthread` 等现代 C++ 设施，代码更安全可读。
- C ABI 边界稳定，便于未来提供 C++ 封装门面。

Negative / trade-offs:

- 需要维护一条清晰的「C 兼容」红线，防止 C++ 类型泄漏进公开头。
- 通过 CI 增加 C ABI 编译校验（`-std=c11` 独立编译公开头）以强制该边界。

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| 全 C（C11）实现 | ABI 最简单、无 C++ 依赖 | 并发/资源管理需手写，可读性与安全性差 |
| 全 C++（含公开头） | 实现统一 | 公开头不再 C 兼容，示例需 C++，背离 cudart 用法 |
| 公开 C + 内部 C++20（本决策） | 兼顾兼容性与实现质量 | 需维护语言边界校验 |
