# AGENTS.md

## Project Identity

- Project: COOLRuntime
- Purpose: 教学 / 研究用的**类 CUDA** 运行时 API 参考实现，公开前缀 `cool`
- Tech stack: CMake（≥ 3.15）、公开层 C ABI、实现层 C++20、CTest、clang-format、clang-tidy
- Target runtime/platform: Linux（x86_64），GCC/Clang；CUDA Driver API（libcuda，必需）

## Directory Overview

```text
coolruntime/
├── CMakeLists.txt
├── include/cool/        # 公开头文件（C ABI）
├── src/
│   ├── api/             # A 层：公开函数实现与参数校验
│   ├── core/            # R 层：错误状态、生命周期、注册表
│   ├── mem/  exec/  kernel/   # R 层后续模块（P1+）
│   └── backend/         # BAL 与后端实现（cuda）
├── tests/               # unit/ conformance/ abi/
├── samples/             # 可运行示例
└── docs/                # 架构文档与 ADR
```

## Hard Constraints

- Module dependency direction（单向）：`api` → `core` → `backend`（BAL）→ 后端实现。下层不得反向依赖上层。
- **R 层（`src/core`、`src/api`、`src/mem`、`src/exec`、`src/kernel`）禁止**使用 `std::thread`、`std::mutex`、`std::atomic`、`std::condition_variable`、`malloc`、`pthread` 等宿主并发/专有调用；这些只允许出现在 `src/backend/**`。
- 公开头文件（`include/cool/*.h`）必须保持 C 兼容：只用 C 类型，以 `extern "C"` 声明，禁止 STL / 模板 / 引用 / C++ 关键字泄漏。
- 全库以 `-std=c++20` 编译，不允许低版本 C++ 回退。
- Architecture docs that require human approval before editing:
  - `docs/ARCHITECTURE.md`、`docs/adr/**`。
- Security red lines:
  - Do not commit secrets, tokens, certificates, cookies, or private keys.
  - Do not bypass authentication, authorization, or input validation.

## Required Commands

```bash
make check      # fmt-check + build + test（提交前必跑）
make test       # 构建并运行全部测试
make lint       # clang-tidy（未安装时跳过）
```

## Coding Rules

- 优先遵循仓库既有风格；格式化由 `.clang-format` 统一。
- 公开符号用 `cool` 前缀；内部符号置于 `cool::` 命名空间或 `cool_` 前缀。
- 优先 RAII（`std::unique_ptr` + 自定义 deleter）；使用固定宽度整型，避免隐式窄化。
- 行为变更必须有测试覆盖；公开 API 与数据兼容性需显式说明。

## Documentation Rules

- API 变更同步更新公开头文件注释与 README / docs。
- 架构变更更新 `docs/ARCHITECTURE.md` 或新增 ADR。
- 用户可见变更更新 `CHANGELOG.md`。
