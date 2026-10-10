# Contributing

感谢为 CoolRuntime 贡献代码。

## Setup

```bash
# 安装 pre-commit 并启用本地钩子（可选但推荐）
python -m pip install pre-commit
pre-commit install

make check
make test
```

## Development Flow

1. 从 `main` 创建短生命周期分支。
2. 聚焦单一变更；行为变更需附带测试。
3. 提交前运行统一质量入口 `make check`。
4. 打开 PR 时说明动机、变更摘要与简短测试计划。

## Commit And PR Expectations

- 推荐使用 Conventional Commits。
- PR 保持可评审；较大改动尽量拆分。
- 用户可见行为变更同步更新文档与 CHANGELOG。
- 不要提交密钥、真实凭据或机器相关路径。

## 后端新增约定

- 新后端必须完整定义 `src/backend/cool_backend.h` 的 `coolBackend` 契约（未实现的能力置 `NULL`）。
- 硬件/宿主专有调用只能出现在后端实现内，R 层必须保持后端无关。

## Security

请勿在公开 issue 中报告安全漏洞，遵循 [SECURITY.md](SECURITY.md) 私下披露。
