// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * R 层生命周期接口：只是 backend/bootstrap 的转发。
 *
 * 为什么状态管理不放在这一层：AGENTS.md 禁止 R 层使用宿主并发原语，而
 * 「首次并发调用不得重复初始化」必须靠同步原语保证（重复 cuCtxCreate 会
 * 泄漏 context）。状态机与同步都放在 src/backend/bootstrap.cpp。
 */
#include "core/lifecycle.hpp"

#include "backend/bootstrap.hpp"

namespace cool {

coolError_t ensureInitialized() {
    return bootstrap::ensureReady();
}

const coolBackend* activeBackend() {
    return bootstrap::backend();
}

void installBackendForTesting(const coolBackend* backend) {
    bootstrap::installForTesting(backend);
}

}  // namespace cool
