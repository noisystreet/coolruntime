// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * R 层生命周期接口（内部头）。
 *
 * 职责：转发到 backend/bootstrap —— 真正的初始化状态机与同步原语都在
 * src/backend/bootstrap.cpp（AGENTS.md 只允许后端层使用宿主并发原语）。
 * 本层保持无状态，便于审计。
 *
 * 历史：一次性初始化原先在本层用函数局部静态量实现，但缺乏同步——多线程
 * 首次并发调用会各自读到「未初始化」，重复执行 backend->init()（重复
 * cuCtxCreate 会泄漏 context）。该状态机已下移到后端层。
 */
#ifndef COOL_CORE_LIFECYCLE_HPP
#define COOL_CORE_LIFECYCLE_HPP

#include "backend/cool_backend.h"
#include "cool/cool_types.h"
#include "core/error.hpp"

namespace cool {

/* 懒初始化：选择并初始化后端。成功返回 coolSuccess，失败返回对应错误码。 */
coolError_t ensureInitialized();

/* 返回已初始化的后端；须在 ensureInitialized() 成功后调用。 */
const coolBackend* activeBackend();

/*
 * 测试专用：注入替代后端（传 nullptr 恢复默认 cuda 后端）。
 *
 * 用途：R 层的转发逻辑（参数校验、能力位检查、错误透传）与后端实现无关，
 * 用假后端即可在无 GPU 环境下完整覆盖。
 *
 * 约束：必须在**任何 cool API 调用之前**、单线程调用一次。
 * 生产代码不得使用。
 */
void installBackendForTesting(const coolBackend* backend);

}  // namespace cool

/*
 * 取「已初始化且支持指定能力」的后端——A 层每个转发函数的公共前置步骤。
 *
 * 展开后：先把 TLS 置为 coolSuccess（对齐 cudart：每次调用都刷新 last
 * error，成功即 success），再校验初始化与能力位；任何一步失败都会写入具体
 * 错误码并直接 return。成功时在当前作用域声明 `const coolBackend* backend`，
 * 且 `backend->member` 保证非空。这样「后端未就绪」与「能力未实现」各返回
 * 各的错误码（后者按 cool_backend.h 的 BAL 契约）。
 *
 * 用法：
 *   COOL_API_ENTER(malloc);
 *   COOL_API_GUARD(backend->malloc(ptr, size));
 */
#define COOL_API_ENTER(member)                            \
    ::cool::setLastError(coolSuccess);                    \
    COOL_API_GUARD(::cool::ensureInitialized());          \
    const coolBackend* backend = ::cool::activeBackend(); \
    if (backend == nullptr) {                             \
        COOL_API_RETURN(coolErrorNotReady);               \
    }                                                     \
    if ((backend)->member == nullptr) {                   \
        COOL_API_RETURN(coolErrorNotSupported);           \
    }

#endif /* COOL_CORE_LIFECYCLE_HPP */
