// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * 后端启动引导（内部头）。
 *
 * 职责：选定后端、完成**线程安全的**一次性初始化，并在进程退出时 finalize。
 *
 * 为什么这段放在 backend/ 而不是 core/：AGENTS.md 规定宿主并发原语
 * （std::mutex / std::atomic）只允许出现在后端层目录。而「多个线程首次
 * 并发调用时不得重复初始化」必须靠同步原语保证——重复执行 cuCtxCreate 会
 * 泄漏 CUDA context 并覆盖全局句柄。所以把整套状态机放在后端层，
 * R 层（core/lifecycle.*）退化为纯转发。
 */
#ifndef COOL_BACKEND_BOOTSTRAP_HPP
#define COOL_BACKEND_BOOTSTRAP_HPP

#include "backend/cool_backend.h"
#include "cool/cool_types.h"

namespace cool {
namespace bootstrap {

/* 线程安全的一次性初始化；之后重复调用直接返回首次结果（无锁快路径）。 */
coolError_t ensureReady();

/* 已初始化的后端；尚未就绪时为 nullptr。 */
const coolBackend* backend();

/*
 * 测试专用：注入替代后端并复位状态（传 nullptr 恢复默认 cuda 后端）。
 * 必须在任何 cool API 调用之前、单线程调用。
 */
void installForTesting(const coolBackend* backend);

}  // namespace bootstrap
}  // namespace cool

#endif /* COOL_BACKEND_BOOTSTRAP_HPP */
