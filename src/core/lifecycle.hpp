// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * R 层生命周期管理（内部头）。
 *
 * 职责：完成一次性的懒初始化，初始化唯一的 cuda 后端。
 * 后端不可用或初始化失败时不做自动回退。
 *
 * 约束：本文件属于 R 层，禁止使用 std::thread/std::mutex/std::atomic 等；
 * 一次性初始化通过函数局部静态量实现。
 */
#ifndef COOL_CORE_LIFECYCLE_HPP
#define COOL_CORE_LIFECYCLE_HPP

#include "backend/cool_backend.h"
#include "cool/cool_types.h"

namespace cool {

/* 懒初始化：选择并初始化后端。成功返回 coolSuccess，失败返回对应错误码。 */
coolError_t ensureInitialized();

/* 返回已初始化的后端；须在 ensureInitialized() 成功后调用。 */
const coolBackend* activeBackend();

}  // namespace cool

#endif /* COOL_CORE_LIFECYCLE_HPP */
