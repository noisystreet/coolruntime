// SPDX-FileCopyrightText: 2026 CoolRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * R 层错误状态（内部头）。
 *
 * last error 采用线程局部存储（TLS），语义与 cudart 一致：
 *   - 每次 API 调用成功/失败都会刷新 TLS；
 *   - coolGetLastError 读取并清空；coolPeekAtLastError 只读取。
 */
#ifndef COOL_CORE_ERROR_HPP
#define COOL_CORE_ERROR_HPP

#include "cool/cool_types.h"

namespace cool {

void setLastError(coolError_t err);
coolError_t lastError();     /* 读取并清空 */
coolError_t peekLastError(); /* 只读取 */

const char* errorName(coolError_t err);
const char* errorString(coolError_t err);

}  // namespace cool

/*
 * API 层卫语句：把内部返回码写入 TLS，失败时立即从当前函数返回。
 * 传入表达式须为 coolError_t。
 */
#define COOL_API_GUARD(expr)                        \
    do {                                            \
        const coolError_t cool_guard_err_ = (expr); \
        if (cool_guard_err_ != coolSuccess) {       \
            ::cool::setLastError(cool_guard_err_);  \
            return cool_guard_err_;                 \
        }                                           \
    } while (0)

/* 直接以给定错误码返回，并同步 TLS。 */
#define COOL_API_RETURN(err)       \
    do {                           \
        ::cool::setLastError(err); \
        return (err);              \
    } while (0)

#endif /* COOL_CORE_ERROR_HPP */
