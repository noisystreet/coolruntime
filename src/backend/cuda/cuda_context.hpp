// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * CUDA 后端的上下文与生命周期（内部头）。
 *
 * Driver API 使用显式上下文，而 Runtime API 是隐式的：init() 里执行
 * cuInit + cuCtxCreate + cuCtxSetCurrent，把上下文管理完全收敛在后端内部，
 * 对 R 层透明。
 *
 * 幂等性由 backend/bootstrap 保证（它只在首次调用时执行 init），这里的
 * 「已就绪」判断是防御性的。
 */
#ifndef COOL_BACKEND_CUDA_CONTEXT_HPP
#define COOL_BACKEND_CUDA_CONTEXT_HPP

#include "cool/cool_types.h"

namespace cool {
namespace cuda {

/* 初始化驱动并创建上下文。 */
coolError_t init();

/* 清空注册表并销毁上下文。 */
coolError_t finalize();

/* 把当前线程绑定到本后端上下文；尚未初始化时返回 coolErrorNotReady。 */
coolError_t ensureContext();

}  // namespace cuda
}  // namespace cool

#endif /* COOL_BACKEND_CUDA_CONTEXT_HPP */
