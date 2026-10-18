// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * CUDA Driver API 返回码到 coolError_t 的映射（内部头）。
 */
#ifndef COOL_BACKEND_CUDA_RESULT_MAP_HPP
#define COOL_BACKEND_CUDA_RESULT_MAP_HPP

#include <cuda.h>

#include "cool/cool_types.h"

namespace cool {
namespace cuda {

/*
 * CUresult → coolError_t。
 *
 * 未列出的驱动错误码收敛为 coolErrorUnknown：公开枚举是 ABI 的一部分，
 * 不把 libcuda 的错误码集合固化进去。
 */
coolError_t resultToError(CUresult res);

}  // namespace cuda
}  // namespace cool

#endif /* COOL_BACKEND_CUDA_RESULT_MAP_HPP */
