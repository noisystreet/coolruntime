// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

#include "backend/cuda/cuda_result_map.hpp"

namespace cool {
namespace cuda {
namespace {

struct Mapping {
    CUresult result;
    coolError_t error;
};

/*
 * 只映射语义明确等价的错误码。
 *
 * 刻意**不**列入的（保留为 coolErrorUnknown）：
 *   - 内核运行期错误：CUDA_ERROR_LAUNCH_FAILED / ILLEGAL_ADDRESS /
 *     LAUNCH_TIMEOUT / LAUNCH_OUT_OF_RESOURCES 等
 *   - 权限/驱动状态类：CUDA_ERROR_NOT_PERMITTED / INSUFFICIENT_LOADER_VERSION 等
 * 把它们硬映射成 coolErrorInvalidValue 会把「内核跑炸了」误导成「参数传错了」，
 * 比报 unknown 更糟。
 *
 * 注意 coolErrorMemoryAllocation 在这里没有对应项：Driver API 分配失败只有
 * CUDA_ERROR_OUT_OF_MEMORY，没有单独的 allocation 错误码（那是 Runtime 的
 * cudaErrorMemoryAllocation 才有的概念）。
 *
 * 用查表而非 switch：映射条目会持续增长，而一个巨型 switch 的圈复杂度会触到
 * 项目的 ccn 门限。分开之后加一行数据即可，逻辑不必动。
 */
constexpr Mapping kMappings[] = {
    /* 参数与内存 */
    {CUDA_SUCCESS, coolSuccess},
    {CUDA_ERROR_INVALID_VALUE, coolErrorInvalidValue},
    {CUDA_ERROR_OUT_OF_MEMORY, coolErrorOutOfMemory},

    /* 设备 */
    {CUDA_ERROR_INVALID_DEVICE, coolErrorInvalidDevice},
    {CUDA_ERROR_NO_DEVICE, coolErrorInvalidDevice},

    /* 句柄 */
    {CUDA_ERROR_INVALID_HANDLE, coolErrorInvalidResourceHandle},

    /* 就绪/支持性：初始化与反初始化都归为「后端未就绪」 */
    {CUDA_ERROR_NOT_READY, coolErrorNotReady},
    {CUDA_ERROR_NOT_INITIALIZED, coolErrorNotReady},
    {CUDA_ERROR_DEINITIALIZED, coolErrorNotReady},
    {CUDA_ERROR_NOT_SUPPORTED, coolErrorNotSupported},

    /* 上下文与模块镜像：均属「传进来的东西不对」 */
    {CUDA_ERROR_INVALID_CONTEXT, coolErrorInvalidValue},
    {CUDA_ERROR_INVALID_IMAGE, coolErrorInvalidValue},
    {CUDA_ERROR_INVALID_SOURCE, coolErrorInvalidValue},
    {CUDA_ERROR_INVALID_PTX, coolErrorInvalidValue},
    {CUDA_ERROR_FILE_NOT_FOUND, coolErrorInvalidValue},
};

}  // namespace

coolError_t resultToError(CUresult res) {
    /* 只在出错路径上调用，十几项的线性查找开销可忽略。 */
    for (const Mapping& mapping : kMappings) {
        if (mapping.result == res) {
            return mapping.error;
        }
    }
    return coolErrorUnknown;
}

}  // namespace cuda
}  // namespace cool
