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
 * P6 R2 起有了独立的「内核执行期」错误码，原先只能保留 Unknown 的那批
 * （LAUNCH_FAILED / ILLEGAL_ADDRESS / LAUNCH_TIMEOUT / LAUNCH_OUT_OF_RESOURCES 等）
 * 已按语义映射；原则不变：**绝不把内核跑炸了报成 coolErrorInvalidValue**，
 * 那会把「kernel 出错」误导成「参数传错了」。
 *
 * 仍保留 Unknown 的：权限/驱动状态类（CUDA_ERROR_NOT_PERMITTED /
 * INSUFFICIENT_LOADER_VERSION 等）与本仓未收录的码 —— 宁可报 unknown，
 * 也不猜一个语义相近的码。
 *
 * 映射按**枚举符号**而非数值：coolError_t 的少数既有数值与 cudart 不同
 * （见 cool_types.h 的「已知数值差异」），这不影响本表正确性。
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

    /* 设备：序号非法与「没有可用设备」必须可区分（P6 R2） */
    {CUDA_ERROR_INVALID_DEVICE, coolErrorInvalidDevice},
    {CUDA_ERROR_NO_DEVICE, coolErrorNoDevice},

    /* 句柄 */
    {CUDA_ERROR_INVALID_HANDLE, coolErrorInvalidResourceHandle},

    /* 页锁定主机内存：注册状态与 cudart 保持一一对应 */
    {CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED, coolErrorHostMemoryAlreadyRegistered},
    {CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED, coolErrorHostMemoryNotRegistered},

    /* 就绪/支持性：初始化与反初始化都归为「后端未就绪」 */
    {CUDA_ERROR_NOT_READY, coolErrorNotReady},
    {CUDA_ERROR_NOT_INITIALIZED, coolErrorNotReady},
    {CUDA_ERROR_DEINITIALIZED, coolErrorNotReady},
    {CUDA_ERROR_NOT_SUPPORTED, coolErrorNotSupported},

    /* 上下文：传进来的上下文不对（仍属参数类） */
    {CUDA_ERROR_INVALID_CONTEXT, coolErrorInvalidValue},
    /* PTX 非法：hacc 生成的映像有问题，与「参数」不同（P6 R2 收窄） */
    {CUDA_ERROR_INVALID_PTX, coolErrorInvalidKernelImage},

    /* 镜像 / 源码 / 文件（P6 R2：有专码后不再笼统归为 InvalidValue） */
    {CUDA_ERROR_INVALID_IMAGE, coolErrorInvalidKernelImage},
    {CUDA_ERROR_INVALID_SOURCE, coolErrorInvalidSource},
    {CUDA_ERROR_FILE_NOT_FOUND, coolErrorFileNotFound},

    /* 内核执行期（P6 R2 新增映射） */
    {CUDA_ERROR_ILLEGAL_ADDRESS, coolErrorIllegalAddress},
    {CUDA_ERROR_MISALIGNED_ADDRESS, coolErrorMisalignedAddress},
    {CUDA_ERROR_INVALID_ADDRESS_SPACE, coolErrorInvalidAddressSpace},
    {CUDA_ERROR_INVALID_PC, coolErrorInvalidPc},
    {CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, coolErrorLaunchOutOfResources},
    {CUDA_ERROR_LAUNCH_TIMEOUT, coolErrorLaunchTimeout},
    {CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING, coolErrorLaunchIncompatibleTexturing},
    {CUDA_ERROR_LAUNCH_FAILED, coolErrorLaunchFailure},
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
