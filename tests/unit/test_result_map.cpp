// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * CUresult → coolError_t 映射表测试（纯函数，无需 GPU）。
 *
 * 一半断言锁住「该映射的映射了」，另一半锁住「该保留 Unknown 的没被顺手映射」
 * —— 后者同样重要：把内核运行期错误硬报成 coolErrorInvalidValue 会把
 * 「kernel 跑炸了」误导成「参数传错了」。
 */
#include <cuda.h>

#include "backend/cuda/cuda_result_map.hpp"
#include "cool/cool_types.h"

#include "support/check.hpp"

int main() {
    using cool::cuda::resultToError;

    /* 成功与既有映射。 */
    COOL_CHECK_EQ(resultToError(CUDA_SUCCESS), coolSuccess);
    COOL_CHECK_EQ(resultToError(CUDA_ERROR_INVALID_VALUE), coolErrorInvalidValue);
    COOL_CHECK_EQ(resultToError(CUDA_ERROR_OUT_OF_MEMORY), coolErrorOutOfMemory);
    COOL_CHECK_EQ(resultToError(CUDA_ERROR_INVALID_DEVICE), coolErrorInvalidDevice);
    COOL_CHECK_EQ(resultToError(CUDA_ERROR_INVALID_HANDLE), coolErrorInvalidResourceHandle);
    COOL_CHECK_EQ(resultToError(CUDA_ERROR_NOT_READY), coolErrorNotReady);
    COOL_CHECK_EQ(resultToError(CUDA_ERROR_NOT_SUPPORTED), coolErrorNotSupported);

    /* 初始化状态：未初始化/已反初始化都归为「后端未就绪」。 */
    COOL_CHECK_EQ(resultToError(CUDA_ERROR_NOT_INITIALIZED), coolErrorNotReady);
    COOL_CHECK_EQ(resultToError(CUDA_ERROR_DEINITIALIZED), coolErrorNotReady);

    /* P6 R2：「没有可用设备」与「设备序号非法」必须可区分。 */
    COOL_CHECK_EQ(resultToError(CUDA_ERROR_NO_DEVICE), coolErrorNoDevice);
    COOL_CHECK_EQ(resultToError(CUDA_ERROR_INVALID_DEVICE), coolErrorInvalidDevice);

    /* 上下文仍是参数类；镜像/源码/文件有专码（P6 R2 收窄）。 */
    COOL_CHECK_EQ(resultToError(CUDA_ERROR_INVALID_CONTEXT), coolErrorInvalidValue);
    COOL_CHECK_EQ(resultToError(CUDA_ERROR_INVALID_IMAGE), coolErrorInvalidKernelImage);
    COOL_CHECK_EQ(resultToError(CUDA_ERROR_INVALID_PTX), coolErrorInvalidKernelImage);
    COOL_CHECK_EQ(resultToError(CUDA_ERROR_INVALID_SOURCE), coolErrorInvalidSource);
    COOL_CHECK_EQ(resultToError(CUDA_ERROR_FILE_NOT_FOUND), coolErrorFileNotFound);

    /* P6 R2 的核心收益：内核执行期错误不再与「参数传错了」混淆。 */
    COOL_CHECK_EQ(resultToError(CUDA_ERROR_LAUNCH_FAILED), coolErrorLaunchFailure);
    COOL_CHECK_EQ(resultToError(CUDA_ERROR_ILLEGAL_ADDRESS), coolErrorIllegalAddress);
    COOL_CHECK_EQ(resultToError(CUDA_ERROR_LAUNCH_TIMEOUT), coolErrorLaunchTimeout);
    COOL_CHECK_EQ(resultToError(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES), coolErrorLaunchOutOfResources);
    COOL_CHECK_EQ(resultToError(CUDA_ERROR_MISALIGNED_ADDRESS), coolErrorMisalignedAddress);
    COOL_CHECK_EQ(resultToError(CUDA_ERROR_INVALID_ADDRESS_SPACE), coolErrorInvalidAddressSpace);
    COOL_CHECK_EQ(resultToError(CUDA_ERROR_INVALID_PC), coolErrorInvalidPc);
    COOL_CHECK_EQ(resultToError(CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING),
                  coolErrorLaunchIncompatibleTexturing);

    /* 仍有意保留 Unknown：本仓未收录的码，宁可报 unknown 也不猜一个近义码。 */
    COOL_CHECK_EQ(resultToError(CUDA_ERROR_ILLEGAL_INSTRUCTION), coolErrorUnknown);

    /* 未来驱动新增的错误码同样收敛。 */
    COOL_CHECK_EQ(resultToError(static_cast<CUresult>(12345)), coolErrorUnknown);

    return cooltest::summary("test_result_map");
}
