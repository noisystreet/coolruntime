// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

#include "backend/cuda/cuda_result_map.hpp"

namespace cool {
namespace cuda {

coolError_t resultToError(CUresult res) {
    switch (res) {
        case CUDA_SUCCESS:
            return coolSuccess;
        case CUDA_ERROR_INVALID_VALUE:
            return coolErrorInvalidValue;
        case CUDA_ERROR_OUT_OF_MEMORY:
            return coolErrorOutOfMemory;
        case CUDA_ERROR_INVALID_DEVICE:
            return coolErrorInvalidDevice;
        case CUDA_ERROR_INVALID_HANDLE:
            return coolErrorInvalidResourceHandle;
        case CUDA_ERROR_NOT_READY:
            return coolErrorNotReady;
        case CUDA_ERROR_NOT_SUPPORTED:
            return coolErrorNotSupported;
        default:
            return coolErrorUnknown;
    }
}

}  // namespace cuda
}  // namespace cool
