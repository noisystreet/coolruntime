// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * A 层模块/内核接口：参数校验后薄透传 BAL。
 * 模块/函数句柄的生存期与校验由 cuda 后端内部承担（见 ADR 0004）。
 * coolLaunchKernel 即启动跳板：args 直接透传，不做参数类型推导。
 */
#include "cool/cool_runtime_api.h"

#include "core/error.hpp"
#include "core/lifecycle.hpp"

extern "C" coolError_t coolModuleLoadData(coolModule_t* module, const void* image) {
    if (module == nullptr || image == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_ENTER(moduleLoadData);
    COOL_API_GUARD(backend->moduleLoadData(module, image));
    return coolSuccess;
}

extern "C" coolError_t coolModuleLoadFromFile(coolModule_t* module, const char* path) {
    if (module == nullptr || path == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_ENTER(moduleLoadFromFile);
    COOL_API_GUARD(backend->moduleLoadFromFile(module, path));
    return coolSuccess;
}

extern "C" coolError_t coolModuleGetFunction(coolKernel_t* kernel, coolModule_t module,
                                             const char* name) {
    if (kernel == nullptr || module == nullptr || name == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_ENTER(moduleGetFunction);
    COOL_API_GUARD(backend->moduleGetFunction(kernel, module, name));
    return coolSuccess;
}

extern "C" coolError_t coolModuleUnload(coolModule_t module) {
    if (module == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_ENTER(moduleUnload);
    COOL_API_GUARD(backend->moduleUnload(module));
    return coolSuccess;
}

extern "C" coolError_t coolLaunchKernel(coolKernel_t kernel, dim3 grid, dim3 block, void** args,
                                        size_t sharedMem, coolStream_t stream) {
    if (kernel == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_ENTER(launchKernel);
    COOL_API_GUARD(backend->launchKernel(kernel, grid, block, args, sharedMem, stream));
    return coolSuccess;
}

extern "C" coolError_t coolOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks,
                                                                     coolKernel_t kernel,
                                                                     int blockSize,
                                                                     size_t dynamicSMemSize) {
    if (numBlocks == nullptr || kernel == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_ENTER(occupancyMaxActiveBlocksPerMultiprocessor);
    COOL_API_GUARD(backend->occupancyMaxActiveBlocksPerMultiprocessor(numBlocks, kernel, blockSize,
                                                                      dynamicSMemSize));
    return coolSuccess;
}

extern "C" coolError_t coolOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize,
                                                          coolKernel_t kernel,
                                                          size_t dynamicSMemSize,
                                                          int blockSizeLimit) {
    if (minGridSize == nullptr || blockSize == nullptr || kernel == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_ENTER(occupancyMaxPotentialBlockSize);
    COOL_API_GUARD(backend->occupancyMaxPotentialBlockSize(minGridSize, blockSize, kernel,
                                                           dynamicSMemSize, blockSizeLimit));
    return coolSuccess;
}
