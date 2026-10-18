// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

#include "backend/cuda/cuda_context.hpp"

#include <cuda.h>

#include "backend/cuda/cuda_registry.hpp"
#include "backend/cuda/cuda_result_map.hpp"

namespace cool {
namespace cuda {
namespace {

CUcontext g_ctx = nullptr;
CUdevice g_dev = 0;

}  // namespace

coolError_t ensureContext() {
    if (g_ctx == nullptr) {
        return coolErrorNotReady;
    }
    return resultToError(cuCtxSetCurrent(g_ctx));
}

coolError_t init() {
    /* 由 backend/bootstrap 保证只调用一次；此处以 context 句柄作为就绪判断（防御）。 */
    if (g_ctx != nullptr) {
        return coolSuccess;
    }

    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        return resultToError(res);
    }

    int count = 0;
    res = cuDeviceGetCount(&count);
    if (res != CUDA_SUCCESS) {
        return resultToError(res);
    }
    if (count <= 0) {
        return coolErrorInvalidDevice;
    }

    res = cuDeviceGet(&g_dev, 0);
    if (res != CUDA_SUCCESS) {
        return resultToError(res);
    }

    res = cuCtxCreate(&g_ctx, nullptr, CU_CTX_SCHED_AUTO, g_dev);
    if (res != CUDA_SUCCESS) {
        g_ctx = nullptr;
        return resultToError(res);
    }

    res = cuCtxSetCurrent(g_ctx);
    if (res != CUDA_SUCCESS) {
        cuCtxDestroy(g_ctx);
        g_ctx = nullptr;
        return resultToError(res);
    }

    /*
     * 预构造两个注册表：它们的析构在生命周期末尾由静态析构器执行。
     * bootstrap 在本函数返回后才注册 atexit 清理，故先在此处构造可保证
     * 「atexit 清理先执行、注册表后析构」，避免清理时访问已析构的静态对象。
     */
    allocations();
    handles();
    return coolSuccess;
}

coolError_t finalize() {
    allocations().clear();
    handles().clear();
    if (g_ctx != nullptr) {
        cuCtxDestroy(g_ctx);
        g_ctx = nullptr;
    }
    return coolSuccess;
}

}  // namespace cuda
}  // namespace cool
