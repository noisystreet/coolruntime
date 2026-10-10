// SPDX-FileCopyrightText: 2026 CoolRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * CUDA 后端：通过 CUDA Driver API（libcuda）实现类 CUDA 的运行时语义。
 *
 * 关键点：Driver API 使用显式上下文，而 Runtime API 是隐式的。
 * 这里在 init 中执行 cuInit + cuCtxCreate + cuCtxSetCurrent，把上下文管理
 * 完全收敛在 cuda 后端内部，对 R 层透明。
 *
 * CUDA Driver API（cuda.h + libcuda）是构建的硬依赖（见顶层 CMakeLists.txt）。
 */
#include <cuda.h>

#include <cstdio>
#include <cstring>

#include "../backend_registry.hpp"

namespace cool {
namespace {

CUcontext g_ctx = nullptr;
CUdevice g_dev = 0;
bool g_initialized = false;

coolError_t cuResultToCool(CUresult res) {
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

/* 确保当前线程绑定到本后端上下文。 */
coolError_t ensureContext(void) {
    if (g_ctx == nullptr) {
        return coolErrorNotReady;
    }
    return cuResultToCool(cuCtxSetCurrent(g_ctx));
}

coolError_t cudaInit(void) {
    if (g_initialized) {
        return coolSuccess;
    }

    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        return cuResultToCool(res);
    }

    int count = 0;
    res = cuDeviceGetCount(&count);
    if (res != CUDA_SUCCESS) {
        return cuResultToCool(res);
    }
    if (count <= 0) {
        return coolErrorInvalidDevice;
    }

    res = cuDeviceGet(&g_dev, 0);
    if (res != CUDA_SUCCESS) {
        return cuResultToCool(res);
    }

    res = cuCtxCreate(&g_ctx, nullptr, CU_CTX_SCHED_AUTO, g_dev);
    if (res != CUDA_SUCCESS) {
        g_ctx = nullptr;
        return cuResultToCool(res);
    }

    res = cuCtxSetCurrent(g_ctx);
    if (res != CUDA_SUCCESS) {
        cuCtxDestroy(g_ctx);
        g_ctx = nullptr;
        return cuResultToCool(res);
    }

    g_initialized = true;
    return coolSuccess;
}

coolError_t cudaFinalize(void) {
    if (g_ctx != nullptr) {
        cuCtxDestroy(g_ctx);
        g_ctx = nullptr;
    }
    g_initialized = false;
    return coolSuccess;
}

coolError_t cudaGetDeviceCount(int* count) {
    if (count == nullptr) {
        return coolErrorInvalidValue;
    }
    return cuResultToCool(cuDeviceGetCount(count));
}

coolError_t cudaGetDeviceProperties(int dev, coolDeviceProp* prop) {
    if (prop == nullptr) {
        return coolErrorInvalidValue;
    }

    CUdevice device = 0;
    CUresult res = cuDeviceGet(&device, dev);
    if (res != CUDA_SUCCESS) {
        return cuResultToCool(res);
    }

    std::memset(prop, 0, sizeof(*prop));
    res = cuDeviceGetName(prop->name, static_cast<int>(sizeof(prop->name)), device);
    if (res != CUDA_SUCCESS) {
        return cuResultToCool(res);
    }

    int value = 0;
    cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    prop->major = value;
    cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    prop->minor = value;
    cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
    prop->multiProcessorCount = value;
    cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
    prop->maxThreadsPerBlock = value;
    cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device);
    prop->maxThreadsPerMultiProcessor = value;
    cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device);
    prop->maxGridSize[0] = value;
    cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, device);
    prop->maxGridSize[1] = value;
    cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, device);
    prop->maxGridSize[2] = value;
    cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device);
    prop->warpSize = value;

    size_t totalBytes = 0;
    res = cuDeviceTotalMem(&totalBytes, device);
    if (res != CUDA_SUCCESS) {
        return cuResultToCool(res);
    }
    prop->totalGlobalMem = totalBytes;
    return coolSuccess;
}

coolError_t cudaDeviceSynchronize(void) {
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }
    return cuResultToCool(cuCtxSynchronize());
}

coolError_t cudaDeviceReset(void) {
    /* 销毁并重建上下文，对应 cudart 的 cudaDeviceReset 语义。 */
    cudaFinalize();
    return cudaInit();
}

coolError_t cudaDriverGetVersion(int* version) {
    if (version == nullptr) {
        return coolErrorInvalidValue;
    }
    return cuResultToCool(cuDriverGetVersion(version));
}

}  // namespace

const coolBackend* cudaBackend() {
    static const coolBackend backend = {
        "cuda",                   /* name */
        COOL_BACKEND_API_VERSION, /* apiVersion */
        cudaInit,                 /* init */
        cudaFinalize,             /* finalize */
        cudaGetDeviceCount,       /* getDeviceCount */
        cudaGetDeviceProperties,  /* getDeviceProperties */
        cudaDeviceSynchronize,    /* deviceSynchronize */
        cudaDeviceReset,          /* deviceReset */
        nullptr,                  /* malloc */
        nullptr,                  /* free */
        nullptr,                  /* mallocHost */
        nullptr,                  /* freeHost */
        nullptr,                  /* memcpy */
        nullptr,                  /* memset */
        nullptr,                  /* memGetInfo */
        nullptr,                  /* streamCreate */
        nullptr,                  /* streamDestroy */
        nullptr,                  /* streamSynchronize */
        nullptr,                  /* streamQuery */
        nullptr,                  /* streamWaitEvent */
        nullptr,                  /* eventCreate */
        nullptr,                  /* eventDestroy */
        nullptr,                  /* eventRecord */
        nullptr,                  /* eventSynchronize */
        nullptr,                  /* eventQuery */
        nullptr,                  /* eventElapsedTime */
        nullptr,                  /* launchKernel */
        cudaDriverGetVersion,     /* driverGetVersion */
    };
    return &backend;
}

}  // namespace cool
