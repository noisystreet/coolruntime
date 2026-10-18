// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * CUDA 后端的 BAL 适配层：把 coolBackend 的能力位逐个映射到 Driver API。
 *
 * 其余职责已拆出：
 *   - 上下文与生命周期 → cuda_context.{hpp,cpp}
 *   - 内存/句柄注册表   → cuda_registry.{hpp,cpp}
 *   - CUresult 映射     → cuda_result_map.{hpp,cpp}
 *
 * 每个适配函数的固定套路：参数校验 → 句柄有效性校验（查注册表）→ 调 Driver
 * API → 把 CUresult 转成 coolError_t。
 */
#include <cuda.h>

#include <cstdint>
#include <cstring>

#include "backend/backend_registry.hpp"
#include "backend/cuda/cuda_context.hpp"
#include "backend/cuda/cuda_registry.hpp"
#include "backend/cuda/cuda_result_map.hpp"

namespace cool {
namespace {

using cuda::allocations;
using cuda::ensureContext;
using cuda::handleAddr;
using cuda::handles;
using cuda::resultToError;

coolError_t cudaGetDeviceCount(int* count) {
    if (count == nullptr) {
        return coolErrorInvalidValue;
    }
    return resultToError(cuDeviceGetCount(count));
}

coolError_t cudaGetDeviceProperties(int dev, coolDeviceProp* prop) {
    if (prop == nullptr) {
        return coolErrorInvalidValue;
    }

    CUdevice device = 0;
    CUresult res = cuDeviceGet(&device, dev);
    if (res != CUDA_SUCCESS) {
        return resultToError(res);
    }

    std::memset(prop, 0, sizeof(*prop));
    res = cuDeviceGetName(prop->name, static_cast<int>(sizeof(prop->name)), device);
    if (res != CUDA_SUCCESS) {
        return resultToError(res);
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
        return resultToError(res);
    }
    prop->totalGlobalMem = totalBytes;
    return coolSuccess;
}

coolError_t cudaDeviceSynchronize(void) {
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }
    return resultToError(cuCtxSynchronize());
}

coolError_t cudaDeviceReset(void) {
    /* 销毁并重建上下文，对应 cudart 的 cudaDeviceReset 语义。 */
    cuda::finalize();
    return cuda::init();
}

coolError_t cudaDriverGetVersion(int* version) {
    if (version == nullptr) {
        return coolErrorInvalidValue;
    }
    return resultToError(cuDriverGetVersion(version));
}

coolError_t cudaMalloc(void** ptr, size_t size) {
    if (ptr == nullptr) {
        return coolErrorInvalidValue;
    }
    *ptr = nullptr;
    if (size == 0) {
        return coolErrorInvalidValue;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }

    CUdeviceptr devicePtr = 0;
    const CUresult res = cuMemAlloc(&devicePtr, size);
    if (res != CUDA_SUCCESS) {
        return resultToError(res);
    }
    allocations().addDevice(static_cast<std::uintptr_t>(devicePtr), size);
    *ptr = reinterpret_cast<void*>(devicePtr);
    return coolSuccess;
}

coolError_t cudaFree(void* ptr) {
    if (ptr == nullptr) {
        return coolSuccess;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }

    const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    if (!allocations().hasDevice(addr)) {
        return coolErrorInvalidValue;
    }
    const CUresult res = cuMemFree(static_cast<CUdeviceptr>(addr));
    if (res != CUDA_SUCCESS) {
        return resultToError(res);
    }
    allocations().removeDevice(addr);
    return coolSuccess;
}

coolError_t cudaMemcpy(void* dst, const void* src, size_t count, coolMemcpyKind kind) {
    if (dst == nullptr || src == nullptr) {
        return coolErrorInvalidValue;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }

    switch (kind) {
        case coolMemcpyHostToHost:
            std::memcpy(dst, src, count);
            return coolSuccess;
        case coolMemcpyHostToDevice:
            return resultToError(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(dst), src, count));
        case coolMemcpyDeviceToHost:
            return resultToError(cuMemcpyDtoH(dst, reinterpret_cast<CUdeviceptr>(src), count));
        case coolMemcpyDeviceToDevice:
            return resultToError(cuMemcpyDtoD(reinterpret_cast<CUdeviceptr>(dst),
                                              reinterpret_cast<CUdeviceptr>(src), count));
        case coolMemcpyDefault:
            return resultToError(cuMemcpy(reinterpret_cast<CUdeviceptr>(dst),
                                          reinterpret_cast<CUdeviceptr>(src), count));
        default:
            return coolErrorInvalidValue;
    }
}

coolError_t cudaMemset(void* ptr, int value, size_t count) {
    if (ptr == nullptr) {
        return coolErrorInvalidValue;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }
    return resultToError(
        cuMemsetD8(reinterpret_cast<CUdeviceptr>(ptr), static_cast<unsigned char>(value), count));
}

coolError_t cudaMemGetInfo(size_t* freeBytes, size_t* totalBytes) {
    if (freeBytes == nullptr || totalBytes == nullptr) {
        return coolErrorInvalidValue;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }
    return resultToError(cuMemGetInfo(freeBytes, totalBytes));
}

/*
 * M3.2：把 coolSymbolDesc 解析为设备符号地址与大小。
 * symbol 为 coolSymbolDesc 地址；校验描述符、模块句柄与符号名。
 */
coolError_t resolveSymbol(const void* symbol, CUdeviceptr* devPtr, size_t* size) {
    if (symbol == nullptr) {
        return coolErrorInvalidValue;
    }
    const auto* desc = static_cast<const coolSymbolDesc*>(symbol);
    if (desc->module == nullptr || desc->symbol_name == nullptr) {
        return coolErrorInvalidValue;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }
    if (!handles().hasModule(handleAddr(desc->module))) {
        return coolErrorInvalidValue;
    }

    CUdeviceptr addr = 0;
    size_t symSize = 0;
    const CUresult res = cuModuleGetGlobal(
        &addr, &symSize, reinterpret_cast<CUmodule>(desc->module), desc->symbol_name);
    if (res != CUDA_SUCCESS) {
        return resultToError(res);
    }
    if (devPtr != nullptr) {
        *devPtr = addr;
    }
    if (size != nullptr) {
        *size = symSize;
    }
    return coolSuccess;
}

coolError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t count, size_t offset,
                               coolMemcpyKind kind) {
    if (src == nullptr) {
        return coolErrorInvalidValue;
    }
    CUdeviceptr addr = 0;
    size_t symSize = 0;
    const coolError_t err = resolveSymbol(symbol, &addr, &symSize);
    if (err != coolSuccess) {
        return err;
    }
    /* 越界校验：offset + count 不得超过符号大小。 */
    if (offset > symSize || count > symSize - offset) {
        return coolErrorInvalidValue;
    }

    switch (kind) {
        case coolMemcpyHostToDevice:
            return resultToError(cuMemcpyHtoD(addr + offset, src, count));
        case coolMemcpyDeviceToDevice:
            return resultToError(
                cuMemcpyDtoD(addr + offset, reinterpret_cast<CUdeviceptr>(src), count));
        default:
            /* 符号位于设备侧，仅支持 H2D / D2D。 */
            return coolErrorInvalidValue;
    }
}

/*
 * 把设备符号拷出到主机/设备指针（对应 cudaMemcpyFromSymbol，与 cudaMemcpyToSymbol 对称）。
 *
 * symbol 为 coolSymbolDesc 地址；dst 为目标指针；kind 决定目标侧内存类别：
 * D2H 走 cuMemcpyDtoH，D2D 走 cuMemcpyDtoD。符号在设备侧，故 H2H/H2D/Default 一律
 * 视为非法（与 cudart 对 FromSymbol 的约束一致）。
 * 越界（offset + count > 符号大小）返回 coolErrorInvalidValue。
 */
coolError_t cudaMemcpyFromSymbol(void* dst, const void* symbol, size_t count, size_t offset,
                                 coolMemcpyKind kind) {
    if (dst == nullptr) {
        return coolErrorInvalidValue;
    }
    CUdeviceptr addr = 0;
    size_t symSize = 0;
    const coolError_t err = resolveSymbol(symbol, &addr, &symSize);
    if (err != coolSuccess) {
        return err;
    }
    /* 越界校验：offset + count 不得超过符号大小。 */
    if (offset > symSize || count > symSize - offset) {
        return coolErrorInvalidValue;
    }

    switch (kind) {
        case coolMemcpyDeviceToHost:
            return resultToError(cuMemcpyDtoH(dst, addr + offset, count));
        case coolMemcpyDeviceToDevice:
            return resultToError(
                cuMemcpyDtoD(reinterpret_cast<CUdeviceptr>(dst), addr + offset, count));
        default:
            /* 源是设备符号，仅支持 D2H / D2D。 */
            return coolErrorInvalidValue;
    }
}

coolError_t cudaGetSymbolAddress(void** devPtr, const void* symbol) {
    if (devPtr == nullptr) {
        return coolErrorInvalidValue;
    }
    *devPtr = nullptr;
    CUdeviceptr addr = 0;
    const coolError_t err = resolveSymbol(symbol, &addr, nullptr);
    if (err != coolSuccess) {
        return err;
    }
    *devPtr = reinterpret_cast<void*>(addr);
    return coolSuccess;
}

coolError_t cudaGetSymbolSize(size_t* size, const void* symbol) {
    if (size == nullptr) {
        return coolErrorInvalidValue;
    }
    *size = 0;
    size_t symSize = 0;
    const coolError_t err = resolveSymbol(symbol, nullptr, &symSize);
    if (err != coolSuccess) {
        return err;
    }
    *size = symSize;
    return coolSuccess;
}

coolError_t cudaMallocHost(void** ptr, size_t size, unsigned flags) {
    if (ptr == nullptr) {
        return coolErrorInvalidValue;
    }
    *ptr = nullptr;
    if (size == 0) {
        return coolErrorInvalidValue;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }

    void* hostPtr = nullptr;
    const CUresult res = cuMemHostAlloc(&hostPtr, size, flags);
    if (res != CUDA_SUCCESS) {
        return resultToError(res);
    }
    allocations().addHost(reinterpret_cast<std::uintptr_t>(hostPtr), size);
    *ptr = hostPtr;
    return coolSuccess;
}

coolError_t cudaFreeHost(void* ptr) {
    if (ptr == nullptr) {
        return coolSuccess;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }

    const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    if (!allocations().hasHost(addr)) {
        return coolErrorInvalidValue;
    }
    const CUresult res = cuMemFreeHost(ptr);
    if (res != CUDA_SUCCESS) {
        return resultToError(res);
    }
    allocations().removeHost(addr);
    return coolSuccess;
}

coolError_t cudaStreamCreate(coolStream_t* stream, unsigned flags, int priority) {
    if (stream == nullptr) {
        return coolErrorInvalidValue;
    }
    *stream = nullptr;
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }

    CUstream cuStream = nullptr;
    const CUresult res = cuStreamCreateWithPriority(&cuStream, flags, priority);
    if (res != CUDA_SUCCESS) {
        return resultToError(res);
    }
    handles().addStream(handleAddr(cuStream));
    *stream = reinterpret_cast<coolStream_t>(cuStream);
    return coolSuccess;
}

coolError_t cudaStreamDestroy(coolStream_t stream) {
    if (stream == nullptr) {
        return coolErrorInvalidValue;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }

    const auto addr = handleAddr(stream);
    if (!handles().hasStream(addr)) {
        return coolErrorInvalidValue;
    }
    const CUresult res = cuStreamDestroy(reinterpret_cast<CUstream>(stream));
    if (res != CUDA_SUCCESS) {
        return resultToError(res);
    }
    handles().removeStream(addr);
    return coolSuccess;
}

coolError_t cudaStreamSynchronize(coolStream_t stream) {
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }
    if (stream != nullptr && !handles().hasStream(handleAddr(stream))) {
        return coolErrorInvalidValue;
    }
    return resultToError(cuStreamSynchronize(reinterpret_cast<CUstream>(stream)));
}

coolError_t cudaStreamQuery(coolStream_t stream) {
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }
    if (stream != nullptr && !handles().hasStream(handleAddr(stream))) {
        return coolErrorInvalidValue;
    }
    return resultToError(cuStreamQuery(reinterpret_cast<CUstream>(stream)));
}

coolError_t cudaStreamWaitEvent(coolStream_t stream, coolEvent_t event) {
    if (event == nullptr) {
        return coolErrorInvalidValue;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }
    if (stream != nullptr && !handles().hasStream(handleAddr(stream))) {
        return coolErrorInvalidValue;
    }
    if (!handles().hasEvent(handleAddr(event))) {
        return coolErrorInvalidValue;
    }
    return resultToError(
        cuStreamWaitEvent(reinterpret_cast<CUstream>(stream), reinterpret_cast<CUevent>(event), 0));
}

coolError_t cudaEventCreate(coolEvent_t* event, unsigned flags) {
    if (event == nullptr) {
        return coolErrorInvalidValue;
    }
    *event = nullptr;
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }

    CUevent cuEvent = nullptr;
    const CUresult res = cuEventCreate(&cuEvent, flags);
    if (res != CUDA_SUCCESS) {
        return resultToError(res);
    }
    handles().addEvent(handleAddr(cuEvent));
    *event = reinterpret_cast<coolEvent_t>(cuEvent);
    return coolSuccess;
}

coolError_t cudaEventDestroy(coolEvent_t event) {
    if (event == nullptr) {
        return coolErrorInvalidValue;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }

    const auto addr = handleAddr(event);
    if (!handles().hasEvent(addr)) {
        return coolErrorInvalidValue;
    }
    const CUresult res = cuEventDestroy(reinterpret_cast<CUevent>(event));
    if (res != CUDA_SUCCESS) {
        return resultToError(res);
    }
    handles().removeEvent(addr);
    return coolSuccess;
}

coolError_t cudaEventRecord(coolEvent_t event, coolStream_t stream) {
    if (event == nullptr) {
        return coolErrorInvalidValue;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }
    if (!handles().hasEvent(handleAddr(event))) {
        return coolErrorInvalidValue;
    }
    if (stream != nullptr && !handles().hasStream(handleAddr(stream))) {
        return coolErrorInvalidValue;
    }
    return resultToError(
        cuEventRecord(reinterpret_cast<CUevent>(event), reinterpret_cast<CUstream>(stream)));
}

coolError_t cudaEventSynchronize(coolEvent_t event) {
    if (event == nullptr) {
        return coolErrorInvalidValue;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }
    if (!handles().hasEvent(handleAddr(event))) {
        return coolErrorInvalidValue;
    }
    return resultToError(cuEventSynchronize(reinterpret_cast<CUevent>(event)));
}

coolError_t cudaEventQuery(coolEvent_t event) {
    if (event == nullptr) {
        return coolErrorInvalidValue;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }
    if (!handles().hasEvent(handleAddr(event))) {
        return coolErrorInvalidValue;
    }
    return resultToError(cuEventQuery(reinterpret_cast<CUevent>(event)));
}

coolError_t cudaEventElapsedTime(float* ms, coolEvent_t start, coolEvent_t end) {
    if (ms == nullptr || start == nullptr || end == nullptr) {
        return coolErrorInvalidValue;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }
    if (!handles().hasEvent(handleAddr(start)) || !handles().hasEvent(handleAddr(end))) {
        return coolErrorInvalidValue;
    }
    return resultToError(
        cuEventElapsedTime(ms, reinterpret_cast<CUevent>(start), reinterpret_cast<CUevent>(end)));
}

coolError_t cudaModuleLoadData(coolModule_t* module, const void* image) {
    if (module == nullptr || image == nullptr) {
        return coolErrorInvalidValue;
    }
    *module = nullptr;
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }

    CUmodule cuModule = nullptr;
    const CUresult res = cuModuleLoadData(&cuModule, image);
    if (res != CUDA_SUCCESS) {
        return resultToError(res);
    }
    handles().addModule(handleAddr(cuModule));
    *module = reinterpret_cast<coolModule_t>(cuModule);
    return coolSuccess;
}

coolError_t cudaModuleLoadFromFile(coolModule_t* module, const char* path) {
    if (module == nullptr || path == nullptr) {
        return coolErrorInvalidValue;
    }
    *module = nullptr;
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }

    CUmodule cuModule = nullptr;
    const CUresult res = cuModuleLoad(&cuModule, path);
    if (res != CUDA_SUCCESS) {
        return resultToError(res);
    }
    handles().addModule(handleAddr(cuModule));
    *module = reinterpret_cast<coolModule_t>(cuModule);
    return coolSuccess;
}

coolError_t cudaModuleGetFunction(coolKernel_t* kernel, coolModule_t module, const char* name) {
    if (kernel == nullptr || module == nullptr || name == nullptr) {
        return coolErrorInvalidValue;
    }
    *kernel = nullptr;
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }

    const auto moduleAddr = handleAddr(module);
    if (!handles().hasModule(moduleAddr)) {
        return coolErrorInvalidValue;
    }

    CUfunction cuFunc = nullptr;
    const CUresult res = cuModuleGetFunction(&cuFunc, reinterpret_cast<CUmodule>(module), name);
    if (res != CUDA_SUCCESS) {
        return resultToError(res);
    }
    handles().addFunction(handleAddr(cuFunc), moduleAddr);
    *kernel = reinterpret_cast<coolKernel_t>(cuFunc);
    return coolSuccess;
}

coolError_t cudaModuleUnload(coolModule_t module) {
    if (module == nullptr) {
        return coolErrorInvalidValue;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }

    const auto moduleAddr = handleAddr(module);
    if (!handles().hasModule(moduleAddr)) {
        return coolErrorInvalidValue;
    }
    const CUresult res = cuModuleUnload(reinterpret_cast<CUmodule>(module));
    if (res != CUDA_SUCCESS) {
        return resultToError(res);
    }
    handles().removeModule(moduleAddr);
    return coolSuccess;
}

coolError_t cudaLaunchKernel(const coolKernel* kernel, dim3 grid, dim3 block, void** args,
                             size_t sharedMem, coolStream_t stream) {
    if (kernel == nullptr) {
        return coolErrorInvalidValue;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }
    if (!handles().hasFunction(handleAddr(kernel))) {
        return coolErrorInvalidValue;
    }
    if (stream != nullptr && !handles().hasStream(handleAddr(stream))) {
        return coolErrorInvalidValue;
    }

    CUfunction func = reinterpret_cast<CUfunction>(const_cast<coolKernel*>(kernel));
    return resultToError(cuLaunchKernel(func, grid.x, grid.y, grid.z, block.x, block.y, block.z,
                                        static_cast<unsigned int>(sharedMem),
                                        reinterpret_cast<CUstream>(stream), args, nullptr));
}

coolError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const coolKernel* kernel,
                                                          int blockSize, size_t dynamicSMemSize) {
    if (numBlocks == nullptr || kernel == nullptr) {
        return coolErrorInvalidValue;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }
    if (!handles().hasFunction(handleAddr(kernel))) {
        return coolErrorInvalidValue;
    }

    CUfunction func = reinterpret_cast<CUfunction>(const_cast<coolKernel*>(kernel));
    return resultToError(
        cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize));
}

coolError_t cudaOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize,
                                               const coolKernel* kernel, size_t dynamicSMemSize,
                                               int blockSizeLimit) {
    if (minGridSize == nullptr || blockSize == nullptr || kernel == nullptr) {
        return coolErrorInvalidValue;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }
    if (!handles().hasFunction(handleAddr(kernel))) {
        return coolErrorInvalidValue;
    }

    CUfunction func = reinterpret_cast<CUfunction>(const_cast<coolKernel*>(kernel));
    return resultToError(cuOccupancyMaxPotentialBlockSize(
        minGridSize, blockSize, func, [](int) -> size_t { return 0; }, dynamicSMemSize,
        blockSizeLimit));
}

}  // namespace

const coolBackend* cudaBackend() {
    static const coolBackend backend = {
        "cuda",                                        /* name */
        COOL_BACKEND_API_VERSION,                      /* apiVersion */
        cuda::init,                                    /* init */
        cuda::finalize,                                /* finalize */
        cudaGetDeviceCount,                            /* getDeviceCount */
        cudaGetDeviceProperties,                       /* getDeviceProperties */
        cudaDeviceSynchronize,                         /* deviceSynchronize */
        cudaDeviceReset,                               /* deviceReset */
        cudaMalloc,                                    /* malloc */
        cudaFree,                                      /* free */
        cudaMallocHost,                                /* mallocHost */
        cudaFreeHost,                                  /* freeHost */
        cudaMemcpy,                                    /* memcpy */
        cudaMemset,                                    /* memset */
        cudaMemGetInfo,                                /* memGetInfo */
        cudaMemcpyToSymbol,                            /* memcpyToSymbol */
        cudaMemcpyFromSymbol,                          /* memcpyFromSymbol */
        cudaGetSymbolAddress,                          /* getSymbolAddress */
        cudaGetSymbolSize,                             /* getSymbolSize */
        cudaStreamCreate,                              /* streamCreate */
        cudaStreamDestroy,                             /* streamDestroy */
        cudaStreamSynchronize,                         /* streamSynchronize */
        cudaStreamQuery,                               /* streamQuery */
        cudaStreamWaitEvent,                           /* streamWaitEvent */
        cudaEventCreate,                               /* eventCreate */
        cudaEventDestroy,                              /* eventDestroy */
        cudaEventRecord,                               /* eventRecord */
        cudaEventSynchronize,                          /* eventSynchronize */
        cudaEventQuery,                                /* eventQuery */
        cudaEventElapsedTime,                          /* eventElapsedTime */
        cudaModuleLoadData,                            /* moduleLoadData */
        cudaModuleLoadFromFile,                        /* moduleLoadFromFile */
        cudaModuleGetFunction,                         /* moduleGetFunction */
        cudaModuleUnload,                              /* moduleUnload */
        cudaLaunchKernel,                              /* launchKernel */
        cudaOccupancyMaxActiveBlocksPerMultiprocessor, /* occupancyMaxActiveBlocksPerMultiprocessor
                                                        */
        cudaOccupancyMaxPotentialBlockSize,            /* occupancyMaxPotentialBlockSize */
        cudaDriverGetVersion,                          /* driverGetVersion */
    };
    return &backend;
}

}  // namespace cool
