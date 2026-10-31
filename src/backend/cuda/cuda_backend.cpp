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

/* coolDeviceAttribute → CU_DEVICE_ATTRIBUTE_*。未列入即 NotSupported。 */
struct DeviceAttributeMapping {
    coolDeviceAttribute attr;
    CUdevice_attribute cu;
};

constexpr DeviceAttributeMapping kDeviceAttributeMappings[] = {
    {coolDevAttrMaxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK},
    {coolDevAttrMaxBlockDimX, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X},
    {coolDevAttrMaxBlockDimY, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y},
    {coolDevAttrMaxBlockDimZ, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z},
    {coolDevAttrMaxGridDimX, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X},
    {coolDevAttrMaxGridDimY, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y},
    {coolDevAttrMaxGridDimZ, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z},
    {coolDevAttrMaxSharedMemoryPerBlock, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK},
    {coolDevAttrMaxSharedMemoryPerMultiprocessor,
     CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR},
    {coolDevAttrMaxRegistersPerBlock, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK},
    {coolDevAttrWarpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE},
    {coolDevAttrMultiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT},
    {coolDevAttrComputeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR},
    {coolDevAttrComputeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR},
    {coolDevAttrMaxThreadsPerMultiProcessor, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR},
    {coolDevAttrClockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE},
    {coolDevAttrMemoryClockRate, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE},
    {coolDevAttrGlobalMemoryBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH},
    {coolDevAttrL2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE},
    {coolDevAttrConcurrentKernels, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS},
    {coolDevAttrAsyncEngineCount, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT},
    {coolDevAttrUnifiedAddressing, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING},
};

const CUdevice_attribute* findCuAttribute(coolDeviceAttribute attr) {
    for (const DeviceAttributeMapping& mapping : kDeviceAttributeMappings) {
        if (mapping.attr == attr) {
            return &mapping.cu;
        }
    }
    return nullptr;
}

/*
 * 填 coolDeviceProp 用的「尽力而为」查询：与既有字段一致，属性查询失败不改返回码
 * （现有实现对 major/minor 等同样忽略 cuDeviceGetAttribute 的返回值）。
 */
int queryAttributeValue(CUdevice device, coolDeviceAttribute attr) {
    int value = 0;
    const CUdevice_attribute* cuAttr = findCuAttribute(attr);
    if (cuAttr != nullptr) {
        cuDeviceGetAttribute(&value, *cuAttr, device);
    }
    return value;
}

coolError_t cudaDeviceGetAttribute(int* value, coolDeviceAttribute attr, int dev) {
    if (value == nullptr) {
        return coolErrorInvalidValue;
    }
    const CUdevice_attribute* cuAttr = findCuAttribute(attr);
    if (cuAttr == nullptr) {
        return coolErrorNotSupported;
    }

    CUdevice device = 0;
    const CUresult res = cuDeviceGet(&device, dev);
    if (res != CUDA_SUCCESS) {
        return resultToError(res);
    }
    return resultToError(cuDeviceGetAttribute(value, *cuAttr, device));
}

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

    /* P6 R1 追加字段：走同一张映射表，保证与 coolDeviceGetAttribute 同源同值。 */
    prop->maxThreadsDim[0] = queryAttributeValue(device, coolDevAttrMaxBlockDimX);
    prop->maxThreadsDim[1] = queryAttributeValue(device, coolDevAttrMaxBlockDimY);
    prop->maxThreadsDim[2] = queryAttributeValue(device, coolDevAttrMaxBlockDimZ);
    prop->sharedMemPerBlock = queryAttributeValue(device, coolDevAttrMaxSharedMemoryPerBlock);
    prop->sharedMemPerMultiprocessor =
        queryAttributeValue(device, coolDevAttrMaxSharedMemoryPerMultiprocessor);
    prop->regsPerBlock = queryAttributeValue(device, coolDevAttrMaxRegistersPerBlock);
    prop->clockRate = queryAttributeValue(device, coolDevAttrClockRate);
    prop->memoryClockRate = queryAttributeValue(device, coolDevAttrMemoryClockRate);
    prop->globalMemoryBusWidth = queryAttributeValue(device, coolDevAttrGlobalMemoryBusWidth);
    prop->l2CacheSize = queryAttributeValue(device, coolDevAttrL2CacheSize);
    prop->unifiedAddressing = queryAttributeValue(device, coolDevAttrUnifiedAddressing);
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

coolError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
    if (leastPriority == nullptr || greatestPriority == nullptr) {
        return coolErrorInvalidValue;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }
    return resultToError(cuCtxGetStreamPriorityRange(leastPriority, greatestPriority));
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

/*
 * 按方向分派异步拷贝。
 *
 * Driver API 没有 HtoHAsync —— 主机→主机的「异步」拷贝在 cudart 里也是同步执行的，
 * 故这里退回 cuMemcpy。coolMemcpyDefault 用 cuMemcpyAsync，由驱动依据统一寻址下的
 * 指针属性推断方向。
 */
CUresult asyncMemcpyByKind(void* dst, const void* src, size_t count, coolMemcpyKind kind,
                           CUstream stream) {
    switch (kind) {
        case coolMemcpyHostToDevice:
            return cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(dst), src, count, stream);
        case coolMemcpyDeviceToHost:
            return cuMemcpyDtoHAsync(dst, reinterpret_cast<CUdeviceptr>(src), count, stream);
        case coolMemcpyDeviceToDevice:
            return cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(dst),
                                     reinterpret_cast<CUdeviceptr>(src), count, stream);
        case coolMemcpyDefault:
            return cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(dst),
                                 reinterpret_cast<CUdeviceptr>(src), count, stream);
        case coolMemcpyHostToHost:
            return cuMemcpy(reinterpret_cast<CUdeviceptr>(dst), reinterpret_cast<CUdeviceptr>(src),
                            count);
        default:
            return CUDA_ERROR_INVALID_VALUE;
    }
}

coolError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, coolMemcpyKind kind,
                            coolStream_t stream) {
    if (dst == nullptr || src == nullptr) {
        return coolErrorInvalidValue;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }
    if (stream != nullptr && !handles().hasStream(handleAddr(stream))) {
        return coolErrorInvalidValue;
    }
    /* count 为 0 是合法空操作，驱动同样接受。 */
    return resultToError(
        asyncMemcpyByKind(dst, src, count, kind, reinterpret_cast<CUstream>(stream)));
}

coolError_t cudaMemsetAsync(void* ptr, int value, size_t count, coolStream_t stream) {
    if (ptr == nullptr) {
        return coolErrorInvalidValue;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }
    if (stream != nullptr && !handles().hasStream(handleAddr(stream))) {
        return coolErrorInvalidValue;
    }
    return resultToError(cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(ptr),
                                         static_cast<unsigned char>(value & 0xFF), count,
                                         reinterpret_cast<CUstream>(stream)));
}

coolError_t cudaMemHostRegister(void* ptr, size_t size, unsigned flags) {
    if (ptr == nullptr || size == 0) {
        return coolErrorInvalidValue;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }
    /*
     * 不自行判重：驱动对已注册地址返回 CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED，
     * 经 resultToError 映射为 coolErrorHostMemoryAlreadyRegistered。以驱动为准可避开
     * 「先查表再注册」的 TOCTOU 窗口。
     */
    const CUresult res = cuMemHostRegister(ptr, size, flags);
    if (res != CUDA_SUCCESS) {
        return resultToError(res);
    }
    allocations().addRegisteredHost(handleAddr(ptr), size);
    return coolSuccess;
}

coolError_t cudaMemHostUnregister(void* ptr) {
    if (ptr == nullptr) {
        return coolErrorInvalidValue;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }
    const CUresult res = cuMemHostUnregister(ptr);
    if (res != CUDA_SUCCESS) {
        return resultToError(res);
    }
    allocations().removeRegisteredHost(handleAddr(ptr));
    return coolSuccess;
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

coolError_t cudaStreamGetPriority(int* priority, coolStream_t stream) {
    if (priority == nullptr) {
        return coolErrorInvalidValue;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }
    if (stream != nullptr && !handles().hasStream(handleAddr(stream))) {
        return coolErrorInvalidValue;
    }
    return resultToError(cuStreamGetPriority(reinterpret_cast<CUstream>(stream), priority));
}

coolError_t cudaStreamGetFlags(unsigned int* flags, coolStream_t stream) {
    if (flags == nullptr) {
        return coolErrorInvalidValue;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }
    if (stream != nullptr && !handles().hasStream(handleAddr(stream))) {
        return coolErrorInvalidValue;
    }
    return resultToError(cuStreamGetFlags(reinterpret_cast<CUstream>(stream), flags));
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

coolError_t cudaFuncGetAttributes(coolFuncAttributes* attr, const coolKernel* kernel) {
    if (attr == nullptr || kernel == nullptr) {
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
    /* Driver 只有逐属性查询（没有 cudart 那样的批量 cudaFuncGetAttributes），
     * 故按 coolFuncAttributes 的字段逐项填；任一失败即整体失败。 */
    struct Query {
        CUfunction_attribute attrib;
        int* out;
    };
    const Query queries[] = {
        {CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, &attr->maxThreadsPerBlock},
        {CU_FUNC_ATTRIBUTE_NUM_REGS, &attr->numRegs},
        {CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, &attr->sharedSizeBytes},
        {CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, &attr->localSizeBytes},
        {CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, &attr->constSizeBytes},
    };
    for (const Query& q : queries) {
        const CUresult res = cuFuncGetAttribute(q.out, q.attrib, func);
        if (res != CUDA_SUCCESS) {
            return resultToError(res);
        }
    }
    return coolSuccess;
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
        cudaDeviceGetStreamPriorityRange,              /* deviceGetStreamPriorityRange */
        cudaDeviceGetAttribute,                        /* deviceGetAttribute */
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
        cudaMemcpyAsync,                               /* memcpyAsync */
        cudaMemsetAsync,                               /* memsetAsync */
        cudaMemHostRegister,                           /* memHostRegister */
        cudaMemHostUnregister,                         /* memHostUnregister */
        cudaStreamCreate,                              /* streamCreate */
        cudaStreamDestroy,                             /* streamDestroy */
        cudaStreamSynchronize,                         /* streamSynchronize */
        cudaStreamQuery,                               /* streamQuery */
        cudaStreamWaitEvent,                           /* streamWaitEvent */
        cudaStreamGetPriority,                         /* streamGetPriority */
        cudaStreamGetFlags,                            /* streamGetFlags */
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
        cudaFuncGetAttributes,                         /* funcGetAttributes */
        cudaDriverGetVersion,                          /* driverGetVersion */
    };
    return &backend;
}

}  // namespace cool
