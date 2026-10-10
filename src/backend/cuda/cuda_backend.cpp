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

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

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

/*
 * 后端侧内存元数据注册表。
 *
 * 记录本后端分配的设备/主机指针及其字节数，用于校验 free（识别未知指针与
 * 重复释放），并在 finalize / deviceReset 时统一清理。宿主并发原语
 * （std::mutex）按 AGENTS.md 硬约束只允许出现在后端，故注册表放在后端而非 R 层。
 */
class AllocationRegistry {
   public:
    void addDevice(std::uintptr_t ptr, size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        device_[ptr] = size;
    }

    bool hasDevice(std::uintptr_t ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        return device_.count(ptr) != 0;
    }

    void removeDevice(std::uintptr_t ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        device_.erase(ptr);
    }

    void addHost(std::uintptr_t ptr, size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        host_[ptr] = size;
    }

    bool hasHost(std::uintptr_t ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        return host_.count(ptr) != 0;
    }

    void removeHost(std::uintptr_t ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        host_.erase(ptr);
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        device_.clear();
        host_.clear();
    }

   private:
    std::mutex mutex_;
    std::unordered_map<std::uintptr_t, size_t> device_;
    std::unordered_map<std::uintptr_t, size_t> host_;
};

AllocationRegistry& registry() {
    static AllocationRegistry instance;
    return instance;
}

/*
 * 后端侧流/事件句柄注册表。
 *
 * 记录本后端创建的流与事件句柄，用于校验销毁（识别未知句柄与重复销毁），
 * 并在 finalize / deviceReset 时统一清理。宿主并发原语只允许出现在后端，
 * 故注册表放在后端而非 R 层（见 ADR 0003）。NULL 表示默认流，不登记。
 */
class HandleRegistry {
   public:
    void addStream(std::uintptr_t handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        streams_.insert(handle);
    }

    bool hasStream(std::uintptr_t handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        return streams_.count(handle) != 0;
    }

    void removeStream(std::uintptr_t handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        streams_.erase(handle);
    }

    void addEvent(std::uintptr_t handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.insert(handle);
    }

    bool hasEvent(std::uintptr_t handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        return events_.count(handle) != 0;
    }

    void removeEvent(std::uintptr_t handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.erase(handle);
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        streams_.clear();
        events_.clear();
    }

   private:
    std::mutex mutex_;
    std::unordered_set<std::uintptr_t> streams_;
    std::unordered_set<std::uintptr_t> events_;
};

HandleRegistry& handleRegistry() {
    static HandleRegistry instance;
    return instance;
}

/* 把 CUstream / CUevent 与其不透明句柄表示互相转换。 */
std::uintptr_t handleAddr(const void* handle) {
    return reinterpret_cast<std::uintptr_t>(handle);
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
    registry().clear();
    handleRegistry().clear();
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
        return cuResultToCool(res);
    }
    registry().addDevice(static_cast<std::uintptr_t>(devicePtr), size);
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
    if (!registry().hasDevice(addr)) {
        return coolErrorInvalidValue;
    }
    const CUresult res = cuMemFree(static_cast<CUdeviceptr>(addr));
    if (res != CUDA_SUCCESS) {
        return cuResultToCool(res);
    }
    registry().removeDevice(addr);
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
            return cuResultToCool(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(dst), src, count));
        case coolMemcpyDeviceToHost:
            return cuResultToCool(cuMemcpyDtoH(dst, reinterpret_cast<CUdeviceptr>(src), count));
        case coolMemcpyDeviceToDevice:
            return cuResultToCool(cuMemcpyDtoD(reinterpret_cast<CUdeviceptr>(dst),
                                               reinterpret_cast<CUdeviceptr>(src), count));
        case coolMemcpyDefault:
            return cuResultToCool(cuMemcpy(reinterpret_cast<CUdeviceptr>(dst),
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
    return cuResultToCool(
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
    return cuResultToCool(cuMemGetInfo(freeBytes, totalBytes));
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
        return cuResultToCool(res);
    }
    registry().addHost(reinterpret_cast<std::uintptr_t>(hostPtr), size);
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
    if (!registry().hasHost(addr)) {
        return coolErrorInvalidValue;
    }
    const CUresult res = cuMemFreeHost(ptr);
    if (res != CUDA_SUCCESS) {
        return cuResultToCool(res);
    }
    registry().removeHost(addr);
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
        return cuResultToCool(res);
    }
    handleRegistry().addStream(handleAddr(cuStream));
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
    if (!handleRegistry().hasStream(addr)) {
        return coolErrorInvalidValue;
    }
    const CUresult res = cuStreamDestroy(reinterpret_cast<CUstream>(stream));
    if (res != CUDA_SUCCESS) {
        return cuResultToCool(res);
    }
    handleRegistry().removeStream(addr);
    return coolSuccess;
}

coolError_t cudaStreamSynchronize(coolStream_t stream) {
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }
    if (stream != nullptr && !handleRegistry().hasStream(handleAddr(stream))) {
        return coolErrorInvalidValue;
    }
    return cuResultToCool(cuStreamSynchronize(reinterpret_cast<CUstream>(stream)));
}

coolError_t cudaStreamQuery(coolStream_t stream) {
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }
    if (stream != nullptr && !handleRegistry().hasStream(handleAddr(stream))) {
        return coolErrorInvalidValue;
    }
    return cuResultToCool(cuStreamQuery(reinterpret_cast<CUstream>(stream)));
}

coolError_t cudaStreamWaitEvent(coolStream_t stream, coolEvent_t event) {
    if (event == nullptr) {
        return coolErrorInvalidValue;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }
    if (stream != nullptr && !handleRegistry().hasStream(handleAddr(stream))) {
        return coolErrorInvalidValue;
    }
    if (!handleRegistry().hasEvent(handleAddr(event))) {
        return coolErrorInvalidValue;
    }
    return cuResultToCool(cuStreamWaitEvent(reinterpret_cast<CUstream>(stream),
                                            reinterpret_cast<CUevent>(event), 0));
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
        return cuResultToCool(res);
    }
    handleRegistry().addEvent(handleAddr(cuEvent));
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
    if (!handleRegistry().hasEvent(addr)) {
        return coolErrorInvalidValue;
    }
    const CUresult res = cuEventDestroy(reinterpret_cast<CUevent>(event));
    if (res != CUDA_SUCCESS) {
        return cuResultToCool(res);
    }
    handleRegistry().removeEvent(addr);
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
    if (!handleRegistry().hasEvent(handleAddr(event))) {
        return coolErrorInvalidValue;
    }
    if (stream != nullptr && !handleRegistry().hasStream(handleAddr(stream))) {
        return coolErrorInvalidValue;
    }
    return cuResultToCool(cuEventRecord(reinterpret_cast<CUevent>(event),
                                        reinterpret_cast<CUstream>(stream)));
}

coolError_t cudaEventSynchronize(coolEvent_t event) {
    if (event == nullptr) {
        return coolErrorInvalidValue;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }
    if (!handleRegistry().hasEvent(handleAddr(event))) {
        return coolErrorInvalidValue;
    }
    return cuResultToCool(cuEventSynchronize(reinterpret_cast<CUevent>(event)));
}

coolError_t cudaEventQuery(coolEvent_t event) {
    if (event == nullptr) {
        return coolErrorInvalidValue;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }
    if (!handleRegistry().hasEvent(handleAddr(event))) {
        return coolErrorInvalidValue;
    }
    return cuResultToCool(cuEventQuery(reinterpret_cast<CUevent>(event)));
}

coolError_t cudaEventElapsedTime(float* ms, coolEvent_t start, coolEvent_t end) {
    if (ms == nullptr || start == nullptr || end == nullptr) {
        return coolErrorInvalidValue;
    }
    const coolError_t err = ensureContext();
    if (err != coolSuccess) {
        return err;
    }
    if (!handleRegistry().hasEvent(handleAddr(start)) ||
        !handleRegistry().hasEvent(handleAddr(end))) {
        return coolErrorInvalidValue;
    }
    return cuResultToCool(cuEventElapsedTime(ms, reinterpret_cast<CUevent>(start),
                                             reinterpret_cast<CUevent>(end)));
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
        cudaMalloc,               /* malloc */
        cudaFree,                 /* free */
        cudaMallocHost,           /* mallocHost */
        cudaFreeHost,             /* freeHost */
        cudaMemcpy,               /* memcpy */
        cudaMemset,               /* memset */
        cudaMemGetInfo,           /* memGetInfo */
        cudaStreamCreate,         /* streamCreate */
        cudaStreamDestroy,        /* streamDestroy */
        cudaStreamSynchronize,    /* streamSynchronize */
        cudaStreamQuery,          /* streamQuery */
        cudaStreamWaitEvent,      /* streamWaitEvent */
        cudaEventCreate,          /* eventCreate */
        cudaEventDestroy,         /* eventDestroy */
        cudaEventRecord,          /* eventRecord */
        cudaEventSynchronize,     /* eventSynchronize */
        cudaEventQuery,           /* eventQuery */
        cudaEventElapsedTime,     /* eventElapsedTime */
        nullptr,                  /* launchKernel */
        cudaDriverGetVersion,     /* driverGetVersion */
    };
    return &backend;
}

}  // namespace cool
