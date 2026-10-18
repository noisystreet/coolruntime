// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * COOLRuntime 公开 API（A 层）。
 *
 * 与 CUDA Runtime API（cudart）命名一一对应：cudaXxx -> coolXxx。
 * 本头文件保持 C 兼容，可被 C、C++ 及 CUDA 示例直接 include。
 *
 * 阶段 P0 覆盖：版本管理（6.7）、错误处理（6.1）、设备管理（6.2）。
 * 阶段 P1 覆盖：内存管理（6.4，含同步拷贝/填充与页锁定主机内存）。
 * 阶段 P2 覆盖：流管理、事件管理。
 * 阶段 P3 覆盖：模块与内核启动。
 * 阶段 P5 覆盖：异步拷贝（coolMemcpyAsync / coolMemsetAsync）与页锁定注册
 *               （coolHostRegister / coolHostUnregister）。
 */
#ifndef COOL_RUNTIME_API_H
#define COOL_RUNTIME_API_H

#include "cool_types.h"
#include "cool_version.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* 版本管理（对应 cudaRuntimeGetVersion / cudaDriverGetVersion）        */
/* ------------------------------------------------------------------ */

/* 返回运行时版本号（COOL_VERSION 组合值）。 */
coolError_t coolRuntimeGetVersion(int* runtimeVersion);

/* 返回驱动版本号（cuda 后端对应 cuDriverGetVersion）。 */
coolError_t coolDriverGetVersion(int* driverVersion);

/* ------------------------------------------------------------------ */
/* 错误处理（对应 cudaGetErrorString 等）                              */
/* ------------------------------------------------------------------ */

/* 返回错误码的可读字符串。 */
const char* coolGetErrorString(coolError_t error);

/* 返回错误码的枚举名。 */
const char* coolGetErrorName(coolError_t error);

/* 返回并清空线程局部 last error。 */
coolError_t coolGetLastError(void);

/* 返回但不清空线程局部 last error。 */
coolError_t coolPeekAtLastError(void);

/* ------------------------------------------------------------------ */
/* 设备管理（对应 cudaGetDeviceCount 等）                              */
/* ------------------------------------------------------------------ */

/* 返回逻辑设备数。 */
coolError_t coolGetDeviceCount(int* count);

/* 设置调用线程当前设备；非法设备返回 coolErrorInvalidDevice。 */
coolError_t coolSetDevice(int device);

/* 获取调用线程当前设备。 */
coolError_t coolGetDevice(int* device);

/* 填充简化版设备属性。 */
coolError_t coolGetDeviceProperties(coolDeviceProp* prop, int device);

/* 等待该设备所有流完成。 */
coolError_t coolDeviceSynchronize(void);

/* 释放运行时持有的该设备资源。 */
coolError_t coolDeviceReset(void);

/*
 * 取该设备支持的流优先级范围（对应 cudaDeviceGetStreamPriorityRange）。
 *
 * 语义容易搞反，务必注意：**数值越小，优先级越高**。
 *   - `leastPriority`    最低优先级，数值**最大**的一端（多数设备为 0）
 *   - `greatestPriority` 最高优先级，数值**最小**的一端（多数设备为 -5）
 * 有意义区间是 [greatestPriority, leastPriority]；区间外的优先级创建流时会被驱动钳制。
 */
coolError_t coolDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);

/* ------------------------------------------------------------------ */
/* 内存管理（对应 cudaMalloc / cudaMemcpy 等）                        */
/* ------------------------------------------------------------------ */

/* 分配 size 字节设备内存，结果写入 *ptr；size 为 0 返回 coolErrorInvalidValue。 */
coolError_t coolMalloc(void** ptr, size_t size);

/* 释放设备内存；ptr 为 NULL 时为空操作。未知或重复释放返回 coolErrorInvalidValue。 */
coolError_t coolFree(void* ptr);

/* 同步拷贝 count 字节，方向由 kind 指定。 */
coolError_t coolMemcpy(void* dst, const void* src, size_t count, coolMemcpyKind kind);

/* 将设备内存前 count 字节设为 value 的低 8 位。 */
coolError_t coolMemset(void* ptr, int value, size_t count);

/*
 * 异步拷贝（对应 cudaMemcpyAsync）。
 *
 * 把拷贝排入 stream 后立即返回，由该流保证顺序；stream 为 NULL 时用 legacy 默认流
 * （见 ADR 0005）。与 coolMemcpy 的区别仅在于是否等待拷贝完成。
 *
 * 主机侧缓冲区应为页锁定内存（coolMallocHost 或 coolHostRegister），否则驱动会把
 * 拷贝退化为同步执行，异步重叠失效 —— 这是 CUDA 自身的约束，coolrt 不做规避。
 * kind 为 coolMemcpyHostToHost 时驱动无对应 async 入口，内部按同步拷贝执行
 * （与 cudart 一致）。
 */
coolError_t coolMemcpyAsync(void* dst, const void* src, size_t count, coolMemcpyKind kind,
                            coolStream_t stream);

/* 异步填充（对应 cudaMemsetAsync）：把 ptr 前 count 字节设为 value 的低 8 位，排入 stream。 */
coolError_t coolMemsetAsync(void* ptr, int value, size_t count, coolStream_t stream);

/* 查询设备内存空闲与总量（字节）。 */
coolError_t coolMemGetInfo(size_t* freeBytes, size_t* totalBytes);

/* 分配页锁定（pinned）主机内存。 */
coolError_t coolMallocHost(void** ptr, size_t size);

/* 释放页锁定主机内存；ptr 为 NULL 时为空操作。 */
coolError_t coolFreeHost(void* ptr);

/*
 * 把一段已存在的主机内存注册为页锁定（对应 cudaHostRegister）。
 *
 * 与 coolMallocHost 的区别：后者分配并锁定，前者锁定调用方已有的缓冲区。
 * flags 取值见 coolHostRegisterFlags。重复注册同一地址返回
 * coolErrorHostMemoryAlreadyRegistered；注册后须用 coolHostUnregister 解除。
 */
coolError_t coolHostRegister(void* ptr, size_t size, unsigned int flags);

/*
 * 解除页锁定（对应 cudaHostUnregister）。
 * 未注册的地址返回 coolErrorHostMemoryNotRegistered。
 */
coolError_t coolHostUnregister(void* ptr);

/*
 * 把主机数据拷入设备符号（对应 cudaMemcpyToSymbol）。
 *
 * symbol 为 coolSymbolDesc 的地址（见 cool_types.h）；src 为源数据，
 * count 为字节数，offset 为符号内偏移，kind 为拷贝方向（默认 H2D）。
 * 越界（offset + count > 符号大小）返回 coolErrorInvalidValue。
 */
coolError_t coolMemcpyToSymbol(const void* symbol, const void* src, size_t count, size_t offset,
                               coolMemcpyKind kind);

/*
 * 把设备符号拷出到主机或设备内存（对应 cudaMemcpyFromSymbol）。
 *
 * dst 为目标指针；symbol 为 coolSymbolDesc 的地址；count 为字节数，offset 为符号内偏移。
 * kind 受符号在设备侧这一事实约束：只接受 coolMemcpyDeviceToHost（默认）与
 * coolMemcpyDeviceToDevice，其余方向（H2H / H2D / Default）返回 coolErrorInvalidValue。
 * 越界（offset + count > 符号大小）返回 coolErrorInvalidValue。
 *
 * 与 ToSymbol 对称：两者合起来覆盖「主机 ↔ 设备符号」的双向搬运。
 */
coolError_t coolMemcpyFromSymbol(void* dst, const void* symbol, size_t count, size_t offset,
                                 coolMemcpyKind kind);

/* 取设备符号地址（对应 cudaGetSymbolAddress）；结果写入 *devPtr。 */
coolError_t coolGetSymbolAddress(void** devPtr, const void* symbol);

/* 取设备符号字节数（对应 cudaGetSymbolSize）；结果写入 *size。 */
coolError_t coolGetSymbolSize(size_t* size, const void* symbol);

/* ------------------------------------------------------------------ */
/* 流管理（对应 cudaStreamCreate / cudaStreamSynchronize 等）          */
/* ------------------------------------------------------------------ */

/* 创建流（默认标志）。结果写入 *stream。 */
coolError_t coolStreamCreate(coolStream_t* stream);

/* 以 flags 创建流；flags 取值见 coolStreamFlags。 */
coolError_t coolStreamCreateWithFlags(coolStream_t* stream, unsigned int flags);

/* 以 flags 与优先级创建流。 */
coolError_t coolStreamCreateWithPriority(coolStream_t* stream, unsigned int flags, int priority);

/* 销毁流；NULL 表示默认流，不可销毁，返回 coolErrorInvalidValue。 */
coolError_t coolStreamDestroy(coolStream_t stream);

/* 等待流中所有任务完成。NULL 表示默认流。 */
coolError_t coolStreamSynchronize(coolStream_t stream);

/* 查询流是否完成；未完成返回 coolErrorNotReady。NULL 表示默认流。 */
coolError_t coolStreamQuery(coolStream_t stream);

/* 令 stream 等待 event。 */
coolError_t coolStreamWaitEvent(coolStream_t stream, coolEvent_t event);

/* 取流的优先级（对应 cudaStreamGetPriority）；结果写入 *priority。NULL 表示默认流。 */
coolError_t coolStreamGetPriority(int* priority, coolStream_t stream);

/* 取流的创建标志（对应 cudaStreamGetFlags）；结果写入 *flags。NULL 表示默认流。 */
coolError_t coolStreamGetFlags(unsigned int* flags, coolStream_t stream);

/* ------------------------------------------------------------------ */
/* 事件管理（对应 cudaEventCreate / cudaEventRecord 等）              */
/* ------------------------------------------------------------------ */

/* 创建事件（默认标志）。结果写入 *event。 */
coolError_t coolEventCreate(coolEvent_t* event);

/* 以 flags 创建事件；flags 取值见 coolEventFlags。 */
coolError_t coolEventCreateWithFlags(coolEvent_t* event, unsigned int flags);

/* 销毁事件。 */
coolError_t coolEventDestroy(coolEvent_t event);

/* 在 stream 上记录 event；NULL 表示默认流。 */
coolError_t coolEventRecord(coolEvent_t event, coolStream_t stream);

/* 等待 event 完成。 */
coolError_t coolEventSynchronize(coolEvent_t event);

/* 查询 event 是否完成；未完成返回 coolErrorNotReady。 */
coolError_t coolEventQuery(coolEvent_t event);

/* 计算两个已记录事件之间的毫秒差。 */
coolError_t coolEventElapsedTime(float* ms, coolEvent_t start, coolEvent_t end);

/* ------------------------------------------------------------------ */
/* 模块与内核（对应 cuModuleLoadData / cuModuleGetFunction 等 Driver API） */
/* ------------------------------------------------------------------ */

/* 从内存映像载入模块（PTX 文本或 cubin），驱动负责 JIT/加载。结果写入 *module。 */
coolError_t coolModuleLoadData(coolModule_t* module, const void* image);

/* 从文件载入模块（对应 cuModuleLoad）。结果写入 *module。 */
coolError_t coolModuleLoadFromFile(coolModule_t* module, const char* path);

/* 按符号名从模块取内核句柄（对应 cuModuleGetFunction）。结果写入 *kernel。 */
coolError_t coolModuleGetFunction(coolKernel_t* kernel, coolModule_t module, const char* name);

/* 卸载模块（对应 cuModuleUnload）。 */
coolError_t coolModuleUnload(coolModule_t module);

/*
 * 启动内核（对应 cuLaunchKernel 的启动跳板）。
 * args 直接透传（与 cuLaunchKernel 的 kernelParams 语义一致，不做参数类型推导）；
 * stream 为 NULL 时使用默认流。
 */
coolError_t coolLaunchKernel(coolKernel_t kernel, dim3 grid, dim3 block, void** args,
                             size_t sharedMem, coolStream_t stream);

/* 查询每 SM 最大活跃块数（对应 cuOccupancyMaxActiveBlocksPerMultiprocessor）。 */
coolError_t coolOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, coolKernel_t kernel,
                                                          int blockSize, size_t dynamicSMemSize);

/* 查询推荐 block 大小与对应最小 grid 大小（对应 cuOccupancyMaxPotentialBlockSize）。 */
coolError_t coolOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize,
                                               coolKernel_t kernel, size_t dynamicSMemSize,
                                               int blockSizeLimit);

/*
 * 取内核函数属性（对应 cudaFuncGetAttributes），结果写入 *attr。
 *
 * Driver 侧只有逐属性查询的 cuFuncGetAttribute，本实现在后端循环填满
 * coolFuncAttributes 的 5 个字段（见 cool_types.h）。任一属性查询失败即整体失败。
 */
coolError_t coolFuncGetAttributes(coolFuncAttributes* attr, coolKernel_t kernel);

/*
 * 可选：把 cudart 名称映射到 cool 名称，便于把小型 CUDA 示例
 * 源码直接改跑到 COOLRuntime 上（教学对照用途）。
 */
#ifdef COOL_ENABLE_CUDA_ALIAS
#define cudaRuntimeGetVersion coolRuntimeGetVersion
#define cudaDriverGetVersion coolDriverGetVersion
#define cudaGetErrorString coolGetErrorString
#define cudaGetErrorName coolGetErrorName
#define cudaGetLastError coolGetLastError
#define cudaPeekAtLastError coolPeekAtLastError
#define cudaGetDeviceCount coolGetDeviceCount
#define cudaSetDevice coolSetDevice
#define cudaGetDevice coolGetDevice
#define cudaGetDeviceProperties coolGetDeviceProperties
#define cudaDeviceSynchronize coolDeviceSynchronize
#define cudaDeviceReset coolDeviceReset
#define cudaMalloc coolMalloc
#define cudaFree coolFree
#define cudaMemcpy coolMemcpy
#define cudaMemset coolMemset
#define cudaMemGetInfo coolMemGetInfo
#define cudaMallocHost coolMallocHost
#define cudaFreeHost coolFreeHost
#define cudaMemcpyToSymbol coolMemcpyToSymbol
#define cudaMemcpyFromSymbol coolMemcpyFromSymbol
#define cudaGetSymbolAddress coolGetSymbolAddress
#define cudaGetSymbolSize coolGetSymbolSize
#define cudaStreamCreate coolStreamCreate
#define cudaStreamCreateWithFlags coolStreamCreateWithFlags
#define cudaStreamCreateWithPriority coolStreamCreateWithPriority
#define cudaStreamDestroy coolStreamDestroy
#define cudaStreamSynchronize coolStreamSynchronize
#define cudaStreamQuery coolStreamQuery
#define cudaStreamWaitEvent coolStreamWaitEvent
#define cudaEventCreate coolEventCreate
#define cudaEventCreateWithFlags coolEventCreateWithFlags
#define cudaEventDestroy coolEventDestroy
#define cudaEventRecord coolEventRecord
#define cudaEventSynchronize coolEventSynchronize
#define cudaEventQuery coolEventQuery
#define cudaEventElapsedTime coolEventElapsedTime
#endif /* COOL_ENABLE_CUDA_ALIAS */

#ifdef __cplusplus
}
#endif

#endif /* COOL_RUNTIME_API_H */
