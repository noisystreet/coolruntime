// SPDX-FileCopyrightText: 2026 CoolRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * CoolRuntime 公开类型体系（C ABI）。
 *
 * 本头文件必须保持 C 兼容：只使用 C 类型，禁止 STL / 模板 / 引用。
 * 句柄统一为不透明指针，结构体定义只在内部头文件可见。
 */
#ifndef COOL_TYPES_H
#define COOL_TYPES_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 错误码：coolSuccess == 0 表示成功。 */
typedef enum coolError {
    coolSuccess = 0,
    coolErrorInvalidValue = 1,
    coolErrorOutOfMemory = 2,
    coolErrorMemoryAllocation = 3,
    coolErrorInvalidDevice = 10,
    coolErrorInvalidResourceHandle = 400,
    coolErrorNotReady = 600,
    coolErrorNotSupported = 801,
    coolErrorUnknown = 999
} coolError_t;

/* 拷贝方向，与 cudart 的 cudaMemcpyKind 一一对应。 */
typedef enum coolMemcpyKind {
    coolMemcpyHostToHost = 0,
    coolMemcpyHostToDevice = 1,
    coolMemcpyDeviceToHost = 2,
    coolMemcpyDeviceToDevice = 3,
    coolMemcpyDefault = 4
} coolMemcpyKind;

/* 流创建标志，与 cudart 的 cudaStreamFlags 对应。 */
typedef enum coolStreamFlags {
    coolStreamDefault = 0x00,
    coolStreamNonBlocking = 0x01
} coolStreamFlags;

/* 事件创建标志，与 cudart 的 cudaEventFlags 对应。 */
typedef enum coolEventFlags {
    coolEventDefault = 0x00,
    coolEventBlockingSync = 0x01,
    coolEventDisableTiming = 0x02,
    coolEventInterprocess = 0x04
} coolEventFlags;

/* 3D 维度（grid/block），默认各维为 1。 */
typedef struct dim3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
} dim3;

/* 简化版设备属性，对应 cudart 的 cudaDeviceProp。 */
typedef struct coolDeviceProp {
    char name[64];
    int major; /* cuda 后端：compute capability 主版本 */
    int minor; /* cuda 后端：compute capability 次版本 */
    int multiProcessorCount;
    size_t totalGlobalMem;
    int maxThreadsPerBlock;
    int maxThreadsPerMultiProcessor;
    int maxGridSize[3];
    int warpSize; /* cuda 后端：真实 warp 大小 */
} coolDeviceProp;

/* 不透明句柄：NULL 表示默认流。 */
typedef struct coolStream_st* coolStream_t;
typedef struct coolEvent_st* coolEvent_t;

/* 内核句柄（P3）：cuda 后端内部即 CUfunction。 */
typedef struct coolKernel coolKernel;
typedef coolKernel* coolKernel_t;

/* 模块句柄（P3）：cuda 后端内部即 CUmodule。 */
typedef struct coolModule_st* coolModule_t;

#ifdef __cplusplus
}
#endif

#endif /* COOL_TYPES_H */
