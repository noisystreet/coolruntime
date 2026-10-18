// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * 后端抽象层（BAL）接口契约（内部头，以 C++20 编译）。
 *
 * 后端以「函数指针结构体」表达能力：返回 NULL 表示未实现该能力，
 * R 层在转发前检查，未实现则返回 coolErrorNotSupported。
 *
 * BAL 不包含上下文接口：Driver API 的显式上下文语义由 cuda 后端内部完成，
 * 对 R 层透明。
 */
#ifndef COOL_BACKEND_H
#define COOL_BACKEND_H

#include <stddef.h>

#include "cool/cool_types.h"

#define COOL_BACKEND_API_VERSION 1

typedef struct coolBackend {
    const char* name; /* 后端名，当前为 "cuda" */
    int apiVersion;   /* BAL 版本，用于兼容校验 */

    /* 生命周期 */
    coolError_t (*init)(void);
    coolError_t (*finalize)(void);

    /* 设备 */
    coolError_t (*getDeviceCount)(int* count);
    coolError_t (*getDeviceProperties)(int dev, coolDeviceProp* prop);
    coolError_t (*deviceSynchronize)(void);
    coolError_t (*deviceReset)(void);
    /* P5.5：流优先级范围（context 级，对应 cuCtxGetStreamPriorityRange）。 */
    coolError_t (*deviceGetStreamPriorityRange)(int* leastPriority, int* greatestPriority);

    /* 内存 */
    coolError_t (*malloc)(void** p, size_t size);
    coolError_t (*free)(void* p);
    coolError_t (*mallocHost)(void** p, size_t size, unsigned flags);
    coolError_t (*freeHost)(void* p);
    coolError_t (*memcpy)(void* dst, const void* src, size_t count, coolMemcpyKind kind);
    coolError_t (*memset)(void* p, int value, size_t count);
    coolError_t (*memGetInfo)(size_t* freeBytes, size_t* totalBytes);
    /* M3.2：设备符号（__constant__）拷贝与地址/大小查询。 */
    coolError_t (*memcpyToSymbol)(const void* symbol, const void* src, size_t count, size_t offset,
                                  coolMemcpyKind kind);
    /* 设备符号 → 主机（对应 cudaMemcpyFromSymbol）；与 memcpyToSymbol 对称。 */
    coolError_t (*memcpyFromSymbol)(void* dst, const void* symbol, size_t count, size_t offset,
                                    coolMemcpyKind kind);
    coolError_t (*getSymbolAddress)(void** devPtr, const void* symbol);
    coolError_t (*getSymbolSize)(size_t* size, const void* symbol);
    /* P5.4：异步拷贝与填充（排入 stream，不等待完成）。 */
    coolError_t (*memcpyAsync)(void* dst, const void* src, size_t count, coolMemcpyKind kind,
                               coolStream_t s);
    coolError_t (*memsetAsync)(void* p, int value, size_t count, coolStream_t s);
    /* P5.4：把既有主机内存注册为页锁定。 */
    coolError_t (*memHostRegister)(void* p, size_t size, unsigned flags);
    coolError_t (*memHostUnregister)(void* p);

    /* 流 */
    coolError_t (*streamCreate)(coolStream_t* s, unsigned flags, int priority);
    coolError_t (*streamDestroy)(coolStream_t s);
    coolError_t (*streamSynchronize)(coolStream_t s);
    coolError_t (*streamQuery)(coolStream_t s);
    coolError_t (*streamWaitEvent)(coolStream_t s, coolEvent_t e);
    /* P5.5：流属性查询。 */
    coolError_t (*streamGetPriority)(int* priority, coolStream_t s);
    coolError_t (*streamGetFlags)(unsigned int* flags, coolStream_t s);

    /* 事件 */
    coolError_t (*eventCreate)(coolEvent_t* e, unsigned flags);
    coolError_t (*eventDestroy)(coolEvent_t e);
    coolError_t (*eventRecord)(coolEvent_t e, coolStream_t s);
    coolError_t (*eventSynchronize)(coolEvent_t e);
    coolError_t (*eventQuery)(coolEvent_t e);
    coolError_t (*eventElapsedTime)(float* ms, coolEvent_t start, coolEvent_t end);

    /* 模块与内核 */
    coolError_t (*moduleLoadData)(coolModule_t* m, const void* image);
    coolError_t (*moduleLoadFromFile)(coolModule_t* m, const char* path);
    coolError_t (*moduleGetFunction)(coolKernel_t* k, coolModule_t m, const char* name);
    coolError_t (*moduleUnload)(coolModule_t m);
    coolError_t (*launchKernel)(const coolKernel* k, dim3 grid, dim3 block, void** args,
                                size_t sharedMem, coolStream_t s);
    coolError_t (*occupancyMaxActiveBlocksPerMultiprocessor)(int* numBlocks, const coolKernel* k,
                                                             int blockSize, size_t dynamicSMemSize);
    coolError_t (*occupancyMaxPotentialBlockSize)(int* minGridSize, int* blockSize,
                                                  const coolKernel* k, size_t dynamicSMemSize,
                                                  int blockSizeLimit);
    /* P5.5：内核函数属性（Driver 侧逐属性查询，由后端填满结构体）。 */
    coolError_t (*funcGetAttributes)(coolFuncAttributes* attr, const coolKernel* k);

    /*
     * 扩展成员：BAL 契约未列出驱动版本查询，但版本管理要求
     * cuda 后端返回 cuDriverGetVersion，故在此补充。
     */
    coolError_t (*driverGetVersion)(int* version);
} coolBackend;

#endif /* COOL_BACKEND_H */
