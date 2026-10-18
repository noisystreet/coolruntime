// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * COOLRuntime 公开类型体系（C ABI）。
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
    coolErrorHostMemoryAlreadyRegistered = 711,
    coolErrorHostMemoryNotRegistered = 712,
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

/* 页锁定注册标志，与 cudart 的 cudaHostRegisterFlags 对应。 */
typedef enum coolHostRegisterFlags {
    coolHostRegisterDefault = 0x00,
    coolHostRegisterPortable = 0x01,
    coolHostRegisterMapped = 0x02
} coolHostRegisterFlags;

/* 事件创建标志，与 cudart 的 cudaEventFlags 对应。 */
typedef enum coolEventFlags {
    coolEventDefault = 0x00,
    coolEventBlockingSync = 0x01,
    coolEventDisableTiming = 0x02,
    coolEventInterprocess = 0x04
} coolEventFlags;

/* 3D 维度（grid/block），默认各维为 1。
 *
 * **归属**：`dim3` 由本头唯一提供（coolrt 公开契约）。hacc 的 `hacc_runtime.h` 只
 * 消费、不定义——两侧另立同义类型会破坏 `coolLaunchKernel` 的 C ABI 一致性。
 *
 * **C++ 构造函数**：对齐 CUDA 的 `dim3(x, y, z)` 用法（如 `dim3 grid((n+255)/256)`）。
 * 代价是 dim3 在 C++ 下**不再是聚合类型**，`dim3{x,y,z}` 改走构造函数（语义等价）；
 * C 下仍是纯聚合 struct，字段布局与 ABI 不变（12 字节、对齐 4）。
 */
typedef struct dim3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
#ifdef __cplusplus
    /* 默认参数对齐 CUDA：`dim3 d;` 即 (1, 1, 1)。 */
    dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
#endif
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

/* 不透明句柄：NULL 表示默认流（legacy 语义：与所有 blocking 流隐式同步；
 * 不支持 per-thread 默认流，见 docs/adr/0005-default-stream-semantics.md）。 */
typedef struct coolStream_st* coolStream_t;
typedef struct coolEvent_st* coolEvent_t;

/* 内核句柄（P3）：cuda 后端内部即 CUfunction。 */
typedef struct coolKernel coolKernel;
typedef coolKernel* coolKernel_t;

/* 模块句柄（P3）：cuda 后端内部即 CUmodule。 */
typedef struct coolModule_st* coolModule_t;

/*
 * 设备符号描述符（M3.2）：用于 coolMemcpyToSymbol / coolGetSymbolAddress /
 * coolGetSymbolSize 定位设备侧符号（如 __constant__ 变量）。
 *
 * 与 CUDA 的差异：cudart 的 cudaMemcpyToSymbol 接收「主机侧符号地址」，
 * 由 nvcc 编译期魔法解析；COOLRuntime 无此机制，故由 hacc 生成主机桩时
 * 构造本描述符，其地址即作为 symbol 参数传入（语义等价于 CUDA 的符号地址）。
 *
 * 字段：
 *   module      —— 符号所属模块句柄；由主机桩在模块加载后填充（运行时）。
 *   symbol_name —— 设备侧符号名（如 "c_pi"），用于 cuModuleGetGlobal 解析。
 *   size        —— 符号字节数（供 coolGetSymbolSize 与越界校验）。
 */
typedef struct coolSymbolDesc {
    coolModule_t module;
    const char* symbol_name;
    size_t size;
} coolSymbolDesc;

#ifdef __cplusplus
}
#endif

#endif /* COOL_TYPES_H */
