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

/*
 * 错误码：coolSuccess == 0 表示成功。
 *
 * 数值按 cudart 的 cudaError 编号，便于与 CUDA 侧逐项对照（P6 R2 起）。
 *
 * **已知数值差异（历史 ABI，本轮未改）** —— 改这些值会破坏已发布 ABI，需单独变更：
 *   - `coolErrorMemoryAllocation = 3`：cudart 的 3 是 cudaErrorInitializationError，
 *     故本仓**没有** InitializationError（初始化失败统一报 coolErrorNotReady）；
 *   - `coolErrorInvalidDevice = 10`：cudart 的 cudaErrorInvalidDevice 是 **101**，
 *     而 10 在 cudart 是 cudaErrorVersionTranslation；
 *   - `coolErrorHostMemoryAlreadyRegistered = 711` / `NotRegistered = 712`：
 *     cudart 分别是 **712** / **713**（711 在 cudart 是 cudaErrorTooManyPeers）。
 * 即：**按枚举名对接，不要按数值跨库比对**。取舍见 docs/runtime-api-plan.md §R2。
 */
typedef enum coolError {
    coolSuccess = 0,
    coolErrorInvalidValue = 1,
    coolErrorOutOfMemory = 2,
    coolErrorMemoryAllocation = 3,

    /* 配置 / 参数形态类（P6 R2 新增）。 */
    coolErrorInvalidConfiguration = 9,
    coolErrorInvalidDevice = 10, /* cudart = 101，见上方数值差异说明 */
    coolErrorInvalidPitchValue = 12,
    coolErrorInvalidSymbol = 13,
    coolErrorInvalidDevicePointer = 17,
    coolErrorInvalidMemcpyDirection = 21,
    coolErrorInsufficientDriver = 35,

    /* 设备可见性 / 函数与镜像（P6 R2 新增）。 */
    coolErrorInvalidDeviceFunction = 98,
    coolErrorNoDevice = 100,
    coolErrorInvalidKernelImage = 200,
    coolErrorUnsupportedLimit = 215,

    /* 模块加载（P6 R2 新增）。 */
    coolErrorInvalidSource = 300,
    coolErrorFileNotFound = 301,

    coolErrorInvalidResourceHandle = 400,
    coolErrorNotReady = 600,

    /* 内核执行期（P6 R2 新增）。主要收益：把「kernel 跑炸了」与「参数传错了」分开。 */
    coolErrorIllegalAddress = 700,
    coolErrorLaunchOutOfResources = 701,
    coolErrorLaunchTimeout = 702,
    coolErrorLaunchIncompatibleTexturing = 703,
    coolErrorHostMemoryAlreadyRegistered = 711, /* cudart = 712，见上方数值差异说明 */
    coolErrorHostMemoryNotRegistered = 712,     /* cudart = 713，见上方数值差异说明 */
    coolErrorMisalignedAddress = 716,
    coolErrorInvalidAddressSpace = 717,
    coolErrorInvalidPc = 718,
    coolErrorLaunchFailure = 719,

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

/*
 * 设备属性查询项（P6 R1），对应 cudart 的 cudaDeviceAttr。
 *
 * 枚举值**按 cudart 的 cudaDeviceAttr 编号**，便于与 CUDA 源码逐行对照；只收录
 * 实际有用、且 Driver API 有对应 CU_DEVICE_ATTRIBUTE_* 的子集。未收录的取值由
 * coolDeviceGetAttribute 返回 coolErrorNotSupported —— 不是 0，因为「查不到」与
 * 「查到 0」必须可区分（0 在 CUDA 里是合法取值，如 UnifiedAddressing 关闭）。
 */
typedef enum coolDeviceAttribute {
    coolDevAttrMaxThreadsPerBlock = 1,
    coolDevAttrMaxBlockDimX = 2,
    coolDevAttrMaxBlockDimY = 3,
    coolDevAttrMaxBlockDimZ = 4,
    coolDevAttrMaxGridDimX = 5,
    coolDevAttrMaxGridDimY = 6,
    coolDevAttrMaxGridDimZ = 7,
    coolDevAttrMaxSharedMemoryPerBlock = 8,
    coolDevAttrWarpSize = 10,
    coolDevAttrMaxRegistersPerBlock = 12,
    coolDevAttrClockRate = 13,
    coolDevAttrMultiProcessorCount = 16,
    coolDevAttrConcurrentKernels = 31,
    coolDevAttrMemoryClockRate = 36,
    coolDevAttrGlobalMemoryBusWidth = 37,
    coolDevAttrL2CacheSize = 38,
    coolDevAttrMaxThreadsPerMultiProcessor = 39,
    coolDevAttrAsyncEngineCount = 40,
    coolDevAttrUnifiedAddressing = 41,
    coolDevAttrComputeCapabilityMajor = 75,
    coolDevAttrComputeCapabilityMinor = 76,
    coolDevAttrMaxSharedMemoryPerMultiprocessor = 81
} coolDeviceAttribute;
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

    /* P6 R1 追加字段：只增不改，一律置于尾部（见 docs/runtime-api-plan.md §7 第 3 条）。
     * 与 coolDeviceGetAttribute 同源，unit 测试交叉断言两者一致。 */
    int maxThreadsDim[3];           /* 每维最大线程数 */
    int sharedMemPerBlock;          /* 每块静态共享内存字节数 */
    int sharedMemPerMultiprocessor; /* 每 SM 共享内存字节数 */
    int regsPerBlock;               /* 每块可用 32 位寄存器数 */
    int clockRate;                  /* 峰值时钟频率（kHz） */
    int memoryClockRate;            /* 峰值显存时钟频率（kHz） */
    int globalMemoryBusWidth;       /* 全局内存总线宽度（bit） */
    int l2CacheSize;                /* L2 缓存字节数 */
    int unifiedAddressing;          /* 是否与主机统一地址空间（0/1） */
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

/*
 * 内核函数属性（P5.5）：对应 cudaFuncAttributes 的核心子集。
 *
 * 与 cudart 的差异：只暴露 Driver API 能直接查到的 5 项（cuFuncGetAttribute 逐项查询，
 * Driver 侧没有批量接口），不含 binaryVersion / cacheModeCA / cluster 等字段 ——
 * 那些要么需要额外上下文，要么尚无用例。字段只增不改，后续按需扩展。
 */
typedef struct coolFuncAttributes {
    int maxThreadsPerBlock; /* 单块最大线程数 */
    int numRegs;            /* 每线程寄存器用量 */
    int sharedSizeBytes;    /* 静态共享内存字节数 */
    int localSizeBytes;     /* 每线程本地内存字节数 */
    int constSizeBytes;     /* 用户常量区字节数 */
} coolFuncAttributes;

#ifdef __cplusplus
}
#endif

#endif /* COOL_TYPES_H */
