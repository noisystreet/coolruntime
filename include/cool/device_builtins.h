// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * COOLRuntime 设备端内建（device builtins）。
 *
 * 对应 CUDA 的 cuda_runtime.h 中「设备侧」部分：执行/存储限定符宏、内建向量类型、
 * 内建变量（threadIdx 等）与内建函数（__syncthreads 等）。
 *
 * 分工（见 hacc 的 tools/il_spike/cuda-attributes.md §5）：
 *   - 限定符宏（__global__ 等）→ 映射为 __attribute__((cool_*))，由 hacc 前端识别。
 *   - 内建变量/函数 → 由 hacc 设备端（coolicc）按名字映射到 NVVM intrinsic。
 *
 * 本头文件保持 C 兼容：只用 C 类型，禁止 STL / 模板 / 引用。
 *
 * 设备/主机区分：内建变量与内建函数**仅在设备编译时**声明（对标 CUDA 的 __CUDACC__）。
 * hacc 的 coolfe++ 定义 __COOLCC__；主机编译器（g++/clang++）不定义，故主机侧
 * 不会看到 threadIdx 等符号，避免未定义引用。限定符宏则两端都定义（主机侧也需解析
 * 内核声明以生成桩）。
 */
#ifndef COOL_DEVICE_BUILTINS_H
#define COOL_DEVICE_BUILTINS_H

#include "cool_types.h"

/* ------------------------------------------------------------------ */
/* 执行 / 存储限定符宏（两端可见）                                      */
/* ------------------------------------------------------------------ */
/*
 * 映射为 EDG 未知属性（A2 决策 (a)）；名字与语义见 cuda-attributes.md §1/§2。
 * 主机编译器（g++/clang++）不认识 cool_* 属性，会告警——故仅在 __COOLCC__ 下展开为
 * 属性，否则展开为空（主机侧限定符无副作用）。
 */
#ifdef __COOLCC__
#define __global__ __attribute__((cool_global))
#define __device__ __attribute__((cool_device))
#define __host__ __attribute__((cool_host))
#define __shared__ __attribute__((cool_shared))
#define __constant__ __attribute__((cool_constant))
/* __restrict__ 在设备编译时映射为 cool_restrict 属性（S1/M4.2，→ LLVM noalias）。
 * 主机侧不覆盖：__restrict__ 是 GCC/Clang 标准关键字，保持原义。 */
#define __restrict__ __attribute__((cool_restrict))
#else
#define __global__
#define __device__
#define __host__
#define __shared__
#define __constant__
#endif

/* ------------------------------------------------------------------ */
/* 内建向量类型（两端可见；与 CUDA 的 make_* 对应）                     */
/* ------------------------------------------------------------------ */

typedef struct uint3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
} uint3;

typedef struct int3 {
    int x;
    int y;
    int z;
} int3;

/* 2 分量向量（对齐 8，与 CUDA 一致）。 */
typedef struct __attribute__((aligned(8))) float2 {
    float x;
    float y;
} float2;

typedef struct __attribute__((aligned(8))) int2 {
    int x;
    int y;
} int2;

typedef struct __attribute__((aligned(8))) uint2 {
    unsigned int x;
    unsigned int y;
} uint2;

typedef struct __attribute__((aligned(16))) double2 {
    double x;
    double y;
} double2;

/* 3 分量向量。注意：与 CUDA 一致，float3 无 padding（sizeof == 12，对齐 4）。 */
typedef struct float3 {
    float x;
    float y;
    float z;
} float3;

/* 4 分量向量（对齐 16，与 CUDA 一致）。 */
typedef struct __attribute__((aligned(16))) float4 {
    float x;
    float y;
    float z;
    float w;
} float4;

typedef struct __attribute__((aligned(16))) int4 {
    int x;
    int y;
    int z;
    int w;
} int4;

typedef struct __attribute__((aligned(16))) uint4 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int w;
} uint4;

typedef struct __attribute__((aligned(16))) double4 {
    double x;
    double y;
    double z;
    double w;
} double4;

/* ------------------------------------------------------------------ */
/* 设备端内建变量与函数（仅设备编译可见）                               */
/* ------------------------------------------------------------------ */
#ifdef __COOLCC__

/* 内建变量：由 coolicc 按名字映射到 NVVM 特殊寄存器（%tid.x 等）。 */
extern uint3 threadIdx;
extern uint3 blockIdx;
extern uint3 blockDim;
extern uint3 gridDim;
extern unsigned int warpSize;

/* 内建函数：由 coolicc 按名字映射到 NVVM intrinsic（llvm.nvvm.barrier0 等）。 */
void __syncthreads(void);

/* 内存栅栏（S1/M4.3）：映射到 llvm.nvvm.membar.{gl,cta,sys}。
 * 语义与 CUDA 一致：__threadfence 影响设备全局可见顺序，__threadfence_block
 * 限于线程块内，__threadfence_system 覆盖主机与对等设备。三者均无参。 */
void __threadfence(void);
void __threadfence_block(void);
void __threadfence_system(void);

/* 原子内建（S1/M4.4）：由 coolicc 映射到 LLVM 原生 `atomicrmw` / `cmpxchg`
 * （不用 llvm.nvvm.atomic.* —— 那套只有 cta/sys 作用域，表达不出 CUDA 默认的
 * device scope；实测见 hacc 的 m4-subset-s1-plan.md §3.5）。
 * 签名按 CUDA，返回**运算前的旧值**。
 * 一期覆盖 int / unsigned int / float 的常用运算；unsigned long long 等按需扩展。 */
int atomicAdd(int* address, int val);
unsigned int atomicAdd(unsigned int* address, unsigned int val);
float atomicAdd(float* address, float val);

int atomicSub(int* address, int val);

int atomicExch(int* address, int val);
unsigned int atomicExch(unsigned int* address, unsigned int val);

/* 有符号/无符号的 min/max 语义不同，必须靠重载区分。 */
int atomicMin(int* address, int val);
int atomicMax(int* address, int val);
unsigned int atomicMin(unsigned int* address, unsigned int val);
unsigned int atomicMax(unsigned int* address, unsigned int val);

/* atomicCAS：若 *address == compare 则写入 val，返回**替换前的旧值**。 */
int atomicCAS(int* address, int compare, int val);
unsigned int atomicCAS(unsigned int* address, unsigned int compare, unsigned int val);

#endif /* __COOLCC__ */

#endif /* COOL_DEVICE_BUILTINS_H */
