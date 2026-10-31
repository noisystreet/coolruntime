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
/* 执行配置提示（两端可见，当前**展开为空**）                            */
/* ------------------------------------------------------------------ */
/*
 * __launch_bounds__(maxThreadsPerBlock[, minBlocksPerMultiprocessor])：
 * CUDA 用它提示每块最大线程数与每 SM 最少驻留块数，影响**寄存器分配与 occupancy**，
 * 不影响正确性。
 *
 * hacc 的处理是「接受但忽略」（见 hacc 的 docs/s2-plan.md T4）：设备端目前 -O0、
 * 不做寄存器压力优化，映射到 PTX `.maxntid` / `.minnctapersm` 无实际收益；反之，
 * 若不给定义，真实工程代码会直接编不过（实测：未定义时 coolfe++ 报
 * `incomplete type "void" is not allowed`）。
 *
 * 因此这里展开为空：能编、能跑，行为与不带该提示时**完全相同**。
 * 将来若做寄存器/占用率优化，再改判成 __attribute__((cool_launch_bounds(...)))
 * 并在 hacc 前端识别。
 */
#define __launch_bounds__(...)

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
/* MMA（Tensor Core）wmma fragment（C12；ADR 0008 已批准）              */
/* ------------------------------------------------------------------ */
/*
 * m16n16k16 的 fragment：A/B 各 8 个 `<2 x half>`，C/D 各 8 个 `float`，一律 **32 字节**。
 *
 * 为什么用整型/浮点数组占位，而不是 16 位浮点：
 *   - hacc 的 `map_float_type` 尚未覆盖 16 位浮点 kind（`fk_float16` / `fk_fp16`），
 *     用户可见的 `half` 类型目前编不过；
 *   - MVP 的正确性由**矩阵数据布局**保证（f16 数据由用户自己填进缓冲），
 *     与 fragment 的内部视图无关。
 *
 * 为什么是 `unsigned int` / `float` 而不是 `unsigned short`：
 *   两者都是 32 字节，但 `unsigned short[16]` 的对齐只有 2，而 hacc 生成的
 *   intrinsic 聚合值（`{<2 x half> × 8}` / `{float × 8}`）对齐是 4。hacc 目前不读
 *   类型的 aligned 属性（会丢弃），局部变量的对齐取自字段类型 —— align 2 的槽位
 *   存 align 4 的值属未定义行为。用 u32 / float 占位让两侧对齐一致，不依赖属性传递。
 *
 * 四个**互不相同**的结构体名是刻意的：hacc 在出参 ABI 下不校验 fragment 类型
 * （内部一律按 32 字节处理），把「A/B/C/D 传反」的检查挪回 C 编译器。
 */
typedef struct cool_wmma_frag_a {
    unsigned int m[8];
} cool_wmma_frag_a; /* 32B，对齐 4：A 矩阵（row major） */

typedef struct cool_wmma_frag_b {
    unsigned int m[8];
} cool_wmma_frag_b; /* 32B，对齐 4：B 矩阵（col major） */

typedef struct cool_wmma_frag_c {
    float m[8];
} cool_wmma_frag_c; /* 32B，对齐 4：C 累加器 */

typedef struct cool_wmma_frag_d {
    float m[8];
} cool_wmma_frag_d; /* 32B，对齐 4：D 结果累加器 */

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

/* 位运算原子（P5.1）：映射到 atomicrmw and / or / xor。
 * 位运算不区分有无符号（PTX 均为 .b32），重载只为与 CUDA 签名一致。
 * 返回值同其它原子算子，为**运算前的旧值**。 */
int atomicAnd(int* address, int val);
unsigned int atomicAnd(unsigned int* address, unsigned int val);

int atomicOr(int* address, int val);
unsigned int atomicOr(unsigned int* address, unsigned int val);

int atomicXor(int* address, int val);
unsigned int atomicXor(unsigned int* address, unsigned int val);

/* wmma（MMA / Tensor Core，C12）：由 hacc 映射到 llvm.nvvm.wmma.* intrinsic。
 *
 * ABI 是**出参**（fragment 指针由实现写入/读取），不是值返回：
 *   - 用户代码不需要对 fragment 做元素级运算，值语义没有收益；
 *   - 值返回要求 hacc 生成「多返回值 struct intrinsic 的 SSA 管道」，
 *     与内建映射不支持带参的缺口叠在一起。
 * 将来要支持元素级运算（手工缩放累加器等）时再补值语义通道。
 *
 * **三条红线（hacc 侧全部不设防，只能靠用户遵守）**：
 *   1. warp 级语义：32 个 lane 必须一致执行到调用点。放进 `if (tid < k)` 之类的
 *      分支里**不报错但结果错**（与 KI-1 同类：能编译、看着跑通、数值不对）。
 *   2. load/store 的 `src` / `dst` 必须 **32 字节对齐**（PTX 用 `.aligned` 变体，
 *      hacc 不读对齐属性）。`coolMalloc` 实测 256B 对齐，缓冲直接用即可；
 *      需要偏移时先拷到对齐缓冲，不要把 `ptr + i` 传进来。
 *   3. A/B/C/D 传反**不报错**：hacc 内部一律按 32 字节处理，编译期能抓的只有
 *      C 层的四个不同结构体名。顺序固定为 D = A×B + C。
 */
void cool_wmma_load_a_row(cool_wmma_frag_a* frag, const void* src);
void cool_wmma_load_b_col(cool_wmma_frag_b* frag, const void* src);
void cool_wmma_load_c_row(cool_wmma_frag_c* frag, const void* src);
void cool_wmma_mma(cool_wmma_frag_d* d, const cool_wmma_frag_a* a, const cool_wmma_frag_b* b,
                   const cool_wmma_frag_c* c);
void cool_wmma_store_d_row(void* dst, const cool_wmma_frag_d* d);

/* ------------------------------------------------------------------ */
/* 数学内建（hacc S2 T3）—— 仅 f32，且**精度契约明确**                 */
/* ------------------------------------------------------------------ */
/*
 * 由 coolicc 按名字映射到单条 PTX 指令或 NVVM intrinsic，不是函数调用。
 *
 * 收录原则（三条，刻意保守）：
 *   1. **能落到单条精确指令的**收录：sqrt.rn / abs / min / max / cvt.rmi·rpi·rzi /
 *      fma.rn / rcp.rn / sqrt.rn —— 与 CUDA 同名函数语义一致，可直接用。
 *   2. **名字本身即「近似」语义的**收录：__expf / __logf / __sinf / __cosf / __fdividef
 *      / rsqrtf —— CUDA 里它们的定义就是 `.approx` 指令，用户选这个名字时已接受
 *      「精度换速度」。
 *   3. **需要 libdevice 精确实现的**（sinf / cosf / expf / logf / powf / tanf /
 *      atan2f / fmodf …）**不收录**：调用时 hacc 会明确报错，而不是悄悄返回一个
 *      `.approx` 的近似值。理由：sinf 与 __sinf 在 CUDA 里是两个不同精度档，
 *      把前者降级成后者是「静默降低精度」，比编不过更糟（同 KI 系列的口径）。
 *
 * 因此本头**故意只声明这 16 个**。将来接入 libdevice 再补精确版，届时逐个标注精度。
 *
 * 注意：这些声明**没有** `__device__` 属性（与 `__syncthreads` 一致）——它们不是
 * 用户可定义的设备函数，而是编译器内建；加属性反而会进入 device 函数表。
 */
float sqrtf(float x);
float fabsf(float x);
float fminf(float x, float y);
float fmaxf(float x, float y);
float floorf(float x);
float ceilf(float x);
float truncf(float x);
float rsqrtf(float x);
float __fmaf_rn(float a, float b, float c);
float __fsqrt_rn(float x);
float __frcp_rn(float x);
float __fdividef(float x, float y);
float __expf(float x);
float __logf(float x);
float __sinf(float x);
float __cosf(float x);

/* ------------------------------------------------------------------ */
/* 整数单指令内建 + 内核内计时（P6/R5 第一批）                          */
/* ------------------------------------------------------------------ */
/*
 * 与数学内建同一套纪律：**一个名字 → 一条（或明确几步）PTX 指令**，由 coolicc
 * 按名字映射，不是函数调用；声明本身同样不带 `__device__` 属性。
 *
 * 边界值逐条写明 —— 这几条正是「静默错值」的高发区：
 *   - `__popc(x)`  置位数（0..32）                                  → `popc.b32`
 *   - `__clz(x)`   前导零个数（0..32），**`__clz(0) == 32`**         → `clz.b32`
 *   - `__ffs(x)`   最低位 1 的**1-based**位置（0..32），**`__ffs(0) == 0`**
 *                  → `cttz(x) + 1`，且 x==0 时特判为 0（CTZ 对 0 给出 32）
 *   - `__brev(x)`  32 位位序反转                                    → `brev.b32`
 *
 * 计时（核内读数，单位是 SM 时钟周期，可用来算区间耗时）：
 *   - `clock()`   → PTX `%clock`（**32 位**，溢出后回绕）
 *   - `clock64()` → PTX `%clock64`（64 位）
 *   **与 CUDA 的差异**：CUDA 的 `clock()` 返回 `clock_t`（Linux 上是 `long`）；
 *   本仓直接返回 `unsigned int` / `unsigned long long`，不为计时引入平台相关的
 *   `clock_t` 定义。差值语义与 CUDA 一致：同一 SM 内单调递增，跨 SM 不保证。
 *   注意两者**只能做差**，绝对值无跨设备意义。
 *
 * `__clz` / `__ffs` 取 `int`、`__popc` / `__brev` 取 `unsigned int`：与 CUDA 签名
 * 一致。位操作本身不区分符号性，重载只为让签名对齐。
 */
int __popc(unsigned int x);
int __clz(int x);
int __ffs(int x);
unsigned int __brev(unsigned int x);

unsigned int clock(void);
unsigned long long clock64(void);

/* ------------------------------------------------------------------ */
/* 只读数据缓存加载（P6 R5 第二批）                                     */
/* ------------------------------------------------------------------ */
/*
 * `__ldg(p)` 声明「p 指向的数据在 kernel 生命周期内只读」，允许走非相干只读
 * 缓存（PTX `ld.global.nc`），省掉 L1 一致性维护的开销。
 *
 * **这是一条强承诺，不是优化提示**：LLVM 的 `!invariant.load` 等价于「这块内存
 * 在本函数内不会变」。若同一 kernel 内既 `__ldg` 又写同一块内存，编译器可能把
 * 重复读合并成一次 —— 与 CUDA 的 `__ldg` 约束一致（用户承诺只读），
 * hacc 侧**不做任何检查**。
 *
 * **红线：只能传指向全局内存的指针**。传 `__shared__` / local 指针是未定义行为
 * （与 CUDA 一致）：实现会把指针 `cvta.to.global` 后再读，指向的区域根本不对，
 * 且不会报错。
 *
 * 第一批只收录标量；向量的 `__ldg`（`float4` / `int4` 等）待补。
 */
int __ldg(const int* p);
unsigned int __ldg(const unsigned int* p);
float __ldg(const float* p);
double __ldg(const double* p);

#endif /* __COOLCC__ */

#endif /* COOL_DEVICE_BUILTINS_H */
