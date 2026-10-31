# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses semantic versioning when releases are created.

## [Unreleased]

### Added

- **整数内建与核内计时声明（P6 R5 第一批）**：`device_builtins.h` 的 `__COOLCC__` 段新增
  `__popc` / `__clz` / `__ffs` / `__brev` / `clock()` / `clock64()` 六个声明，由 hacc 的
  coolicc 映射到单条 PTX 指令（`popc.b32` / `clz.b32` / `brev.b32` / `%clock` / `%clock64`；
  `__ffs` 走 `cttz + 1` 且对 0 特判）。

  **边界语义写进头注释，并在 hacc 的 e2e 里逐个断言**：`__clz(0) == 32`、`__ffs(0) == 0`、
  `__ffs` 是 **1-based** —— 这几条少一个特判就是静默错值（实测去掉特判后返回 33）。

  计时**与 CUDA 的差异**：`clock()` 返回 `unsigned int`（CUDA 的 `clock_t` 在 Linux 上是
  `long`），`clock64()` 返回 `unsigned long long`；不为计时引入平台相关的 `clock_t`。
  两者**只能做差**：同一 SM 内单调递增，跨 SM 不保证，绝对值无跨设备意义。

  `__ldg` **本批不做**：需先 spike「NVPTX 如何生成 `ld.global.nc`」与「按实参类型重载」。

- **`coolDeviceGetAttribute` + 设备属性补全（P6 R1）**：新增公开 API
  `coolError_t coolDeviceGetAttribute(int* value, coolDeviceAttribute attr, int device)`，
  枚举 `coolDeviceAttribute` 的取值**按 cudart 的 `cudaDeviceAttr` 编号**（22 项：块/网格维度上限、
  共享内存、寄存器、时钟、L2、并发内核、异步引擎、统一寻址、CC 版本等）。
  未收录属性返回 `coolErrorNotSupported` 且**不写 `*value`** —— 「查不到」与「查到 0」必须可区分
  （0 在 CUDA 里是合法值，如 `UnifiedAddressing` 关闭）。`coolDeviceProp` 尾部追加 10 个字段
  （`maxThreadsDim[3]` / `sharedMemPerBlock` / `sharedMemPerMultiprocessor` / `regsPerBlock` /
  `clockRate` / `memoryClockRate` / `globalMemoryBusWidth` / `l2CacheSize` / `unifiedAddressing`），
  **只增不改**。两个入口（属性查询与 `coolGetDeviceProperties`）走同一张映射表，
  `test_device` 交叉断言同源同值。

- **错误码补全（P6 R2）**：`coolError_t` 从 11 项扩到 31 项，新增项的数值按 cudart 的
  `cudaError` 编号。主要收益是可诊断性：**内核执行期错误不再与「参数传错了」混淆** ——
  `CUresult` 的 `LAUNCH_FAILED` / `ILLEGAL_ADDRESS` / `LAUNCH_OUT_OF_RESOURCES` /
  `LAUNCH_TIMEOUT` / `MISALIGNED_ADDRESS` / `INVALID_PC` / `INVALID_ADDRESS_SPACE` 等
  此前只能报 `coolErrorUnknown`（当时确实没有对应码），现在各自映射到专码；
  `NO_DEVICE` 也从 `coolErrorInvalidDevice` 改为 `coolErrorNoDevice`（「没有卡」≠「序号非法」）。
  镜像 / 源码 / 文件类同步收窄（`INVALID_IMAGE` / `INVALID_PTX` → `InvalidKernelImage`，
  `INVALID_SOURCE` → `InvalidSource`，`FILE_NOT_FOUND` → `FileNotFound`）。

  **已知数值差异（本轮刻意未改，属历史 ABI）**：`coolErrorInvalidDevice = 10`（cudart 是 101）、
  `coolErrorMemoryAllocation = 3`（cudart 的 3 是 InitializationError，故本仓没有该码）、
  `coolErrorHostMemoryAlreadyRegistered = 711` / `NotRegistered = 712`（cudart 是 712 / 713）。
  改这些值会破坏已发布 ABI，需单独变更并写迁移说明；`cool_types.h` 与 `test_error` 均显式锁住
  现状，避免被顺手改掉。**对接请按枚举名，不要按数值跨库比对。**

  验证：`test_error` 逐项锁数值锚点 + name/string 可查；`test_result_map` 覆盖新增映射与
  「仍保留 Unknown」的边界。**反向验证**：把 `NO_DEVICE` 映射改回旧值 + 让属性查表恒返回
  `nullptr`，14 条断言变红；恢复后 `make check` 15/15。

- **接口补齐规划文档（P6）**：[docs/runtime-api-plan.md](docs/runtime-api-plan.md) 立项。
  清点现状（主机 API **52 个** + 设备内建清单），按「纯主机侧优先 / 声明先行 ≠ 能力可用 /
  可诊断性 > 功能数量」三原则分档（R1 设备属性、R2 错误码、R3 2D·pitch 内存族、R4 warp 原语、
  R5 位操作与向量、R6 printf·assert、R7 共享内存配套、R8 流补遗、R9 JIT 选项），
  并列出三个开工前置 spike（warp intrinsic 形态 / `extern __shared__` 的 IL 形态 /
  可变参调用与聚合返回值）。**只是规划，未改动任何头文件与实现**。

- **数学内建声明（hacc S2 T3）**：`device_builtins.h` 的 `__COOLCC__` 段新增 16 个 f32 内建
  声明，由 coolicc 按名字映射到单条 PTX 指令：`sqrtf`、`fabsf`、`fminf`、`fmaxf`、`floorf`、
  `ceilf`、`truncf`、`rsqrtf`、`__fmaf_rn`、`__fsqrt_rn`、`__frcp_rn`、`__fdividef`、
  `__expf`、`__logf`、`__sinf`、`__cosf`。

  **故意只声明这 16 个**：需要 libdevice 精确实现的 `sinf`/`cosf`/`expf`/`logf`/`powf` 等
  **不声明**，因为 hacc 尚未内置它们的精确实现。用户若自行声明并调用（或 `#include
  <math.h>`），hacc 会明确报「需要 libdevice 的精确实现」并建议改用近似内建——而不是
  悄悄给出一个 `.approx` 的近似值。

  这些声明**不带 `__device__` 属性**（与 `__syncthreads` 一致）：它们是编译器内建，加属性
  反而会让它们进入 hacc 的设备函数表。

- **`__launch_bounds__` 接受但忽略（hacc S2 T4）**：`device_builtins.h` 新增两端可见的空宏
  `#define __launch_bounds__(...)`。此前它**没有任何定义**，导致带该提示的真实代码
  **直接编不过**——coolfe++ 把 `void __launch_bounds__(256, 4) k(...)` 当成「`void` 类型的
  变量声明」，报 `incomplete type "void" is not allowed` 与 `expected a ";"`。
  展开为空而非映射到 PTX `.maxntid` / `.minnctapersm`：设备端目前 `-O0`、不做寄存器压力
  优化，映射无收益；而拒绝它会让工程代码无法迁移。行为与不带该提示时**完全相同**。
  回归用例在 hacc：`tests/golden/e2e/launch_bounds.cu`（单参 / 双参 + 真机数值校验；
  反向验证：去掉本宏后该用例在编译阶段失败）。

- **wmma fragment 与内建声明（MMA / hacc C12 的 A1）**：`device_builtins.h` 新增 4 个
  fragment 类型（`cool_wmma_frag_a` / `_b` / `_c` / `_d`，各 32 字节、对齐 4）与 5 条内建
  声明（`cool_wmma_load_a_row` / `_b_col` / `_c_row`、`cool_wmma_mma`、`cool_wmma_store_d_row`），
  全部是**出参 ABI**（fragment 指针由实现写入/读取，不返回值）。
  MVP 不需要 `half`：A/B 用 `unsigned int[8]` 占位、C/D 用 `float[8]`，正确性由矩阵数据
  布局保证 —— 用户自己把 f16 数据填进缓冲，与 fragment 的内部视图无关。
  **为什么不是 `unsigned short[16]`**：同样 32 字节，但对齐只有 2，而 hacc 侧 intrinsic
  聚合值（`{<2 x half> × 8}` / `{float × 8}`）对齐是 4；hacc 目前不读类型的 `aligned`
  属性（会丢弃），align 2 的槽位存 align 4 的值属未定义行为，且只在真机数值上暴露。
  四个**互不相同**的结构体名是刻意的 —— 出参 ABI 下 hacc 不校验 fragment 类型，
  「A/B/C/D 传反」只能靠 C 编译器抓。
  头注释写死了三条红线（hacc 侧都不设防）：**整 warp 一致执行**（包进 `if (tid < k)`
  不报错但结果错）、**load/store 的缓冲 32 字节对齐**、**A/B/C/D 顺序固定 D = A×B + C**。
  新增 `tests/unit/test_wmma_frag.cpp`（14 项检查：32 字节 / 对齐 4 / 可平凡复制 /
  四个类型互不相同）。`make check` **15/15**。

- **查询类与元数据 API（P5.5）**：补齐 `coolStreamGetPriority` / `coolStreamGetFlags` /
  `coolDeviceGetStreamPriorityRange` / `coolFuncGetAttributes`，均为薄透传
  （`cuStreamGetPriority` / `cuStreamGetFlags` / `cuCtxGetStreamPriorityRange` /
  逐项 `cuFuncGetAttribute`）。
  新增公开结构体 **`coolFuncAttributes`**（5 项：`maxThreadsPerBlock` / `numRegs` /
  `sharedSizeBytes` / `localSizeBytes` / `constSizeBytes`）。Driver 侧没有 cudart 那样的批量
  查询接口，故由后端循环填；不含 `binaryVersion` / `cacheModeCA` / cluster 字段，字段只增不改。
  **踩到一处语义坑并已纠正**：流优先级**数值越小优先级越高**，
  所以 `leastPriority` 是数值**最大**的一端（本机实测 0）、`greatestPriority` 是最小的一端（-5），
  有效区间 `[greatestPriority, leastPriority]`。我最初把注释和测试断言都写反了，
  是 `test_query` 跑出 `[0, -5]` 才暴露。
  新增 `tests/unit/test_query.cpp`（30 项检查）：参数校验（含未注册 kernel 句柄必须被拒
  且**失败时不改写输出结构**）、优先级范围单调性、默认流优先级/标志恒为 0、
  「按最高优先级 + non-blocking 建流后原样读出」、内嵌 PTX 读内核属性。
  `make check` **14/14**。

- **设备端位运算原子（P5.1）**：`device_builtins.h` 新增 `atomicAnd` / `atomicOr` / `atomicXor`
  各两个重载（`int*` + `unsigned int*`）。纯声明，无运行时改动 —— 位运算不分符号，
  PTX 均为 `.b32`，重载只为对齐 CUDA 签名。
  hacc 侧同步补上映射：`atomicrmw and/or/xor` → `atom.global.{and,or,xor}.b32`，
  真机竞争验证通过（`0 OR 2` 重复 128 次得 2、`0xFFFFFFFF AND 0xFF` 得 `0xFF`）。
  **已知限制**：hacc 的表达式翻译尚不支持 `<<` `>>` `&` `|` `^` `~`，因此目前只能传
  字面量或已算好的变量，写不出 `atomicOr(&flag, 1u << tid)` —— 已记入本仓 `docs/TODO.md`
  与 hacc 的 TODO B9。

- **异步拷贝与页锁定注册（P5.4）**：`coolMemcpyAsync` / `coolMemsetAsync` /
  `coolHostRegister` / `coolHostUnregister`。此前有流、有事件、有 `coolMallocHost`，
  但**所有拷贝都是同步的**，写不出「拷贝与计算重叠」的流水。
  - `coolMemcpyAsync` 按 `kind` 分派到 `cuMemcpyHtoDAsync` / `DtoHAsync` / `DtoDAsync`；
    `coolMemcpyDefault` 走 `cuMemcpyAsync`（UVA 下由驱动推断方向）；
    `coolMemcpyHostToHost` 因 Driver API 无 `HtoHAsync` 退化为同步 `cuMemcpy`（与 cudart 一致）。
  - `coolHostRegister` 的判重与「未注册」交给驱动权威判定，经 `resultToError` 映射为新增的
    `coolErrorHostMemoryAlreadyRegistered`(711) / `coolErrorHostMemoryNotRegistered`(712)，
    数值与 cudart 对齐；以驱动为准可避开「先查表再注册」的 TOCTOU 窗口。
  - 后端注册表新增 `registeredHost_`（与 `mallocHost` 用的 `host_` 分开），用于最终清理。
  - 新增 `test_mem_async`：参数与句柄校验、`H2D → D2H` 异步流水（自定义流与默认流）、
    异步 `memset` 往返、注册状态机（重复注册 / 重复注销）。
    刻意**不**断言「确实发生了重叠」—— 那取决于驱动与硬件调度，无法在单元测试里稳定观测。
- **`docs/TODO.md`**：把设备内建（原子补完 / 数学函数 / warp 原语 / 位操作 / `__ldg` 等）、
  主机异步拷贝与查询类 API 的待办，连同 Non-Goals 集中到一处。
  此前这些只知道「缺什么」，散落在 `ARCHITECTURE.md` 的 3 行 Open Decisions 里，没有可跟踪的清单。
  内容基于对两个仓的实测盘点：例如数学函数区分「零依赖」与「需 libdevice」两档，
  避免把近似指令误当精确语义。
- `README` 「已实现」表补上**设备符号四件套**（`coolMemcpyToSymbol` / `coolMemcpyFromSymbol` /
  `coolGetSymbolAddress` / `coolGetSymbolSize`）—— 代码早已实现并有 `test_symbol` 覆盖，表格漏列。

- **R 层转发逻辑可在无 GPU 下测试**：新增 `cool::installBackendForTesting()`（内部头
  `core/lifecycle.hpp`，测试专用），可注入假后端替换真实 cuda 后端。这一点能成立是因为
  BAL 用函数指针结构体表达能力 —— 转发逻辑本就与后端实现无关。
  新增 `test_api_dispatch`，**不需要 GPU**（跑在 0.00 s）即覆盖：参数校验先于初始化、
  懒初始化只发生一次、能力位缺失 → `coolErrorNotSupported`、后端错误码透传到返回值与
  TLS、实参按值转发（`size` / `coolMemcpyKind` 等）。此前所有测试都依赖真实驱动，
  R 层的这层逻辑等于没有回归保护。
- **`dim3` 补齐 CUDA 风格的 C++ 构造函数**（跨项目对齐，hacc TODO 的 B5）：新增
  `dim3(unsigned x = 1, unsigned y = 1, unsigned z = 1)`，支持 `dim3 block(256)` /
  `dim3 grid((n + 255) / 256)` 这类 CUDA 惯用写法（此前只能 `dim3 g = {a,b,c}`）。
  默认构造为 `(1,1,1)`，与 CUDA 一致（此前是未初始化）。
  **ABI 不变**：仍是 12 字节、对齐 4、可平凡复制（`coolLaunchKernel` 按值传 grid/block 的前提），
  C 下仍是纯聚合 struct（`#ifdef __cplusplus` 包住构造函数）。
  副作用：`dim3` 在 C++ 下不再是聚合类型，`dim3{x,y,z}` 改走构造函数（语义等价；hacc 的
  `to_dim3` 依赖这一点，已在其注释里写明）。
  新增 `test_dim3`：构造语义、默认值、`sizeof`/`alignof`/`is_trivially_copyable`、非聚合性。

- 共享库产物 `libcoolrt.so`（与静态库同源；经 OBJECT 库复用，不重复编译）。
- **Debian 打包**：`make package` 生成 `libcoolrt0_<ver>_<arch>.deb`（运行时）与
  `libcoolrt-dev_<ver>_<arch>.deb`（开发包）。按 Debian 惯例拆包：运行时包只带 soname
  （`libcoolrt.so.0`）；开发包带头文件、`libcoolrt_static.a`、裸名链接与 `pkg-config`
  文件 `coolrt.pc`。共享库新增 `VERSION` / `SOVERSION`（对齐 `libcudart.so.12`，
  soname 为 `libcoolrt.so.0`）。安装布局对齐 CUDA：前缀 **`/usr/local/cool`**，
  头在 `include/`、库在 `lib64/`（`CPACK_PACKAGING_INSTALL_PREFIX` 必须显式设置，
  CPack 默认会截成 `/usr`）。`coolrt.pc` 的 `Libs.private` 含 `-lcuda -lstdc++`，
  保证 C 程序用 `pkg-config --static` 链静态库时能成功（库内部为 C++ 实现）。
- 初始项目骨架（P0）：CMake 构建、CI、pre-commit。
- 公开头文件 `cool_version.h` / `cool_types.h` / `cool_runtime_api.h`（C ABI）。
- 统一入口头 `cool_runtime.h`（聚合 `cool_types.h` / `cool_version.h` / `cool_runtime_api.h`）。
- 后端抽象层 `cool_backend.h` 与后端访问器。
- `cuda`（Driver API）后端骨架。
- 版本管理、错误处理、设备管理接口。
- 内存管理接口（P1）：`coolMalloc` / `coolFree` / `coolMemcpy` / `coolMemset` / `coolMemGetInfo` / `coolMallocHost` / `coolFreeHost`。
- `cuda` 后端内存实现，以及后端侧内存元数据注册表（跟踪设备/主机指针，用于释放校验与重复释放识别）。
- 内存单元测试 `test_mem`。
- 流管理接口（P2）：`coolStreamCreate` / `coolStreamCreateWithFlags` / `coolStreamCreateWithPriority` / `coolStreamDestroy` / `coolStreamSynchronize` / `coolStreamQuery` / `coolStreamWaitEvent`。
- 事件管理接口（P2）：`coolEventCreate` / `coolEventCreateWithFlags` / `coolEventDestroy` / `coolEventRecord` / `coolEventSynchronize` / `coolEventQuery` / `coolEventElapsedTime`。
- `cuda` 后端流/事件实现，以及后端侧句柄注册表（跟踪流/事件句柄，用于销毁校验与重复销毁识别）。
- 流/事件单元测试 `test_stream` / `test_event`。
- 模块与内核接口（P3）：`coolModuleLoadData` / `coolModuleLoadFromFile` / `coolModuleGetFunction` / `coolModuleUnload` / `coolLaunchKernel` / `coolOccupancyMaxActiveBlocksPerMultiprocessor` / `coolOccupancyMaxPotentialBlockSize`。
- `cool_types.h` 新增模块句柄 `coolModule_t`。
- `cuda` 后端模块/函数/启动/占用率实现，并扩展后端侧句柄注册表（跟踪模块与函数地址，卸载模块时级联失效其函数）。
- 内核单元测试 `test_kernel`（内嵌最小 PTX，经驱动 JIT 加载，无需 nvcc/ptxas）。
- 单元测试（version / error / device）与 C ABI 编译校验、`samples/hello` 示例。
- 设备端内建头 `device_builtins.h`：执行/存储限定符宏（`__global__` / `__device__` /
  `__host__` / `__shared__` / `__constant__`，映射为 `__attribute__((cool_*))`）、内建向量类型
  （`uint3` / `int3`）、内建变量（`threadIdx` / `blockIdx` / `blockDim` / `gridDim` / `warpSize`）
  与内建函数（`__syncthreads`）。内建变量/函数仅在设备编译（`__COOLCC__`）下声明；
  由 `cool_runtime.h` 聚合，并纳入 C ABI 校验。
- `device_builtins.h` 新增原子内建声明（仅 `__COOLCC__` 下）：`atomicAdd` / `atomicSub` /
  `atomicExch` / `atomicMin` / `atomicMax` / `atomicCAS`，覆盖 `int` / `unsigned int` / `float`
  常用重载（`atomicAdd` 与 `atomicExch` 含 unsigned 重载；`atomicMin`/`atomicMax` 靠有符号与
  无符号重载区分语义）。由 `coolicc` 按名字映射到 LLVM `atomicrmw` / `cmpxchg`；返回运算前旧值。
- 设备符号拷入接口 `coolMemcpyToSymbol`（API 对齐 cudart 的 `cudaMemcpyToSymbol`）：
  `(symbol, src, count, offset, kind)`，把主机 / 设备内存写入设备符号；`kind` 支持
  `coolMemcpyHostToDevice`（默认语义）与 `coolMemcpyDeviceToDevice`，其余方向返回
  `coolErrorInvalidValue`，`offset + count` 越界亦拒绝。`symbol` 传 `coolSymbolDesc`
  描述符地址（替代 cudart 的「主机侧符号地址」编译期魔法，由 hacc 主机桩构造）。
  后端函数表新增 `memcpyToSymbol`，`cuda` 后端经 `cuModuleGetGlobal` 解析符号地址后
  走 `cuMemcpyHtoD` / `cuMemcpyDtoD`。同时提供 `cudaMemcpyToSymbol` 兼容别名（宏）。
- 设备符号拷出接口 `coolMemcpyFromSymbol`（API 对齐 cudart 的 `cudaMemcpyFromSymbol`）：
  与既有的 `coolMemcpyToSymbol` 对称，覆盖「设备符号 → 主机 / 设备内存」方向。
  `kind` 只接受 `coolMemcpyDeviceToHost`（默认语义）与 `coolMemcpyDeviceToDevice`，
  其余方向与越界均返回 `coolErrorInvalidValue`；同时提供 `cudaMemcpyFromSymbol` 兼容别名。
  后端函数表新增 `memcpyFromSymbol`，`cuda` 后端以 `cuMemcpyDtoH` / `cuMemcpyDtoD` 实现。
  补齐 `test_symbol` 覆盖（往返一致、偏移、越界、非法方向、未初始化描述符、别名可用）。

### Changed

- **默认流语义明确为 CUDA legacy 默认流**（[ADR 0005](docs/adr/0005-default-stream-semantics.md)）：
  `coolStream_t NULL` 与所有 blocking 流隐式同步，不支持 per-thread 默认流，也不为其预留开关。
  `ARCHITECTURE.md` 的 Open Decision「`COOL_STRICT_DEFAULT_STREAM` 严格模式是否默认开启」一并关闭 ——
  该项自 P2 起只有一行表格、代码中从未存在该开关，本次决定不引入（理由见 ADR，含两种可能解读的否定论证）。
  **无行为变更**：cuda 后端一直把 `NULL` 直接交给 `cuStream*` 系调用（`CUstream 0` 即 legacy 默认流），
  本条目只是把既有事实固化为契约，并让 hacc 的 `<<<>>>`（`stream = NULL`）语义有据可依。
  `cool_types.h` 的句柄注释同步补明语义。

- **`cuda_backend.cpp` 按职责拆分**（994 行 → 4 个模块）：原文件同时装着结果码映射、两个
  资源注册表、上下文生命周期和 40 个 BAL 适配函数。现拆为：
  `cuda_result_map`（`CUresult` → `coolError_t`）、`cuda_registry`（内存与句柄登记）、
  `cuda_context`（上下文与生命周期）、`cuda_backend`（BAL 适配层 + 后端描述符）。
  纯结构调整、零行为变化。命名同时澄清：`cuResultToCool` → `cuda::resultToError`、
  `registry()` → `cuda::allocations()`、`handleRegistry()` → `cuda::handles()`。
- **`lastError` 现在在成功调用后也刷新（对齐 cudart）**：`error.hpp` 一直声明「每次 API
  调用成功/失败都会刷新 TLS」，但实现只在失败路径写入，成功路径直接返回 `coolSuccess`。
  于是「失败 → 紧接着成功 → `coolPeekAtLastError()`」仍返回**上一次**的错误码，容易把
  调用方带偏。现由 `COOL_API_ENTER` 进入时先把 TLS 置为 `coolSuccess`（成功时就是这个
  值），失败再覆盖具体错误码。
  **可观测行为变更**：成功调用后 `coolPeekAtLastError()` 返回 `coolSuccess`，此前返回上一次
  的错误。`coolGetLastError` 的「读取即清空」语义不变；`coolGetErrorName/String` 这类纯
  查询接口仍不改变 TLS。由 `test_api_dispatch` 第 7 组断言锁住。
- **生命周期状态机下移到后端层**：新增 `src/backend/bootstrap.{hpp,cpp}`，负责选定后端、
  线程安全的一次性初始化与进程退出时的 finalize。`core/lifecycle.cpp` 从 84 行降到 30 行，
  只剩对 bootstrap 的转发，R 层保持无状态（符合 AGENTS.md「宿主并发原语只允许出现在
  后端层」）。
- **A 层转发样板收敛为一个宏**：44 个 `extern "C"` 中 35 个都在重复同一段「懒初始化 →
  取后端 → 检查空指针 → 检查能力位」的 6 行代码（约 210 行）。新增
  `COOL_API_ENTER(member)`（`core/lifecycle.hpp`），每个转发函数降到 2 行。
  `src/api/` 从 666 行降到 410 行。
  这不仅是行数问题：35 份手写检查里漏写一处就是静默的空指针解引用，收敛后「检查能力位」
  只有一个实现，新增 API 也不会忘。
- **库产物重命名**，对齐 CUDA（`libcudart_static.a` / `libcudart.so`）：
  `libcoolruntime.a` → **`libcoolrt_static.a`**（静态带 `_static` 后缀），并新增 **`libcoolrt.so`**。
  CMake target 名仍为 `coolruntime`（共享库为 `coolruntime_shared`，别名
  `COOLRuntime::coolruntime` / `COOLRuntime::coolruntime_shared`），测试与 samples 无需改动。
  下游 hacc 的默认链接路径已同步为 `libcoolrt_static.a`。

### Fixed

- **驱动错误码映射从 7 项扩到 15 项**：`cuda::resultToError` 此前只认识 7 个 `CUresult`，
  其余一律收敛成 `coolErrorUnknown`，用户只看到 `"unknown error"`。CUDA 13.4 共定义 105 个
  驱动错误码，这意味着除分配失败外的多数失败都拿不到有意义的分类。
  现补齐语义明确等价的：`NOT_INITIALIZED` / `DEINITIALIZED` → `coolErrorNotReady`、
  `NO_DEVICE` → `coolErrorInvalidDevice`、`INVALID_CONTEXT` / `INVALID_IMAGE` /
  `INVALID_SOURCE` / `INVALID_PTX` / `FILE_NOT_FOUND` → `coolErrorInvalidValue`。
  **有意不映射**内核运行期错误（`LAUNCH_FAILED` / `ILLEGAL_ADDRESS` / `LAUNCH_TIMEOUT`
  等）：把它们报成 `coolErrorInvalidValue` 会把「kernel 跑炸了」误导成「参数传错了」，
  比报 unknown 更糟。这条决策由 `test_result_map` 的反向断言锁住。
  实现同时从 switch 改为查表（16 个 case 的 switch 圈复杂度 16，撞上项目 ccn 门限 10；
  查表后 CCN = 2，加映射只需加一行数据）。
- **首次并发调用会重复初始化后端（重复 `cuCtxCreate` 泄漏 context）**：懒初始化原先在
  `core/lifecycle.cpp` 用函数局部静态量 + 普通 `bool` 实现，读写都没有同步。多个线程首次
  并发调用任意 `cool*` API 时都会读到「未初始化」，各自执行 `backend->init()`；在 cuda
  后端下就是两次 `cuCtxCreate` —— 第一个 context 被覆盖且永不销毁，`std::atexit` 也可能
  被重复注册。CUDA 运行时 API 本身是线程安全的，用户从线程池里调 `coolMalloc` 是正常
  用法，所以这不是理论风险。
  现在由 `bootstrap` 用 `std::mutex` 串行化「初始化 / 复位 / atexit 清理」，
  `std::atomic<bool>` 做无锁快路径（`release`/`acquire` 配对保证状态字段可见），
  避免每次 API 调用都进临界区。
  新增 `test_lifecycle_concurrency`：8 线程同时冲进 `coolMalloc`，假后端 `init` 内
  sleep 50 ms 放大竞态窗口，断言 init 恰好一次。**该测试已实测有效**——临时去掉锁后
  它会失败（`g_initCalls == 1` 不成立）。
- **初始化状态从两处收敛为一处**：此前 `core/lifecycle` 的 `LifecycleState` 与 cuda 后端的
  `g_initialized` 各记一份「是否已初始化」，且互不通知 —— `cudaFinalize()` 只清后端那份，
  于是 atexit 之后运行时仍声称「初始化成功」却没有后端（调用返回 `coolErrorNotReady`）。
  现在以 `bootstrap` 的状态为唯一权威，cuda 后端改用 `g_ctx != nullptr` 表示就绪，
  `g_initialized` 已删除。
- **4 处错误码不符合 BAL 契约**：`coolFree` / `coolMemcpy` / `coolMemset` / `coolFreeHost`
  在「后端能力位为 `NULL`」时返回 `coolErrorNotReady`，而 `cool_backend.h` 明确约定
  「未实现该能力则返回 `coolErrorNotSupported`」（其余 API 也都这么做的）。
  现把「后端未就绪」与「能力未实现」拆成两个判断，分别返回正确错误码。
  由新增的 `test_api_dispatch` 锁住。

### Removed

- 移除 `cpu` 软件后端及其自动回退逻辑；后端不可用/初始化失败时直接返回错误码。
- 移除 `null` 测试后端与 `COOL_BACKEND` 后端选择逻辑；CUDA Driver API（`cuda.h` + `libcuda`）成为唯一后端与构建硬依赖。
