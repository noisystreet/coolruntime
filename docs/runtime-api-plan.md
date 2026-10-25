# Runtime 接口补齐规划（P6）

> 状态：**规划中**（2026-10 立项，未开工）。本文只定义「补什么、按什么顺序、怎么验收」，不替代
> [TODO.md](TODO.md)（活清单）与 [ARCHITECTURE.md](ARCHITECTURE.md)（范围与 Non-Goals）。
> 清点依据：`include/cool/*.h` 四个头文件 + `src/api/` 七个实现单元的**只读清点**（2026-10）。
> **实现细节未逐行核对**，凡依赖实现内部行为的判断都在文中标注「未核对」。
> 设备侧每一项都需 hacc 的 `coolicc` 同步加映射，遵循「跨仓成对推进」纪律（见 §6）。

---

## 1. 现状清点

### 1.1 主机 API：52 个

| 分组 | 个数 | 已覆盖 |
| --- | --- | --- |
| 版本 | 2 | `coolRuntimeGetVersion` / `coolDriverGetVersion` |
| 错误 | 4 | `GetErrorString` / `GetErrorName` / `GetLastError` / `PeekAtLastError` |
| 设备 | 7 | `GetDeviceCount` / `SetDevice` / `GetDevice` / `GetDeviceProperties` / `DeviceSynchronize` / `DeviceReset` / `DeviceGetStreamPriorityRange` |
| 内存 | 15 | `Malloc` / `Free` / `Memcpy` / `Memset` / `MemcpyAsync` / `MemsetAsync` / `MemGetInfo` / `MallocHost` / `FreeHost` / `HostRegister` / `HostUnregister` / `MemcpyToSymbol` / `MemcpyFromSymbol` / `GetSymbolAddress` / `GetSymbolSize` |
| 流 | 9 | `Create` / `CreateWithFlags` / `CreateWithPriority` / `Destroy` / `Synchronize` / `Query` / `WaitEvent` / `GetPriority` / `GetFlags` |
| 事件 | 7 | `Create` / `CreateWithFlags` / `Destroy` / `Record` / `Synchronize` / `Query` / `ElapsedTime` |
| 模块 / 内核 | 8 | `ModuleLoadData` / `ModuleLoadFromFile` / `ModuleGetFunction` / `ModuleUnload` / `LaunchKernel` / `OccupancyMaxActiveBlocksPerMultiprocessor` / `OccupancyMaxPotentialBlockSize` / `FuncGetAttributes` |

### 1.2 设备内建

| 类别 | 现状 |
| --- | --- |
| 限定符宏 | `__global__` / `__device__` / `__host__` / `__shared__` / `__constant__` / `__restrict__` / `__launch_bounds__`（空宏） |
| 内建变量 | `threadIdx` / `blockIdx` / `blockDim` / `gridDim` / `warpSize` |
| 同步 | `__syncthreads`；`__threadfence{,_block,_system}` |
| 原子 | 20 个重载（`Add` / `Sub` / `Exch` / `Min` / `Max` / `CAS` / `And` / `Or` / `Xor`，int / uint / float 交集） |
| wmma | 5 条（`load_a_row` / `load_b_col` / `load_c_row` / `mma` / `store_d_row`） |
| 数学 | 16 个 f32（精确档 + `__` 近似档，精度契约见头文件注释） |
| 向量类型 | `uint3` / `int3` / `float2` / `int2` / `uint2` / `double2` / `float3` / `float4` / `int4` / `uint4` / `double4` |

**一句话**：一维数组程序、静态共享内存、块级同步、单指令数学、Tensor Core MVP 都能写；
**二维/pitched 内存、warp 协作、设备内调试、设备属性细查**都不能写。

---

## 2. 排序原则

1. **纯主机侧优先**：不牵动 hacc 的项先做，收益即时、风险为零。
2. **「声明先行」≠「能力可用」**：设备侧声明是纯头文件改动，但用户真正需要的是 `coolicc` 能映射；
   没做映射就声明，等于把「编不过」变成「链接期或运行期怪错」。故此类项一律**带 spike 前置**。
3. **可诊断性 > 功能数量**：能看到「启动失败是因为 blockDim 超限」比多十个内建更有价值。

---

## 3. 子项

### R1 设备属性与限制（**优先**，纯主机侧）

**问题**：`coolGetDeviceProperties` 只填 8 个字段（`coolDeviceProp`），而 `cudaDeviceProp` 有上百个；
且**没有** `coolDeviceGetAttribute`。真实代码里 `cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlock, 0)`
是查上限的标准姿势，移植必炸。

**要做**

1. `coolError_t coolDeviceGetAttribute(int* value, coolDeviceAttribute attr, int device)`，
   枚举先覆盖实际有用的子集（Driver 侧 `CU_DEVICE_ATTRIBUTE_*` 一一对应）：
   `MaxThreadsPerBlock` / `MaxBlockDimX·Y·Z` / `MaxGridDimX` / `MaxSharedMemoryPerBlock` /
   `MaxSharedMemoryPerMultiprocessor` / `MaxRegistersPerBlock` / `WarpSize` / `MultiProcessorCount` /
   `ComputeCapabilityMajor·Minor` / `MaxThreadsPerMultiProcessor` / `ClockRate` / `MemoryClockRate` /
   `GlobalMemoryBusWidth` / `L2CacheSize` / `ConcurrentKernels` / `AsyncEngineCount` / `UnifiedAddressing`。
   枚举值**按 CUDA 的 `cudaDeviceAttr` 编号**（便于源码对照），未实现的 attr 返回 `coolErrorNotSupported`。
2. `coolDeviceProp` 补字段：`sharedMemPerBlock` / `sharedMemPerMultiprocessor` / `regsPerBlock` /
   `maxThreadsDim[3]` / `clockRate` / `memoryClockRate` / `memoryBusWidth` / `l2CacheSize` / `unifiedAddressing`。
   **只增不改**（结构体尾部追加，ABI 风险见 §7 第 3 条）。

**依赖**：无（后端已有 `cuDeviceGetAttribute` 可直接转发）。
**验收**：unit 测试对比 `coolDeviceGetAttribute` 与 `coolGetDeviceProperties` 的同一属性取值一致；
未支持的 attr 返回 `coolErrorNotSupported` 而非 0；无设备环境下按现有三态探针走 SKIP。
**风险**：低。最大风险是枚举编号抄错 → 用「属性 ↔ 枚举值 ↔ 字段名」三者交叉断言。

### R2 错误码补全（**优先**，纯主机侧）

**问题**：`coolError_t` 只有 11 个。blockDim 超限、非法访存、启动失败、显存不足的「原因」无法区分，
统一落到 `coolErrorInvalidValue` / `coolErrorUnknown`，排障成本高。

**要做**：按 cudart 数值补齐高频项 —— `InvalidConfiguration`(9)、`InvalidPitchValue`(12)、
`InvalidSymbol`(13)、`InvalidDevicePointer`(17)、`InvalidResourceHandle`(400，已有) / `InvalidMemcpyDirection`(21)、
`InvalidValue`(1，已有) / `InvalidDeviceFunction`(98) / `NoDevice`(100) / `InvalidDevice`(10，已有) /
`DeviceNotLicensed`(略) / `InvalidKernelImage`(200) / `InvalidKernelArgs` / `LaunchFailure`(719) /
`LaunchOutOfResources`(701) / `LaunchTimeout` / `LaunchIncompatibleTexturing` /
`IllegalAddress`(700) / `IllegalMemoryAccess`(700) / `MisalignedAddress`(716) / `UnmapFailed` /
`InvalidFilterSetting` / `StubLibraryNotFound`(略) / `SharedObjectSymbolNotFound` /
**`InvalidSource`(1000)** / `FileNotFound`(301) / `UnsupportedLimit` / `DuplicateVariableName` 等。

同时补 `coolGetErrorString` / `coolGetErrorName` 的映射（两者都依赖错误码到字符串的表）。
`resultToError`（后端 CUresult → coolError_t）同步扩充。

**依赖**：无。**验收**：新错误码有 unit 测试，覆盖「CUresult → coolError_t」映射表逐项；
错误码数值与 cudart 对齐（可对照 `driver_types.h`，属公开常量，不涉版权）。
**风险**：低。注意**不要把 ABI 现有数值改掉**（只增不改）。

### R3 2D / pitch 内存族（**收益最大，成本也最大**）

**问题**：现在只有一维拷贝。图像、矩阵、带行 padding 的缓冲**完全写不了**。这是「只能跑一维数组程序」的根因。

**要做（主机侧）**

1. `coolMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height)`
2. `coolMalloc2D` / `coolMalloc3D`（可选：带 extent 的 `coolPitchedPtr` 一族，见 cudart 的 `cudaExtent` / `cudaPitchedPtr`）
3. `coolMemcpy2D` / `coolMemcpy2DAsync`（`coolPitchedPtr` 或 `spitch/dpitch` 形式）
4. `coolMemset2D` / `coolMemset2DAsync`
5. `coolMemcpy3D`（若做 `coolPitchedPtr` 则顺带）
6. 类型定义：`coolExtent` / `coolPos` / `coolPitchedPtr`（放 `cool_types.h`，C 兼容）

**hacc 侧影响（关键）**：`coolMallocPitch` 返回的 pitch 通常 **> width**。设备侧取元素需
`row = idx / width; col = idx % width; data[row * pitch/4 + col]`——**用户的 kernel 代码需要自己按 pitch 索引**，
这与 CUDA 一致，不需要编译器改动。但若要在 host 桩里自动处理 `<<<>>>` 之后的多维拷贝，
则不在本期（见 §4）。

**验收**：e2e 用 `width=13, height=7`（**故意不是 2 的幂**，保证 pitch 真的 padding），
验证 `pitch >= width*4`、双向拷贝逐元素一致；`coolMemcpy2D` 带 span 的子区域拷贝也要测。
**反向验证**：故意用 `width*4` 当 pitch 的索引写法必须算出错值（证明用例真能红）。

**风险**：中。主要风险是 `pitch` 语义（字节 vs 元素）与 offset 单位错误；用非常规宽度用例兜住。

### R4 warp 原语（跨仓，先 spike）

**问题**：reduce / scan / 排序 / warp 内广播全部写不出。TODO A.3 已列。

**要做（coolrt 声明，仅 `__COOLCC__` 段）**

`__shfl_sync(mask, var, srcLane)` / `__shfl_up_sync` / `__shfl_down_sync` / `__shfl_xor_sync` /
`__ballot_sync` / `__any_sync` / `__all_sync` / `__syncwarp` / `__activemask`。

**形态问题（必须先 spike）**：CUDA 里 `__shfl_*` 是按类型重载的（`int` / `unsigned` / `float` /
`double` / `long long`），而 LLVM 侧是 `llvm.nvvm.shfl.sync.{idx,up,down,bfly}.{i32,f32,...}`。
需要确认：① 声明成 C 重载还是 C 宏 + `_Generic`；② hacc 的调用分派能否按实参类型选 intrinsic；
③ `mask` 参数在现代 NVVM intrinsic 里是显式操作数。

**验收**：e2e 覆盖 `shfl_down` 做 warp reduce（32 个 lane 求和的精确值）、`ballot` 做前缀判定、
`__syncwarp` 无实际数值效果但必须编过；**至少一条**要证明「传错 srcLane 会得到错值」。
**反向验证**：把 `__shfl_down_sync` 换成普通数组访问，结果必须变红。

**风险**：中高。真正的工作量在 hacc 的重载分派，不在声明。

### R5 位操作 + `__ldg` + 向量补全（跨仓，最便宜的一批）

**要做**

1. `__popc` / `__clz` / `__ffs` / `__brev` / `__byte_perm` / `__funnelshift_l` / `__funnelshift_r`
   （各对应 `llvm.ctpop` / `llvm.ctlz` / `llvm.cttz` / `llvm.bitreverse` / PTX `prmt` / `shf.*`）
2. `__ldg(const T*)` —— 走 `ld.global.nc`；需按类型重载
3. 向量构造 `make_float2/3/4` / `make_int2/3/4` / `make_uint3/4` / `make_double2/4`
4. 向量类型补全：`char2` / `char4` / `uchar4` / `short2` / `ushort2` / `longlong2` / `double3` / `long3`

**依赖**：hacc 需支持对应 intrinsic。`__ldg` 的 `T` 重载是唯一有形态风险的项（同 R4 的②）。
**验收**：每项一个 e2e（`__popc(0xF0F0F0F0) == 16` 这类可手算的值）；`make_float3` 用聚合返回，
需验证 hacc 的**聚合返回**路径（wmma 是出参 ABI，这里要求值返回 —— **可能是坑，先 spike**）。
**风险**：中。`make_*` 的值返回聚合如果 hacc 不支持，要退化为「构造后字段赋值」，届时改文档而非硬上。

### R6 设备端 printf / assert（跨仓，成本最高）

**问题**：kernel 内部无法输出、无法断言。当前调试只能靠 `out[]` 回传猜。与本项目
「宁可编不过、不要静默错值」的口径直接相关：**没有 assert，就没有运行期不变量检查**。

**要做**

1. `int printf(const char* fmt, ...)`（设备版）+ `__assertfail` / `assert` 宏
2. `coolDeviceProp` 无关，但需要 `coolDeviceSetLimit(coolLimitPrintfFifoSize, n)` 配套（见 R1/R2 的 limit 项）

**可行性 spike（必须先做）**：确认 EDG IL 里**可变参调用**保留了哪些信息（`enk_operation` 的
call 表达式、参数个数是否可静态取到）；PTX 侧需要 `vprintf`（NVPTX 有该指令）还是
`llvm.nvvm.printf.*`。**这一项的可行性与 hacc 是否支持可变参调用强耦合，未核对。**
若 EDG 侧拿不到可变参实参列表 → R6 直接降级为 Non-Goal，不要硬做。

**验收**：e2e 在 kernel 里打印一个确定字符串 + 一个整数，并在 host 端 `coolDeviceSynchronize`
之后能读到输出（注意：**输出时机由 FIFO 刷写决定**，测试须允许「sync 后才可见」）。
**风险**：高。可拆成「printf 只支持 `%d`/`%u`/`%f`/`%s` 的受限子集」降低难度。

### R7 shared memory 配套

**问题**：目前只有**静态** `__shared__`（hacc M3.1：文件级 `addrspace(3)` 全局）。
动态共享内存（`extern __shared__ float s[]` + 启动时 `sharedMem` 参数）在 hacc 侧
**未见处理路径**（`src/` 内只有 `kCoolSharedAttributeName` 的静态变量分支，未核对 extern 形式）。

**要做**

1. coolrt：无新 API（`coolLaunchKernel` 已有 `sharedMem` 参数）；但需在文档中明确
   **合法上限来自 `coolDeviceGetAttribute(MaxSharedMemoryPerBlock)`**（依赖 R1）
2. `__syncthreads_count` / `__syncthreads_and` / `__syncthreads_or`（块级统计，PTX `bar.red.*`）
3. `__cvta_generic_to_shared`（指针跨地址空间转换；hacc 的地址空间按 `cool_address_space(n)` 显式标注，
   缺这个原语时 generic → shared 只能靠属性传播）

**前置 spike**：`extern __shared__ T name[]` 在 EDG IL 里长什么样（`a_variable` 的
`is_extern` / 长度未知如何表达），以及 hacc 能否按启动参数 `sharedMem` 生成 `addrspace(3)` 的
`extern` 全局 + 正确对齐。**未核对，不做完不要承诺。**

**验收**：动态共享内存 e2e（`extern __shared__ int s[]` 反转数组），`sharedMem` 传
`n*sizeof(int)`，真机数值正确；`bar.red` 三条各一个断言。
**风险**：中高，主要在 hacc 侧。

### R8 流 / 事件补遗（纯主机侧，按需）

- `coolStreamAddCallback`（host 回调）、`coolStreamWaitValue32` / `coolStreamWriteValue32`
  （设备侧轮询同步，需 `cuStreamWaitValue32`）、`coolEventCreateWithFlags(Interprocess)` 的实际支持
- `coolStreamGetCaptureInfo` / graph 系列：**不做**（Non-Goal）

**验收**：callback 用例验证「回调在流内工作完成后触发、且晚于此前排入的拷贝」；
waitValue32 需一个 kernel 侧写标志 + host 侧等待的 e2e。
**风险**：中（waitValue32 依赖驱动能力，可返回 `coolErrorNotSupported`）。

### R9 JIT 选项（纯主机侧，低优先）

`coolModuleLoadDataEx(module, image, numOptions, options, optionValues)`：让用户指定
`arch` / `optLevel` / `lineinfo`。现状是 `coolModuleLoadData` 完全不可控，调试/调优都受限。

**验收**：用 `lineinfo` 选项加载后 `cuModuleGetFunction` 成功，且 ptxas 输出的行号信息可见（或至少不报错）。
**风险**：低。

---

## 4. Non-Goals（本期明确不做）

沿用 [ARCHITECTURE.md](ARCHITECTURE.md) §Non-Goals，并补充本规划的判定：

- 图 / Graph、纹理 / surface、外部内存与信号量、内存池、peer 访问、驱动入口点 —— 架构级 Non-Goal
- 统一内存（`__managed__`）/ 高级统一寻址 / `coolMallocManaged` / `coolMallocAsync` —— 不改决策
- 多 GPU、动态并行、协作组（cooperative groups）与 grid 级同步
- `<<<>>>` 之后的多维自动拷贝（host 桩替用户处理 pitch）—— 与 CUDA 语义不同，易生歧义
- 设备端 C++ 异常 / RTTI / 协程 / 虚函数 / 函数指针
- **精确档 libdevice 数学**：仍是独立决策项（现状 fail loud，安全）。若做，须先定「精度契约」
  并在头文件逐个标注，不能悄悄把 `sinf` 变 `.approx`

---

## 5. 开工前必须做的 spike（三个，都不长）

| # | 主题 | 要回答的问题 | 不做完的后果 |
| --- | --- | --- | --- |
| S1 | **warp intrinsic 形态** | `llvm.nvvm.shfl.sync.*` 在 LLVM 19 的准确名字与 i32/f32 变体；`mask` 是否显式操作数 | R4 的声明怎么写、hacc 怎么分派都无从下手 |
| S2 | **`extern __shared__` 的 IL 形态** | EDG 里 extern 数组变量的长度/对齐如何表达；hacc 能否生成 `addrspace(3)` extern 全局 | R7 可能整项不可做 |
| S3 | **可变参调用与聚合返回值** | EDG IL 是否保留 varargs 实参；hacc 能否翻译聚合（struct）值返回 | R6 可能与 Non-Goal 无异；R5 的 `make_*` 要改设计 |

三个 spike 都是**小探针 + 一次 IL/PTX 对照**，各自半天量级；结论写回本文 §5 与
[edg-il-reference 对应章节](../../hacc/docs/edg-il-reference.md)（hacc 侧）。

---

## 6. 排序与工作量（量级，不是精确估算）

| 阶段 | 项 | 跨仓 | 量级 |
| --- | --- | --- | --- |
| 1 | R1 设备属性 + R2 错误码 | 否 | 小（各半天～1 天） |
| 2 | R5 位操作 / 向量补全（先 S3 中「位操作」部分） | 是 | 小（每项一个映射 + 一个 e2e） |
| 3 | R3 2D / pitch 族 | 否（主机侧） | 中（1～2 天，形态多、易漏 offset） |
| 4 | R7 静态侧的 `bar.red` + R8 流补遗 | 半（bar.red 是跨仓） | 中 |
| 5 | R4 warp 原语（先 S1） | 是 | 中～大 |
| 6 | R9 JIT 选项 | 否 | 小 |
| 7 | R6 printf / assert（先 S3） | 是 | 大（或降级为 Non-Goal） |
| — | R5 的 `make_*` 聚合返回、R7 的动态共享内存 | 是 | 取决于 S2 / S3 结论 |

**纪律**：设备内建一律**跨仓成对推进**——coolrt 声明 + hacc 映射 + 真机 e2e，三件套同一轮提交；
不做「先声明、以后再说映射」，那会把编译期错误推迟成运行期怪错。

---

## 7. 验收总则

1. **每项都要有「能红」的证据**：反向验证（改坏实现或改坏期望值必须变红）。本项目历史上
   KI-1 / KI-4 / KI-5 都是「测试全绿、真机行为错」，新接口不接受「只跑通 happy path」。
2. **三态探针**：无 CUDA 设备时打印 SKIP 返回 77；**初始化失败返回 1**，不得混同（C4 已定）。
3. **ABI 只增不改**：`coolDeviceProp` / `coolFuncAttributes` / `coolError_t` 尾部追加或复用空号，
   不改既有字段顺序与数值。若必须破坏性调整，需单独 PR + CHANGELOG 迁移说明。
4. **错误语义**：新接口的非法输入必须返回具体错误码（不是 `Unknown`），并在头文件注释里写清。
5. **文档同步**：每项完成当轮更新 [TODO.md](TODO.md)、[CHANGELOG](../CHANGELOG.md)、
   README「已实现」表与测试计数。

---

## 8. 未核对项（本文的边界）

- 未逐行读 `src/api/*.cpp` 与 `src/backend/cuda/cuda_backend.cpp`：个别函数可能实现得比声明窄
  （例如 `coolLaunchKernel` 是否校验 blockDim 上限 —— 已知它把 `sharedMem` 直接透传给
  `cuLaunchKernel`，无上限校验）。
- `coolHostRegisterMapped` 标志存在但**没有 `coolHostGetDevicePointer`**，该标志当前是否真的可用
  未验证；若不支持，应在头文件标注或补 API（属 R1 的附带项）。
- hacc 侧 `extern __shared__`（S2）与可变参调用（S3）均未实测，相关子项的可行性以 spike 结论为准。
- `double` 设备算术的实际支持度未验证，故「double 数学全缺」只按头文件判定。
