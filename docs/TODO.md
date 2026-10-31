# TODO

待办与未决项。按主题分组，`[ ]` 未做 / `[x]` 已完成 / `[~]` 进行中。

跨项目（编译器等）的配对项会标注对端仓库，coolrt 侧只负责声明与运行时。

---

## A. 设备端内建（`include/cool/device_builtins.h`）

> 现状：`__syncthreads`、`__threadfence{,_block,_system}`、6 种原子、5 条 wmma 内建（出参 ABI）、
> 16 个 f32 数学内建、`__launch_bounds__` 空宏。warp 原语 / `__ldg` / 位操作 / `clock` 尚未提供。
> 内建声明本身是纯头文件改动（不改行为），但每项都需要对端 `hacc` 的 `coolicc` 同步加映射，
> 因此按「声明先行、映射跟上」推进。
>
> **详细方案与验收口径见 [runtime-api-plan.md](runtime-api-plan.md)**：warp 原语 = R4、
> 位操作 / `__ldg` / 向量补全 = R5、`printf`·`assert` = R6、动态共享内存 = R7。
> 本节只记**状态**，不复制方案（两处写同一件事必然漂移）。

### A.1 原子补完 —— 性价比最高

- [x] `atomicAnd` / `atomicOr` / `atomicXor`（`int*` + `unsigned int*` 重载）—— P5.1 完成。
      hacc 侧已映射到 `atomicrmw and/or/xor` → `atom.global.{and,or,xor}.b32`，真机竞争验证通过。
      **原来的边界已消除**：hacc 的表达式翻译当初不支持 `<<` `>>` `&` `|` `^` `~`，
      只能传字面量或已算好的变量，写不出 `atomicOr(&flag, 1u << tid)`。
      该缺口已在 hacc TODO B9 中补完（183/183 通过），现在这类写法可以正常编译。
      复合赋值（`|=` 等）也已在其 B9-2 中补齐（191/191），故 `atomicOr(&f, 1u << tid)`
      与 `flags |= bit` 都能正常编译。
- [ ] 评估 `atomicInc` / `atomicDec`：CUDA 语义是 `old >= val ? 0 : old+1` 的环形计数，
      **无法用 `atomicrmw` 表达**，需要 `llvm.nvvm.atomic.load.inc.32` / `.dec.32`（调研结论见 hacc 的 `docs/m4-subset-s1-plan.md` §3.5）。

### A.2 数学函数 —— 分两档，不能一锅端

**第一批 · 零依赖**（`llvm.*` 标准 intrinsic，NVPTX 直接生成 PTX 指令）—— **已随 hacc S2 T3 落地**

- [x] `sqrtf` / `rsqrtf` / `fabsf` / `fminf` / `fmaxf` / `floorf` / `ceilf` / `truncf`
      —— 声明在本头 `__COOLCC__` 段，由 hacc `try_translate_math_call` 映射到单条 PTX 指令
      （`sqrt.rn` / `rsqrt.approx` / `abs` / `min` / `max` / `cvt.rmi·rpi·rzi`）。
- [x] `fmaf` 的精确档 —— 按 CUDA 命名收录为 **`__fmaf_rn`**（`fma.rn.f32`），
      未提供无 `_rn` 后缀的 `fmaf` 别名。
- [ ] `roundf`（`cvt.rni.f32`）与 `fmaf` 别名 —— 属一行映射的补遗项，尚未提供。

**第二批 · 需要决策**（`llvm.sin.f32` 等在 NVPTX **不会**直接落到 PTX，后端会生成对
libdevice `__nv_sinf` 的调用，需链接 `libdevice.10.bc`）

- [ ] `sinf` / `cosf` / `expf` / `logf` / `powf`（**精确档**）—— 仍需决策：接 libdevice
      （精度 IEEE，成本高）或维持现状。现状是 hacc **明确报错**并建议改用近似内建，不静默降级。
- [x] 若走近似：**必须用 CUDA 自己的快速内建命名** —— 已按此原则落地：只提供
      `__expf` / `__logf` / `__sinf` / `__cosf` / `__fdividef` / `rsqrtf`，
      未把 `sinf` 等精确函数悄悄换成 `.approx` 实现。

### A.3 warp 原语

> **前置已解除**：hacc 的位运算符（`<<` `>>` `&` `|` `^` `~`）已在 hacc TODO B9 补完，
> warp 代码的典型写法（`threadIdx.x & 0x1F`、`srcLane ^ 1`）现在能正常编译。

- [ ] 先 spike：最小 `.cu`（仅 `__shfl_down_sync`）跑通 EDG IL → LLVM → PTX，确认 intrinsic
      名与重载形态（`i32`/`f32`/`b32` 变体），再批量铺。
- [ ] `__shfl_sync`（idx / up / down / xor）、`__ballot_sync`、`__any_sync`、`__all_sync`
- [ ] `__syncwarp`、`__activemask`

### A.4 其他内建

- [ ] `__ldg`（只读数据缓存加载）
- [ ] 位操作：`__popc` / `__clz` / `__ffs` / `__brev`
- [ ] `clock()` / `clock64()`（内核内计时）
- [ ] `make_float3` 等向量构造函数
- [ ] 向量类型覆盖：`char4` / `uchar4` / `short2` / `ushort2` / `longlong2`

---

## B. 主机端

### B.1 异步拷贝 —— 流才真正有用（✅ 已完成）

> 有流、有事件、有 `coolMallocHost`，此前**所有拷贝都是同步的**，写不出拷贝/计算重叠。

- [x] `coolMemcpyAsync(dst, src, count, kind, stream)`
- [x] `coolMemsetAsync(ptr, value, count, stream)`
- [x] `coolHostRegister` / `coolHostUnregister`（页锁定任意主机内存）

### B.2 查询类与元数据（✅ P5.5 完成）

- [x] `coolStreamGetPriority` / `coolStreamGetFlags` → `cuStreamGetPriority` / `cuStreamGetFlags`
- [x] `coolDeviceGetStreamPriorityRange`（没有它，`coolStreamCreateWithPriority` 的参数靠猜）
      → `cuCtxGetStreamPriorityRange`。**语义易搞反：数值越小优先级越高**，
      `leastPriority` 是数值最大的一端（实测本机 0），`greatestPriority` 是最小的一端（-5）。
- [x] `coolFuncGetAttributes`（对端 hacc 的 S2 需要内核元数据）
      新增公开结构体 `coolFuncAttributes`（5 项），Driver 侧无批量接口，
      由后端循环调用 `cuFuncGetAttribute` 填充。
      与 cudart 的差异：不含 `binaryVersion` / `cacheModeCA` / cluster 字段，字段只增不改。

### B.3 接口补齐规划（P6）—— 已立项，未开工

- [ ] 按主题分档补齐缺失接口，计划见 **[runtime-api-plan.md](runtime-api-plan.md)**。
      优先级：纯主机侧（设备属性 / 错误码 → 2D·pitch 内存族 → JIT 选项）先做，
      设备侧（warp 原语 / 位操作 / printf·assert / 动态共享内存）一律带 spike 前置、跨仓成对推进。
      **三个开工前置 spike**：warp intrinsic 形态、`extern __shared__` 的 IL 形态、可变参调用与聚合返回值。

---

## C. 文档与一致性

- [x] 默认流语义定稿 → [ADR 0005](adr/0005-default-stream-semantics.md)
      （`NULL` = legacy 默认流，不引入 `COOL_STRICT_DEFAULT_STREAM`）
- [x] `README` 「已实现」表补 symbol 四件套
- [ ] 一致性对照测试框架形态（独立 target / 编译期双实现）—— ARCHITECTURE 的 Open Decision，Due P4

---

## D. 明确不做（Non-Goals）

沿用 `docs/ARCHITECTURE.md` §Non-Goals 与实施方案 P5+ 的判断：

- 分离编译 / 跨 TU 设备链接、动态并行
- 统一内存（`__managed__`）、高级统一寻址
- 多 GPU / peer / Graph / texture
- per-thread 默认流（见 [ADR 0005](adr/0005-default-stream-semantics.md)）
- 设备端 C++ 异常 / RTTI / 协程 / 虚函数 / 函数指针
