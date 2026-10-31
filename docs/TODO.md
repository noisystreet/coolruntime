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
- [x] `roundf` 与 `fmaf` 别名 —— **已完成，但原判断有误，值得记一笔**。
      原条目写的是「`roundf`（`cvt.rni.f32`）」：**错的**。spike 实测 NVPTX 把
      `llvm.rint` / `llvm.nearbyint` 都降到 `cvt.rni.f32.f32` —— `.rni` 是
      **rint 语义（.5 舍入到偶数）**，而 CUDA `roundf` 是 **.5 远离零**。
      照原判断实现，`roundf(2.5)` 会返回 2 而不是 3，且能编、能跑、自测全绿
      （只有 .5 边界值能发现）。实际映射为 `llvm.round.f32`，展开成
      `copysign(0.5,x)+add.rn+cvt.rzi` 加两个边界 select（约 8 条指令）。
      同时补了 `rintf`（单条 `cvt.rni`），两者在 hacc 的 `round_fma.cu` 里用
      同一组 .5 输入**互相区分**。`make check` 252/252。

**第二批 · 需要决策**（`llvm.sin.f32` 等在 NVPTX **不会**直接落到 PTX，后端会生成对
libdevice `__nv_sinf` 的调用，需链接 `libdevice.10.bc`）

- [ ] `sinf` / `cosf` / `expf` / `logf` / `powf`（**精确档**）—— 仍需决策：接 libdevice
      （精度 IEEE，成本高）或维持现状。现状是 hacc **明确报错**并建议改用近似内建，不静默降级。
- [x] 若走近似：**必须用 CUDA 自己的快速内建命名** —— 已按此原则落地：只提供
      `__expf` / `__logf` / `__sinf` / `__cosf` / `__fdividef` / `rsqrtf`，
      未把 `sinf` 等精确函数悄悄换成 `.approx` 实现。

### A.3 warp 原语

> **块级归约已单独落地**（P6 R7，不在 warp 原语之列）：`__syncthreads_count/_and/_or`
> 已随本轮完成，映射到 PTX `bar.red.{popc,and,or}`。**关键实测**：LLVM 19 没有
> `llvm.nvvm.bar.red.*` intrinsic（会被当成普通外部函数、生成未定义 call），
> 故 hacc 侧用内联汇编发射；`bar.red` 是**三个**操作数 `d, a, c`（中间是参与
> 线程数，`0` = 取块大小），少写一个只在 ptxas 阶段报 `Arguments mismatch`。
> 下面这些 warp 原语**仍未做**。

> **前置已解除**：hacc 的位运算符（`<<` `>>` `&` `|` `^` `~`）已在 hacc TODO B9 补完，
> warp 代码的典型写法（`threadIdx.x & 0x1F`、`srcLane ^ 1`）现在能正常编译。

- [ ] 先 spike：最小 `.cu`（仅 `__shfl_down_sync`）跑通 EDG IL → LLVM → PTX，确认 intrinsic
      名与重载形态（`i32`/`f32`/`b32` 变体），再批量铺。
- [ ] `__shfl_sync`（idx / up / down / xor）、`__ballot_sync`、`__any_sync`、`__all_sync`
- [ ] `__syncwarp`、`__activemask`

### A.4 其他内建

- [x] `__ldg`（只读数据缓存加载）—— P6 R5 第二批，hacc 侧已映射到 `ld.global.nc`。
      标量 4 个重载（`int` / `unsigned int` / `float` / `double`），**向量版待补**。
      两条红线（写在头注释里）：强承诺不是提示、只能传全局内存指针。
- [x] `__ldg` 的向量重载（`float4` / `int4` / `uint4`）—— 已完成。与标量走同一条
      翻译路径（load 宽度由调用表达式类型决定），hacc 侧零改动；golden 断言追加
      `ld.global.nc.v4`，证明是 128 位单次读而不是 4 条标量读。
      **顺带暴露的语言子集缺口已修**：`out4[i].x = ...` 这类**成员左值**当时 hacc 不支持
      （报「不支持的左值表达式」）；hacc C18 已完成（成员访问统一成「成员地址」，
      左值/右值共用一套），该用例现已改为成员左值写法并在 hacc 侧当回归用例。
- [x] 位操作：`__popc` / `__clz` / `__ffs` / `__brev` —— P6 R5 第一批，hacc 侧已映射到
      `popc.b32` / `clz.b32` / `brev.b32`；`__ffs` 走 `cttz + 1` 且**特判 0**（否则 `__ffs(0)`
      静默返回 33）。边界值在 hacc 的 `int_builtins.cu` 里逐个断言。
- [x] `clock()` / `clock64()`（内核内计时）—— 映射到 `%clock` / `%clock64`。返回
      `unsigned int` / `unsigned long long`（**不是** CUDA 的 `clock_t`），两者只能做差。
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

### B.3 接口补齐规划（P6）—— R1 / R2 已完成，其余未开工

- [x] **R1 设备属性 + R2 错误码** —— `coolDeviceGetAttribute` + `coolDeviceProp` 追加 10 个字段；
  `coolError_t` 11 → 31 项（`NO_DEVICE` 与内核执行期错误各自有了专码）。
  详见 [runtime-api-plan.md](runtime-api-plan.md) §R1 / §R2 与 CHANGELOG。
  **遗留决策项**：3 处历史数值与 cudart 不同（`InvalidDevice` 10 vs 101、
  `MemoryAllocation` 3 vs `InitializationError`、`HostMemory*` 711/712 vs 712/713），
  本轮刻意未改（改即破坏 ABI），现状由 `cool_types.h` 注释与 `test_error` 锚点锁住。
- [ ] 其余子项（R3–R9）按 **[runtime-api-plan.md](runtime-api-plan.md)** 分档推进。
      优先级：纯主机侧（设备属性 / 错误码 → 2D·pitch 内存族 → JIT 选项）先做，
      设备侧（warp 原语 / 位操作 / printf·assert / 动态共享内存）一律带 spike 前置、跨仓成对推进。
      **三个开工前置 spike**：warp intrinsic 形态、`extern __shared__` 的 IL 形态、可变参调用与聚合返回值。
- [ ] **`coolDeviceSetLimit`（补录：此前只在 hacc 侧被提及，本清单一直没有条目）**：
      两个 limit 各有硬需求，且都**不需要任何 spike**（纯主机侧，Driver `cuCtxSetLimit` 一一对应）：
      - `coolLimitStackSize`：调大 per-thread 调用栈。默认 1 KiB ≈ 30 层递归，**超出后是静默错值**
        （`coolDeviceSynchronize` 返 999 但结果保持初值），见 hacc `docs/known-issues.md` **KI-4**；
        hacc 的 `docs/TODO.md` C11 里写着「若要排期 coolrt 封装 `coolDeviceSetLimit`」——
        本条就是把它落到真相清单里。
      - `coolLimitPrintfFifoSize`：设备端 printf 的配套（R6 的一半前置；printf 本身仍需 S3 spike）。
      **验收**：`coolDeviceSetLimit` + `coolDeviceGetLimit` 往返一致；用 hacc 的
      `tests/golden/e2e/recursion.cu` 把栈调大后 n=50 的用例由失败转通过（该用例当前是 KI-4 的回归锚点）。

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
