// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * P5.5 查询类 API 测试：
 *   coolDeviceGetStreamPriorityRange / coolStreamGetPriority /
 *   coolStreamGetFlags / coolFuncGetAttributes。
 *
 * 前三项是纯查询，只需可用设备；funcGetAttributes 依赖已加载的 kernel，
 * 故沿用 test_kernel 的做法：内嵌最小 PTX，由驱动 JIT，无需 nvcc/ptxas。
 * 缺设备或 JIT 不可用时跳过（返回 0），与其它 e2e 用例的处理一致。
 */
#include <cstdio>

#include "cool/cool_runtime_api.h"

#include "support/check.hpp"

namespace {

/* 依据设备 compute capability 生成最小 PTX：只有一个空内核。 */
void buildPtx(char* buf, std::size_t bufSize, int major, int minor) {
    std::snprintf(buf, bufSize,
                  ".version 8.0\n"
                  ".target sm_%d%d\n"
                  ".address_size 64\n"
                  ".visible .entry query_noop()\n"
                  "{\n"
                  "    ret;\n"
                  "}\n",
                  major, minor);
}

}  // namespace

int main() {
    /*
     * 1) 参数校验先于任何设备访问：nullptr 一律 InvalidValue。
     *    这一组不需要设备，故放在最前 —— 无 GPU 的机器上也应能验证边界。
     */
    int probe = 0;
    COOL_CHECK_EQ(coolDeviceGetStreamPriorityRange(nullptr, nullptr), coolErrorInvalidValue);
    COOL_CHECK_EQ(coolDeviceGetStreamPriorityRange(&probe, nullptr), coolErrorInvalidValue);
    COOL_CHECK_EQ(coolDeviceGetStreamPriorityRange(nullptr, &probe), coolErrorInvalidValue);
    COOL_CHECK_EQ(coolStreamGetPriority(nullptr, nullptr), coolErrorInvalidValue);
    COOL_CHECK_EQ(coolStreamGetFlags(nullptr, nullptr), coolErrorInvalidValue);
    COOL_CHECK_EQ(coolFuncGetAttributes(nullptr, nullptr), coolErrorInvalidValue);

    int count = 0;
    if (coolGetDeviceCount(&count) != coolSuccess || count <= 0) {
        std::printf("[SKIP] test_query (no usable CUDA device)\n");
        return 0;
    }

    /*
     * 2) 优先级范围：数值越小优先级越高，故 greatest（最高优先级）的数值 ≤ least。
     *    实测本机返回 [least=0, greatest=-5]，与 CUDA 文档的常见取值一致。
     */
    int least = 0;
    int greatest = 0;
    COOL_CHECK_EQ(coolDeviceGetStreamPriorityRange(&least, &greatest), coolSuccess);
    COOL_CHECK(greatest <= least);

    /* 3) 默认流的优先级与标志恒为 0：legacy 流既非 non-blocking，也无特殊优先级。 */
    int defPriority = -1;
    unsigned int defFlags = 0xFFFFFFFFu;
    COOL_CHECK_EQ(coolStreamGetPriority(&defPriority, nullptr), coolSuccess);
    COOL_CHECK_EQ(coolStreamGetFlags(&defFlags, nullptr), coolSuccess);
    COOL_CHECK_EQ(defPriority, 0);
    COOL_CHECK_EQ(defFlags, 0u);

    /*
     * 4) 以最高优先级 + non-blocking 建流，查回来必须一致 ——
     *    这是本次新增 API 的核心契约：创建时给的值能被原样读出。
     *    两个方向分开测（priority 走 cuStreamGetPriority，flags 走 cuStreamGetFlags）。
     */
    coolStream_t stream = nullptr;
    COOL_CHECK_EQ(coolStreamCreateWithPriority(&stream, coolStreamNonBlocking, greatest),
                  coolSuccess);
    int priority = 0;
    unsigned int flags = 0;
    COOL_CHECK_EQ(coolStreamGetPriority(&priority, stream), coolSuccess);
    COOL_CHECK_EQ(coolStreamGetFlags(&flags, stream), coolSuccess);
    COOL_CHECK_EQ(priority, greatest);
    COOL_CHECK((flags & coolStreamNonBlocking) != 0u);
    COOL_CHECK_EQ(coolStreamDestroy(stream), coolSuccess);

    /*
     * 5) 内核属性：加载最小 PTX 后读 5 项。
     *    maxThreadsPerBlock 必为正（任何内核对 ≥1 线程可启动）；
     *    其余是可为零的计数，只断言非负 —— 具体数值取决于驱动与架构，不该硬编码。
     */
    coolDeviceProp prop{};
    COOL_CHECK_EQ(coolGetDeviceProperties(&prop, 0), coolSuccess);

    char ptx[512];
    buildPtx(ptx, sizeof(ptx), prop.major, prop.minor);

    coolModule_t module = nullptr;
    if (coolModuleLoadData(&module, ptx) != coolSuccess) {
        std::printf("[SKIP] test_query (PTX JIT unavailable)\n");
        return 0;
    }

    coolKernel_t kernel = nullptr;
    const coolError_t getErr = coolModuleGetFunction(&kernel, module, "query_noop");
    COOL_CHECK_EQ(getErr, coolSuccess);

    /* 未注册的 kernel 句柄必须被拒，且失败时不得改写调用方的输出结构。 */
    coolFuncAttributes staleAttr{};
    staleAttr.maxThreadsPerBlock = 12345;
    staleAttr.numRegs = 7;
    COOL_CHECK_EQ(coolFuncGetAttributes(&staleAttr, reinterpret_cast<coolKernel_t>(0x1234)),
                  coolErrorInvalidValue);
    COOL_CHECK_EQ(staleAttr.maxThreadsPerBlock, 12345);
    COOL_CHECK_EQ(staleAttr.numRegs, 7);

    if (getErr == coolSuccess) {
        coolFuncAttributes attr{};
        COOL_CHECK_EQ(coolFuncGetAttributes(&attr, kernel), coolSuccess);
        COOL_CHECK(attr.maxThreadsPerBlock > 0);
        COOL_CHECK(attr.numRegs >= 0);
        COOL_CHECK(attr.sharedSizeBytes >= 0);
        COOL_CHECK(attr.localSizeBytes >= 0);
        COOL_CHECK(attr.constSizeBytes >= 0);
        std::printf("funcAttrs: maxThreads=%d regs=%d shared=%d local=%d const=%d\n",
                    attr.maxThreadsPerBlock, attr.numRegs, attr.sharedSizeBytes,
                    attr.localSizeBytes, attr.constSizeBytes);
    }

    COOL_CHECK_EQ(coolModuleUnload(module), coolSuccess);

    std::printf("query OK: priorityRange=[%d, %d]\n", least, greatest);
    return cooltest::summary("test_query");
}
