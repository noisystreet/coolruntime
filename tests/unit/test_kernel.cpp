// SPDX-FileCopyrightText: 2026 CoolRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * 模块 / 内核测试：在 cuda 后端下运行（需要可用的 NVIDIA 驱动）。
 *
 * 映像以 PTX 文本内嵌，由驱动在 cuModuleLoadData 时 JIT 编译，故无需 nvcc/ptxas。
 * .target 依据设备 compute capability 生成，避免对具体架构硬编码；当驱动不支持
 * 运行时 JIT 该 PTX、或缺少可用设备时，测试跳过（返回 0），避免不稳定。
 */
#include <cstdio>

#include "cool/cool_runtime_api.h"

#include "support/check.hpp"

namespace {

/* 依据设备 compute capability 生成最小 PTX：一个空内核与一个写值内核。 */
void buildPtx(char* buf, std::size_t bufSize, int major, int minor) {
    std::snprintf(buf, bufSize,
                  ".version 8.0\n"
                  ".target sm_%d%d\n"
                  ".address_size 64\n"
                  ".visible .entry cool_noop()\n"
                  "{\n"
                  "    ret;\n"
                  "}\n"
                  ".visible .entry cool_fill_one(.param .u64 p0)\n"
                  "{\n"
                  "    .reg .b64 %%rd<2>;\n"
                  "    .reg .b32 %%r<2>;\n"
                  "    ld.param.u64 %%rd1, [p0];\n"
                  "    mov.u32 %%r1, 42;\n"
                  "    st.global.u32 [%%rd1], %%r1;\n"
                  "    ret;\n"
                  "}\n",
                  major, minor);
}

}  // namespace

int main() {
    int count = 0;
    if (coolGetDeviceCount(&count) != coolSuccess || count <= 0) {
        std::printf("[SKIP] test_kernel (no usable CUDA device)\n");
        return 0;
    }

    coolDeviceProp prop{};
    if (coolGetDeviceProperties(&prop, 0) != coolSuccess) {
        std::printf("[SKIP] test_kernel (device properties unavailable)\n");
        return 0;
    }

    char ptx[1024];
    buildPtx(ptx, sizeof(ptx), prop.major, prop.minor);

    const dim3 grid = {1, 1, 1};
    const dim3 block = {1, 1, 1};

    /* 非法参数。 */
    coolModule_t module = nullptr;
    COOL_CHECK_EQ(coolModuleLoadData(nullptr, ptx), coolErrorInvalidValue);
    COOL_CHECK_EQ(coolModuleLoadData(&module, nullptr), coolErrorInvalidValue);

    /* 载入模块（驱动对 PTX 做 JIT）。 */
    const coolError_t loadErr = coolModuleLoadData(&module, ptx);
    if (loadErr != coolSuccess) {
        std::printf("[SKIP] test_kernel (PTX JIT unavailable, err=%d)\n",
                    static_cast<int>(loadErr));
        return 0;
    }
    COOL_CHECK(module != nullptr);

    /* 取内核函数。 */
    coolKernel_t noop = nullptr;
    COOL_CHECK_EQ(coolModuleGetFunction(&noop, module, "cool_noop"), coolSuccess);
    COOL_CHECK(noop != nullptr);

    coolKernel_t fill = nullptr;
    COOL_CHECK_EQ(coolModuleGetFunction(&fill, module, "cool_fill_one"), coolSuccess);
    COOL_CHECK(fill != nullptr);

    /* 未知名应失败。 */
    coolKernel_t bogus = nullptr;
    COOL_CHECK(coolModuleGetFunction(&bogus, module, "does_not_exist") != coolSuccess);

    /* getFunction 非法参数。 */
    COOL_CHECK_EQ(coolModuleGetFunction(nullptr, module, "cool_noop"), coolErrorInvalidValue);
    COOL_CHECK_EQ(coolModuleGetFunction(&noop, nullptr, "cool_noop"), coolErrorInvalidValue);
    COOL_CHECK_EQ(coolModuleGetFunction(&noop, module, nullptr), coolErrorInvalidValue);

    /* 启动空内核（无参数、默认流）。 */
    COOL_CHECK_EQ(coolLaunchKernel(noop, grid, block, nullptr, 0, nullptr), coolSuccess);
    COOL_CHECK_EQ(coolDeviceSynchronize(), coolSuccess);

    /* 启动写值内核，并校验写入结果。 */
    void* devPtr = nullptr;
    COOL_CHECK_EQ(coolMalloc(&devPtr, sizeof(unsigned int)), coolSuccess);
    COOL_CHECK_EQ(coolMemset(devPtr, 0, sizeof(unsigned int)), coolSuccess);

    void* args[] = {&devPtr};
    COOL_CHECK_EQ(coolLaunchKernel(fill, grid, block, args, 0, nullptr), coolSuccess);
    COOL_CHECK_EQ(coolDeviceSynchronize(), coolSuccess);

    unsigned int host = 0;
    COOL_CHECK_EQ(coolMemcpy(&host, devPtr, sizeof(unsigned int), coolMemcpyDeviceToHost),
                  coolSuccess);
    COOL_CHECK_EQ(host, 42u);

    /* 占用率查询。 */
    int numBlocks = 0;
    COOL_CHECK_EQ(coolOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, noop, 1, 0),
                  coolSuccess);
    COOL_CHECK(numBlocks > 0);

    int minGrid = 0;
    int blockSize = 0;
    COOL_CHECK_EQ(coolOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, noop, 0, 0),
                  coolSuccess);
    COOL_CHECK(blockSize > 0);
    COOL_CHECK(minGrid > 0);

    /* 占用率 / 启动的非法参数。 */
    COOL_CHECK_EQ(coolOccupancyMaxActiveBlocksPerMultiprocessor(nullptr, noop, 1, 0),
                  coolErrorInvalidValue);
    COOL_CHECK_EQ(coolOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, nullptr, 1, 0),
                  coolErrorInvalidValue);
    COOL_CHECK_EQ(coolOccupancyMaxPotentialBlockSize(nullptr, &blockSize, noop, 0, 0),
                  coolErrorInvalidValue);
    COOL_CHECK_EQ(coolOccupancyMaxPotentialBlockSize(&minGrid, nullptr, noop, 0, 0),
                  coolErrorInvalidValue);
    COOL_CHECK_EQ(coolOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, nullptr, 0, 0),
                  coolErrorInvalidValue);
    COOL_CHECK_EQ(coolLaunchKernel(nullptr, grid, block, nullptr, 0, nullptr),
                  coolErrorInvalidValue);

    /* 卸载模块后，其函数句柄级联失效。 */
    COOL_CHECK_EQ(coolModuleUnload(module), coolSuccess);
    COOL_CHECK_EQ(coolModuleUnload(module), coolErrorInvalidValue); /* 重复卸载 */
    COOL_CHECK_EQ(coolLaunchKernel(noop, grid, block, nullptr, 0, nullptr), coolErrorInvalidValue);
    COOL_CHECK_EQ(coolModuleUnload(nullptr), coolErrorInvalidValue);

    COOL_CHECK_EQ(coolFree(devPtr), coolSuccess);

    return cooltest::summary("test_kernel");
}
