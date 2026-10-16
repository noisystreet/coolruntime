// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * 设备符号（M3.2）测试：coolMemcpyToSymbol / coolGetSymbolAddress / coolGetSymbolSize。
 *
 * 在 cuda 后端下运行（需要可用的 NVIDIA 驱动）。映像以 PTX 文本内嵌，由驱动在
 * cuModuleLoadData 时 JIT 编译，故无需 nvcc/ptxas。PTX 中声明一个 .global 符号
 * （模拟 __constant__ 变量），用于验证符号解析与拷贝。
 *
 * 当驱动不支持运行时 JIT 该 PTX、或缺少可用设备时，测试跳过（返回 0）。
 */
#include <cstdio>
#include <cstring>

#include "cool/cool_runtime_api.h"

#include "support/check.hpp"

namespace {

/* 依据设备 compute capability 生成含一个 .global 符号的最小 PTX。 */
void buildPtx(char* buf, std::size_t bufSize, int major, int minor) {
    std::snprintf(buf, bufSize,
                  ".version 8.0\n"
                  ".target sm_%d%d\n"
                  ".address_size 64\n"
                  ".visible .global .align 4 .b8 c_pi[4];\n"
                  ".visible .entry cool_noop()\n"
                  "{\n"
                  "    ret;\n"
                  "}\n",
                  major, minor);
}

}  // namespace

int main() {
    int count = 0;
    if (coolGetDeviceCount(&count) != coolSuccess || count <= 0) {
        std::printf("[SKIP] test_symbol (no usable CUDA device)\n");
        return 0;
    }

    coolDeviceProp prop{};
    if (coolGetDeviceProperties(&prop, 0) != coolSuccess) {
        std::printf("[SKIP] test_symbol (device properties unavailable)\n");
        return 0;
    }

    char ptx[1024];
    buildPtx(ptx, sizeof(ptx), prop.major, prop.minor);

    coolModule_t module = nullptr;
    const coolError_t loadErr = coolModuleLoadData(&module, ptx);
    if (loadErr != coolSuccess) {
        std::printf("[SKIP] test_symbol (PTX JIT unavailable, err=%d)\n",
                    static_cast<int>(loadErr));
        return 0;
    }
    COOL_CHECK(module != nullptr);

    /* 构造符号描述符（hacc 生成主机桩时的等价物）。 */
    coolSymbolDesc desc{};
    desc.module = module;
    desc.symbol_name = "c_pi";
    desc.size = sizeof(float);

    /* 非法参数。 */
    COOL_CHECK_EQ(coolMemcpyToSymbol(nullptr, &desc, 4, 0, coolMemcpyHostToDevice),
                  coolErrorInvalidValue);
    COOL_CHECK_EQ(coolMemcpyToSymbol(&desc, nullptr, 4, 0, coolMemcpyHostToDevice),
                  coolErrorInvalidValue);
    COOL_CHECK_EQ(coolGetSymbolAddress(nullptr, &desc), coolErrorInvalidValue);
    COOL_CHECK_EQ(coolGetSymbolAddress(reinterpret_cast<void**>(&module), nullptr),
                  coolErrorInvalidValue);
    COOL_CHECK_EQ(coolGetSymbolSize(nullptr, &desc), coolErrorInvalidValue);

    /* 未初始化模块句柄的描述符应被拒。 */
    coolSymbolDesc badDesc{};
    badDesc.module = nullptr;
    badDesc.symbol_name = "c_pi";
    badDesc.size = sizeof(float);
    COOL_CHECK_EQ(coolGetSymbolSize(&desc.size, &badDesc), coolErrorInvalidValue);

    /* 取符号大小（PTX 声明为 4 字节）。 */
    size_t symSize = 0;
    COOL_CHECK_EQ(coolGetSymbolSize(&symSize, &desc), coolSuccess);
    COOL_CHECK_EQ(symSize, sizeof(float));

    /* 取符号地址。 */
    void* symAddr = nullptr;
    COOL_CHECK_EQ(coolGetSymbolAddress(&symAddr, &desc), coolSuccess);
    COOL_CHECK(symAddr != nullptr);

    /* 拷贝主机数据到符号，再拷回校验。 */
    const float pi = 3.14159f;
    COOL_CHECK_EQ(coolMemcpyToSymbol(&desc, &pi, sizeof(float), 0, coolMemcpyHostToDevice),
                  coolSuccess);

    float readback = 0.0f;
    COOL_CHECK_EQ(coolMemcpy(&readback, symAddr, sizeof(float), coolMemcpyDeviceToHost),
                  coolSuccess);
    COOL_CHECK(readback == pi);

    /* 越界拷贝应被拒（offset + count > 符号大小）。 */
    COOL_CHECK_EQ(coolMemcpyToSymbol(&desc, &pi, sizeof(float), 2, coolMemcpyHostToDevice),
                  coolErrorInvalidValue);
    COOL_CHECK_EQ(coolMemcpyToSymbol(&desc, &pi, 8, 0, coolMemcpyHostToDevice),
                  coolErrorInvalidValue);

    /* 不支持的拷贝方向（符号在设备侧，仅 H2D / D2D）。 */
    COOL_CHECK_EQ(coolMemcpyToSymbol(&desc, &pi, sizeof(float), 0, coolMemcpyDeviceToHost),
                  coolErrorInvalidValue);

    /* 未知名符号应失败。 */
    coolSymbolDesc missing{};
    missing.module = module;
    missing.symbol_name = "does_not_exist";
    missing.size = 4;
    COOL_CHECK(coolGetSymbolSize(&symSize, &missing) != coolSuccess);

    COOL_CHECK_EQ(coolModuleUnload(module), coolSuccess);

    return cooltest::summary("test_symbol");
}
