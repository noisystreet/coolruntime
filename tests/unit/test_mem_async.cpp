// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * P5.4 异步拷贝与页锁定注册测试（在 cuda 后端下运行，需要可用的 NVIDIA 驱动）。
 *
 * 覆盖三层：
 *   1. 参数与句柄校验（伪造的 stream 句柄应被注册表拦截）
 *   2. 异步流水：H2D / D2H / memset 走自定义流与默认流，结果与预期一致
 *   3. coolHostRegister / coolHostUnregister 的注册状态机（重复注册、重复注销）
 *
 * 刻意**不**断言「异步真的发生了重叠」：那取决于驱动与硬件调度，无法在单元测试里
 * 稳定观测。这里只保证语义正确 —— 结果对、顺序对、错误码对。
 */
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "cool/cool_runtime_api.h"

#include "support/check.hpp"

namespace {

/* 伪造一个未在注册表中的流句柄，用于验证后端会拦截非法句柄。 */
coolStream_t bogusStream() {
    return reinterpret_cast<coolStream_t>(static_cast<std::uintptr_t>(0x1));
}

/* cuMemHostRegister 要求页对齐，故取一份页对齐缓冲。 */
void* allocatePageAligned() {
    return std::aligned_alloc(4096, 4096);
}

}  // namespace

int main() {
    /* ---------- 1. 参数与句柄校验 ---------- */
    int hostValue = 42;
    void* deviceBuf = nullptr;
    COOL_CHECK_EQ(coolMalloc(&deviceBuf, sizeof(int)), coolSuccess);

    COOL_CHECK_EQ(
        coolMemcpyAsync(nullptr, &hostValue, sizeof(int), coolMemcpyHostToDevice, nullptr),
        coolErrorInvalidValue);
    COOL_CHECK_EQ(coolMemcpyAsync(deviceBuf, nullptr, sizeof(int), coolMemcpyHostToDevice, nullptr),
                  coolErrorInvalidValue);
    COOL_CHECK_EQ(coolMemsetAsync(nullptr, 0, sizeof(int), nullptr), coolErrorInvalidValue);

    /* 非法流句柄：与同步版本一致，由句柄注册表拦截。 */
    COOL_CHECK_EQ(
        coolMemcpyAsync(deviceBuf, &hostValue, sizeof(int), coolMemcpyHostToDevice, bogusStream()),
        coolErrorInvalidValue);
    COOL_CHECK_EQ(coolMemsetAsync(deviceBuf, 0, sizeof(int), bogusStream()), coolErrorInvalidValue);

    COOL_CHECK_EQ(coolFree(deviceBuf), coolSuccess);

    /* ---------- 2. 异步流水（pinned 主机内存 ↔ 设备） ---------- */
    int* pinned = nullptr;
    COOL_CHECK_EQ(coolMallocHost(reinterpret_cast<void**>(&pinned), 4 * sizeof(int)), coolSuccess);
    COOL_CHECK(pinned != nullptr);
    for (int i = 0; i < 4; ++i) {
        pinned[i] = i * 10;
    }

    void* deviceQuad = nullptr;
    COOL_CHECK_EQ(coolMalloc(&deviceQuad, 4 * sizeof(int)), coolSuccess);

    coolStream_t stream = nullptr;
    COOL_CHECK_EQ(coolStreamCreate(&stream), coolSuccess);

    /* H2D 异步 → 同步流。 */
    COOL_CHECK_EQ(
        coolMemcpyAsync(deviceQuad, pinned, 4 * sizeof(int), coolMemcpyHostToDevice, stream),
        coolSuccess);
    COOL_CHECK_EQ(coolStreamSynchronize(stream), coolSuccess);

    /* D2H 异步取回，内容应原样返回。 */
    for (int i = 0; i < 4; ++i) {
        pinned[i] = 0;
    }
    COOL_CHECK_EQ(
        coolMemcpyAsync(pinned, deviceQuad, 4 * sizeof(int), coolMemcpyDeviceToHost, stream),
        coolSuccess);
    COOL_CHECK_EQ(coolStreamSynchronize(stream), coolSuccess);
    for (int i = 0; i < 4; ++i) {
        COOL_CHECK_EQ(pinned[i], i * 10);
    }

    /* 异步填充清零，再异步取回验证。 */
    COOL_CHECK_EQ(coolMemsetAsync(deviceQuad, 0, 4 * sizeof(int), stream), coolSuccess);
    COOL_CHECK_EQ(
        coolMemcpyAsync(pinned, deviceQuad, 4 * sizeof(int), coolMemcpyDeviceToHost, stream),
        coolSuccess);
    COOL_CHECK_EQ(coolStreamSynchronize(stream), coolSuccess);
    for (int i = 0; i < 4; ++i) {
        COOL_CHECK_EQ(pinned[i], 0);
    }

    /* 默认流（NULL）路径同样可用（ADR 0005：NULL = legacy 默认流）。 */
    for (int i = 0; i < 4; ++i) {
        pinned[i] = i + 1;
    }
    COOL_CHECK_EQ(
        coolMemcpyAsync(deviceQuad, pinned, 4 * sizeof(int), coolMemcpyHostToDevice, nullptr),
        coolSuccess);
    COOL_CHECK_EQ(coolStreamSynchronize(nullptr), coolSuccess);

    COOL_CHECK_EQ(coolStreamDestroy(stream), coolSuccess);
    COOL_CHECK_EQ(coolFreeHost(pinned), coolSuccess);
    COOL_CHECK_EQ(coolFree(deviceQuad), coolSuccess);

    /* ---------- 3. 页锁定注册状态机 ---------- */
    void* pageAligned = allocatePageAligned();
    COOL_CHECK(pageAligned != nullptr);

    COOL_CHECK_EQ(coolHostRegister(pageAligned, 4096, coolHostRegisterDefault), coolSuccess);
    /* 重复注册同一地址：由驱动权威判定，映射为 ALREADY_REGISTERED。 */
    COOL_CHECK_EQ(coolHostRegister(pageAligned, 4096, coolHostRegisterDefault),
                  coolErrorHostMemoryAlreadyRegistered);

    /* 注册过的内存可直接参与异步拷贝。 */
    int* registeredHost = static_cast<int*>(pageAligned);
    registeredHost[0] = 7;
    void* registeredDev = nullptr;
    COOL_CHECK_EQ(coolMalloc(&registeredDev, sizeof(int)), coolSuccess);
    COOL_CHECK_EQ(coolMemcpyAsync(registeredDev, registeredHost, sizeof(int),
                                  coolMemcpyHostToDevice, nullptr),
                  coolSuccess);
    COOL_CHECK_EQ(coolStreamSynchronize(nullptr), coolSuccess);
    registeredHost[0] = 0;
    COOL_CHECK_EQ(coolMemcpyAsync(registeredHost, registeredDev, sizeof(int),
                                  coolMemcpyDeviceToHost, nullptr),
                  coolSuccess);
    COOL_CHECK_EQ(coolStreamSynchronize(nullptr), coolSuccess);
    COOL_CHECK_EQ(registeredHost[0], 7);

    COOL_CHECK_EQ(coolFree(registeredDev), coolSuccess);
    COOL_CHECK_EQ(coolHostUnregister(pageAligned), coolSuccess);
    /* 已注销后再注销：NOT_REGISTERED。 */
    COOL_CHECK_EQ(coolHostUnregister(pageAligned), coolErrorHostMemoryNotRegistered);

    /* 参数校验。 */
    COOL_CHECK_EQ(coolHostRegister(nullptr, 4096, coolHostRegisterDefault), coolErrorInvalidValue);
    COOL_CHECK_EQ(coolHostRegister(pageAligned, 0, coolHostRegisterDefault), coolErrorInvalidValue);
    COOL_CHECK_EQ(coolHostUnregister(nullptr), coolErrorInvalidValue);

    std::free(pageAligned);

    return cooltest::summary("test_mem_async");
}
