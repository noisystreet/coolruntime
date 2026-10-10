// SPDX-FileCopyrightText: 2026 CoolRuntime contributors
// SPDX-License-Identifier: MIT

/* 内存管理测试：在 cuda 后端下运行（需要可用的 NVIDIA 驱动）。 */
#include <cstddef>

#include "cool/cool_runtime_api.h"

#include "support/check.hpp"

int main() {
    /* 分配 / 释放。 */
    void* deviceBuf = nullptr;
    COOL_CHECK_EQ(coolMalloc(&deviceBuf, 64), coolSuccess);
    COOL_CHECK(deviceBuf != nullptr);
    COOL_CHECK_EQ(coolFree(deviceBuf), coolSuccess);

    /* 非法参数。 */
    COOL_CHECK_EQ(coolMalloc(nullptr, 16), coolErrorInvalidValue);
    void* zeroBuf = nullptr;
    COOL_CHECK_EQ(coolMalloc(&zeroBuf, 0), coolErrorInvalidValue);
    COOL_CHECK_EQ(coolFree(nullptr), coolSuccess);

    /* 释放未知/重复指针应被注册表拦截。 */
    int hostInt = 0;
    COOL_CHECK_EQ(coolFree(&hostInt), coolErrorInvalidValue);

    /* 分配一块并做 memset / memcpy 往返校验。 */
    void* buf = nullptr;
    COOL_CHECK_EQ(coolMalloc(&buf, 64), coolSuccess);
    COOL_CHECK_EQ(coolMemset(buf, 0xAB, 64), coolSuccess);

    unsigned char host[64];
    for (int i = 0; i < 64; ++i) {
        host[i] = 0;
    }
    COOL_CHECK_EQ(coolMemcpy(host, buf, 64, coolMemcpyDeviceToHost), coolSuccess);
    for (int i = 0; i < 64; ++i) {
        COOL_CHECK_EQ(host[i], 0xAB);
    }

    /* H2D -> D2H 往返。 */
    unsigned char source[64];
    for (int i = 0; i < 64; ++i) {
        source[i] = static_cast<unsigned char>(i);
    }
    COOL_CHECK_EQ(coolMemcpy(buf, source, 64, coolMemcpyHostToDevice), coolSuccess);
    COOL_CHECK_EQ(coolMemcpy(host, buf, 64, coolMemcpyDeviceToHost), coolSuccess);
    for (int i = 0; i < 64; ++i) {
        COOL_CHECK_EQ(host[i], source[i]);
    }

    /* D2D 拷贝。 */
    void* buf2 = nullptr;
    COOL_CHECK_EQ(coolMalloc(&buf2, 64), coolSuccess);
    COOL_CHECK_EQ(coolMemcpy(buf2, buf, 64, coolMemcpyDeviceToDevice), coolSuccess);
    COOL_CHECK_EQ(coolMemcpy(host, buf2, 64, coolMemcpyDeviceToHost), coolSuccess);
    for (int i = 0; i < 64; ++i) {
        COOL_CHECK_EQ(host[i], source[i]);
    }

    /* H2H 拷贝。 */
    unsigned char hostDst[64];
    COOL_CHECK_EQ(coolMemcpy(hostDst, source, 64, coolMemcpyHostToHost), coolSuccess);
    for (int i = 0; i < 64; ++i) {
        COOL_CHECK_EQ(hostDst[i], source[i]);
    }

    /* 非法拷贝参数。 */
    COOL_CHECK_EQ(coolMemcpy(nullptr, buf, 8, coolMemcpyDeviceToHost), coolErrorInvalidValue);
    COOL_CHECK_EQ(coolMemcpy(host, buf, 8, static_cast<coolMemcpyKind>(999)),
                  coolErrorInvalidValue);

    /* 重复释放应返回错误，且已释放的指针不再命中注册表。 */
    COOL_CHECK_EQ(coolFree(buf2), coolSuccess);
    COOL_CHECK_EQ(coolFree(buf2), coolErrorInvalidValue);
    COOL_CHECK_EQ(coolFree(buf), coolSuccess);

    /* 设备内存信息。 */
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    COOL_CHECK_EQ(coolMemGetInfo(&freeBytes, &totalBytes), coolSuccess);
    COOL_CHECK(totalBytes > 0);
    COOL_CHECK(freeBytes <= totalBytes);
    COOL_CHECK_EQ(coolMemGetInfo(nullptr, &totalBytes), coolErrorInvalidValue);
    COOL_CHECK_EQ(coolMemGetInfo(&freeBytes, nullptr), coolErrorInvalidValue);

    /* 页锁定主机内存。 */
    void* pinned = nullptr;
    COOL_CHECK_EQ(coolMallocHost(&pinned, 128), coolSuccess);
    COOL_CHECK(pinned != nullptr);
    COOL_CHECK_EQ(coolFreeHost(pinned), coolSuccess);
    COOL_CHECK_EQ(coolFreeHost(nullptr), coolSuccess);
    COOL_CHECK_EQ(coolMallocHost(nullptr, 128), coolErrorInvalidValue);

    return cooltest::summary("test_mem");
}
