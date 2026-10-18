// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * R 层转发逻辑测试（无需 GPU）。
 *
 * 用假后端替换真实 cuda 后端，覆盖与后端实现无关的部分：
 *   1. 参数校验先于初始化（非法参数不应触发 init）；
 *   2. 懒初始化只发生一次；
 *   3. 能力位缺失 → coolErrorNotSupported（BAL 契约，见 cool_backend.h）；
 *   4. 后端错误码透传到调用方与 TLS；
 *   5. 实参按值转发（size / kind 等）。
 */
#include <cstddef>

#include "backend/cool_backend.h"
#include "cool/cool_runtime_api.h"
#include "core/lifecycle.hpp"

#include "support/check.hpp"

namespace {

struct FakeBackendState {
    int initCalls = 0;
    int finalizeCalls = 0;
    int mallocCalls = 0;
    int memcpyCalls = 0;
    std::size_t lastMallocSize = 0;
    coolMemcpyKind lastMemcpyKind = coolMemcpyHostToHost;
    void* mallocResultPtr = nullptr;
    coolError_t mallocResult = coolSuccess;
};

FakeBackendState g_fake;

coolError_t fakeInit(void) {
    ++g_fake.initCalls;
    return coolSuccess;
}

coolError_t fakeFinalize(void) {
    ++g_fake.finalizeCalls;
    return coolSuccess;
}

coolError_t fakeMalloc(void** ptr, std::size_t size) {
    ++g_fake.mallocCalls;
    g_fake.lastMallocSize = size;
    if (ptr != nullptr) {
        *ptr = g_fake.mallocResultPtr;
    }
    return g_fake.mallocResult;
}

coolError_t fakeMemcpy(void* dst, const void* src, std::size_t count, coolMemcpyKind kind) {
    (void)dst;
    (void)src;
    (void)count;
    ++g_fake.memcpyCalls;
    g_fake.lastMemcpyKind = kind;
    return coolSuccess;
}

/*
 * 只实现 init/finalize/malloc/memcpy 的最小后端。
 * 其余能力位保持 nullptr —— 正好用来覆盖 NotSupported 分支。
 */
coolBackend makeFakeBackend(void) {
    coolBackend backend{};
    backend.name = "fake";
    backend.apiVersion = COOL_BACKEND_API_VERSION;
    backend.init = fakeInit;
    backend.finalize = fakeFinalize;
    backend.malloc = fakeMalloc;
    backend.memcpy = fakeMemcpy;
    return backend;
}

}  // namespace

int main() {
    coolBackend fake = makeFakeBackend();
    cool::installBackendForTesting(&fake);

    /* 1) 参数校验先于初始化：非法参数不应触发后端 init。 */
    COOL_CHECK_EQ(coolMalloc(nullptr, 8), coolErrorInvalidValue);
    COOL_CHECK_EQ(g_fake.initCalls, 0);

    /* 2) 首次合法调用触发一次 init，并把实参透传给后端。 */
    void* p = nullptr;
    g_fake.mallocResultPtr = reinterpret_cast<void*>(0x1000);
    COOL_CHECK_EQ(coolMalloc(&p, 256), coolSuccess);
    COOL_CHECK_EQ(g_fake.initCalls, 1);
    COOL_CHECK_EQ(g_fake.mallocCalls, 1);
    COOL_CHECK_EQ(g_fake.lastMallocSize, std::size_t{256});
    COOL_CHECK(p == reinterpret_cast<void*>(0x1000));

    /* 3) 懒初始化只发生一次。 */
    COOL_CHECK_EQ(coolMalloc(&p, 16), coolSuccess);
    COOL_CHECK_EQ(g_fake.initCalls, 1);
    COOL_CHECK_EQ(g_fake.mallocCalls, 2);

    /* 4) 实参按值转发：memcpy 的 kind 应原样到达后端。 */
    COOL_CHECK_EQ(coolMemcpy(p, p, 4, coolMemcpyDeviceToHost), coolSuccess);
    COOL_CHECK_EQ(g_fake.memcpyCalls, 1);
    COOL_CHECK_EQ(g_fake.lastMemcpyKind, coolMemcpyDeviceToHost);

    /* 5) 能力位缺失 → coolErrorNotSupported（BAL 契约，不是 NotReady）。 */
    COOL_CHECK_EQ(coolFree(p), coolErrorNotSupported);
    COOL_CHECK_EQ(coolMemset(p, 0, 4), coolErrorNotSupported);
    COOL_CHECK_EQ(coolFreeHost(p), coolErrorNotSupported);
    coolStream_t stream = nullptr;
    COOL_CHECK_EQ(coolStreamCreate(&stream), coolErrorNotSupported);
    /* 这两处原本把「未实现」返回成 NotReady，由统一前置步骤修正。 */
    COOL_CHECK_EQ(coolDeviceSynchronize(), coolErrorNotSupported);
    int driverVersion = 0;
    COOL_CHECK_EQ(coolDriverGetVersion(&driverVersion), coolErrorNotSupported);

    /* 6) 后端错误码透传到返回值与 TLS。 */
    g_fake.mallocResult = coolErrorOutOfMemory;
    COOL_CHECK_EQ(coolMalloc(&p, 1024), coolErrorOutOfMemory);
    COOL_CHECK_EQ(coolPeekAtLastError(), coolErrorOutOfMemory);
    COOL_CHECK_EQ(coolGetLastError(), coolErrorOutOfMemory);
    COOL_CHECK_EQ(coolGetLastError(), coolSuccess); /* 读取即清空 */

    return cooltest::summary("test_api_dispatch");
}
