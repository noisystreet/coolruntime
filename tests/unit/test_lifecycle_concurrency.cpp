// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * 首次并发调用只初始化一次（一次性初始化的同步回归）。
 *
 * 历史缺陷：懒初始化用函数局部静态量实现但**没有同步**——多个线程首次并发
 * 调用时会各自读到「未初始化」，都去执行 backend->init()。在 cuda 后端下
 * 就是重复 cuCtxCreate：泄漏 context 并覆盖全局句柄。
 *
 * 本测试让 8 个线程同时冲进 coolMalloc，假后端的 init 内部 sleep 以放大竞态
 * 窗口，然后断言 init 恰好发生一次。
 */
#include <atomic>
#include <chrono>
#include <cstddef>
#include <thread>
#include <vector>

#include "backend/cool_backend.h"
#include "cool/cool_runtime_api.h"
#include "core/lifecycle.hpp"

#include "support/check.hpp"

namespace {

std::atomic<int> g_initCalls{0};
std::atomic<int> g_finalizeCalls{0};
std::atomic<int> g_mallocCalls{0};

coolError_t fakeInit(void) {
    ++g_initCalls;
    /* 放大竞态窗口：若同步缺失，其他线程会在这段时间里也进入 init。 */
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    return coolSuccess;
}

coolError_t fakeFinalize(void) {
    ++g_finalizeCalls;
    return coolSuccess;
}

coolError_t fakeMalloc(void** ptr, std::size_t size) {
    (void)size;
    ++g_mallocCalls;
    if (ptr != nullptr) {
        *ptr = reinterpret_cast<void*>(0x1);
    }
    return coolSuccess;
}

coolBackend makeFakeBackend(void) {
    coolBackend backend{};
    backend.name = "fake-concurrent";
    backend.apiVersion = COOL_BACKEND_API_VERSION;
    backend.init = fakeInit;
    backend.finalize = fakeFinalize;
    backend.malloc = fakeMalloc;
    return backend;
}

}  // namespace

int main() {
    coolBackend fake = makeFakeBackend();
    cool::installBackendForTesting(&fake);

    constexpr int kThreads = 8;
    std::atomic<int> failures{0};

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&failures] {
            void* p = nullptr;
            if (coolMalloc(&p, 64) != coolSuccess) {
                ++failures;
            }
        });
    }
    for (std::thread& t : threads) {
        t.join();
    }

    /* 核心断言：8 个线程并发首次调用，后端初始化只发生一次。 */
    COOL_CHECK_EQ(g_initCalls.load(), 1);
    /* 每个线程的调用都真正转发到了后端。 */
    COOL_CHECK_EQ(g_mallocCalls.load(), kThreads);
    COOL_CHECK_EQ(failures.load(), 0);

    return cooltest::summary("test_lifecycle_concurrency");
}
