// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

#include "backend/bootstrap.hpp"

#include <atomic>
#include <cstdlib>
#include <mutex>

#include "backend/backend_registry.hpp"

namespace cool {
namespace bootstrap {
namespace {

struct Ready {
    const coolBackend* backend = nullptr;
    coolError_t result = coolErrorNotReady;
    bool atexitRegistered = false;
};

/*
 * 构造顺序 = 首次 ensureReady() 内的首次访问顺序：
 *   initialised() → initMutex() → ready()
 * 三者都在 std::atexit(finalizeAtExit) 之前构造，故 atexit 处理器的执行
 * （LIFO，最先注册的最后执行……此处 finalizeAtExit 注册最晚）先于它们析构，
 * finalizeAtExit 里访问 ready()/initMutex() 是安全的。
 */
Ready& ready() {
    static Ready r;
    return r;
}

/* 串行化「初始化 / 复位 / atexit 清理」。 */
std::mutex& initMutex() {
    static std::mutex m;
    return m;
}

/* 初始化是否已定局。置位后 ready() 的字段可被无锁读取（release/acquire 配对）。 */
std::atomic<bool>& initialised() {
    static std::atomic<bool> done{false};
    return done;
}

/* 测试注入的后端；非 nullptr 时优先于内置 cuda 后端。受 initMutex 保护。 */
const coolBackend* g_testBackend = nullptr;

void finalizeAtExit() {
    std::lock_guard<std::mutex> lock(initMutex());
    Ready& r = ready();
    if (r.backend != nullptr && r.backend->finalize != nullptr) {
        r.backend->finalize();
    }
    r.backend = nullptr;
    r.result = coolErrorNotReady;
    /* initialised() 保持 true：进程正在退出，不允许再重新初始化。 */
}

}  // namespace

coolError_t ensureReady() {
    /* 快路径：已完成初始化时无锁。 */
    if (initialised().load(std::memory_order_acquire)) {
        return ready().result;
    }

    std::lock_guard<std::mutex> lock(initMutex());
    /* 二次检查：并发线程可能已在本线程等锁期间完成初始化。 */
    if (initialised().load(std::memory_order_relaxed)) {
        return ready().result;
    }

    Ready& r = ready();
    const coolBackend* backend = g_testBackend != nullptr ? g_testBackend : cudaBackend();

    coolError_t err = coolErrorInvalidValue;
    if (backend != nullptr && backend->init != nullptr) {
        err = backend->init();
    }

    if (err != coolSuccess) {
        r.backend = nullptr;
        r.result = err;
        initialised().store(true, std::memory_order_release);
        return err;
    }

    r.backend = backend;
    r.result = coolSuccess;
    if (!r.atexitRegistered) {
        r.atexitRegistered = true;
        std::atexit(finalizeAtExit);
    }
    initialised().store(true, std::memory_order_release);
    return coolSuccess;
}

const coolBackend* backend() {
    if (!initialised().load(std::memory_order_acquire)) {
        return nullptr;
    }
    return ready().backend;
}

void installForTesting(const coolBackend* backend) {
    std::lock_guard<std::mutex> lock(initMutex());
    g_testBackend = backend;
    Ready& r = ready();
    r.backend = nullptr;
    r.result = coolErrorNotReady;
    initialised().store(false, std::memory_order_release);
}

}  // namespace bootstrap
}  // namespace cool
