// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

#include "core/lifecycle.hpp"

#include <cstdlib>

#include "backend/backend_registry.hpp"
#include "core/error.hpp"

namespace cool {
namespace {

struct LifecycleState {
    const coolBackend* backend = nullptr;
    coolError_t result = coolErrorNotReady;
    bool attempted = false;
    bool atexitRegistered = false;
};

LifecycleState& state() {
    static LifecycleState s;
    return s;
}

/*
 * 测试注入的后端；非 nullptr 时优先于内置 cuda 后端。
 * 只在测试进程启动时（任何 cool API 调用之前）单线程写入一次，故不做同步。
 */
const coolBackend* g_testBackend = nullptr;

void finalizeAtExit() {
    LifecycleState& s = state();
    if (s.backend != nullptr && s.backend->finalize != nullptr) {
        s.backend->finalize();
    }
    s.backend = nullptr;
}

}  // namespace

void installBackendForTesting(const coolBackend* backend) {
    g_testBackend = backend;
    LifecycleState& s = state();
    s.backend = nullptr;
    s.result = coolErrorNotReady;
    s.attempted = false;
}

coolError_t ensureInitialized() {
    LifecycleState& s = state();
    if (s.attempted) {
        return s.result;
    }
    s.attempted = true;

    const coolBackend* backend = g_testBackend != nullptr ? g_testBackend : cudaBackend();

    coolError_t err = coolErrorInvalidValue;
    if (backend != nullptr && backend->init != nullptr) {
        err = backend->init();
    }

    if (err != coolSuccess) {
        s.backend = nullptr;
        s.result = err;
        return s.result;
    }

    s.backend = backend;
    s.result = coolSuccess;
    /* 只注册一次：测试可能反复注入后端，重复注册会多次触发 finalize。 */
    if (!s.atexitRegistered) {
        s.atexitRegistered = true;
        std::atexit(finalizeAtExit);
    }
    return coolSuccess;
}

const coolBackend* activeBackend() {
    return state().backend;
}

}  // namespace cool
