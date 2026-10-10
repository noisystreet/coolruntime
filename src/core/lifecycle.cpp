// SPDX-FileCopyrightText: 2026 CoolRuntime contributors
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
};

LifecycleState& state() {
    static LifecycleState s;
    return s;
}

void finalizeAtExit() {
    LifecycleState& s = state();
    if (s.backend != nullptr && s.backend->finalize != nullptr) {
        s.backend->finalize();
    }
    s.backend = nullptr;
}

}  // namespace

coolError_t ensureInitialized() {
    LifecycleState& s = state();
    if (s.attempted) {
        return s.result;
    }
    s.attempted = true;

    const coolBackend* backend = cudaBackend();

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
    std::atexit(finalizeAtExit);
    return coolSuccess;
}

const coolBackend* activeBackend() {
    return state().backend;
}

}  // namespace cool
