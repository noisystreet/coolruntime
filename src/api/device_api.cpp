// SPDX-FileCopyrightText: 2026 CoolRuntime contributors
// SPDX-License-Identifier: MIT

#include "cool/cool_runtime_api.h"

#include "core/error.hpp"
#include "core/lifecycle.hpp"

namespace {
/* 当前设备为线程局部状态，语义与 cudart 一致。 */
thread_local int t_currentDevice = 0;
}  // namespace

extern "C" coolError_t coolGetDeviceCount(int* count) {
    if (count == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_GUARD(cool::ensureInitialized());

    const coolBackend* backend = cool::activeBackend();
    if (backend == nullptr) {
        COOL_API_RETURN(coolErrorNotReady);
    }
    if (backend->getDeviceCount == nullptr) {
        COOL_API_RETURN(coolErrorNotSupported);
    }
    COOL_API_GUARD(backend->getDeviceCount(count));
    return coolSuccess;
}

extern "C" coolError_t coolSetDevice(int device) {
    if (device < 0) {
        COOL_API_RETURN(coolErrorInvalidDevice);
    }
    COOL_API_GUARD(cool::ensureInitialized());

    const coolBackend* backend = cool::activeBackend();
    if (backend == nullptr || backend->getDeviceCount == nullptr) {
        COOL_API_RETURN(coolErrorNotReady);
    }

    int count = 0;
    COOL_API_GUARD(backend->getDeviceCount(&count));
    if (device >= count) {
        COOL_API_RETURN(coolErrorInvalidDevice);
    }

    t_currentDevice = device;
    return coolSuccess;
}

extern "C" coolError_t coolGetDevice(int* device) {
    if (device == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    *device = t_currentDevice;
    return coolSuccess;
}

extern "C" coolError_t coolGetDeviceProperties(coolDeviceProp* prop, int device) {
    if (prop == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_GUARD(cool::ensureInitialized());

    const coolBackend* backend = cool::activeBackend();
    if (backend == nullptr) {
        COOL_API_RETURN(coolErrorNotReady);
    }
    if (backend->getDeviceProperties == nullptr) {
        COOL_API_RETURN(coolErrorNotSupported);
    }
    COOL_API_GUARD(backend->getDeviceProperties(device, prop));
    return coolSuccess;
}

extern "C" coolError_t coolDeviceSynchronize(void) {
    COOL_API_GUARD(cool::ensureInitialized());

    const coolBackend* backend = cool::activeBackend();
    if (backend == nullptr || backend->deviceSynchronize == nullptr) {
        COOL_API_RETURN(coolErrorNotReady);
    }
    COOL_API_GUARD(backend->deviceSynchronize());
    return coolSuccess;
}

extern "C" coolError_t coolDeviceReset(void) {
    COOL_API_GUARD(cool::ensureInitialized());

    const coolBackend* backend = cool::activeBackend();
    if (backend == nullptr || backend->deviceReset == nullptr) {
        COOL_API_RETURN(coolErrorNotReady);
    }
    COOL_API_GUARD(backend->deviceReset());
    return coolSuccess;
}
