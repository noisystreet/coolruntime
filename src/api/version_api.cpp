// SPDX-FileCopyrightText: 2026 CoolRuntime contributors
// SPDX-License-Identifier: MIT

#include "cool/cool_runtime_api.h"

#include "core/error.hpp"
#include "core/lifecycle.hpp"

extern "C" coolError_t coolRuntimeGetVersion(int* runtimeVersion) {
    if (runtimeVersion == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    *runtimeVersion = COOL_VERSION;
    return coolSuccess;
}

extern "C" coolError_t coolDriverGetVersion(int* driverVersion) {
    if (driverVersion == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_GUARD(cool::ensureInitialized());

    const coolBackend* backend = cool::activeBackend();
    if (backend == nullptr || backend->driverGetVersion == nullptr) {
        COOL_API_RETURN(coolErrorNotReady);
    }
    COOL_API_GUARD(backend->driverGetVersion(driverVersion));
    return coolSuccess;
}
