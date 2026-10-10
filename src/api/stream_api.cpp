// SPDX-FileCopyrightText: 2026 CoolRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * A 层流管理接口：参数校验后薄透传 BAL，不持有任何后端状态。
 * 流句柄的状态与校验由 cuda 后端内部承担（见 ADR 0003）。
 */
#include "cool/cool_runtime_api.h"

#include "core/error.hpp"
#include "core/lifecycle.hpp"

extern "C" coolError_t coolStreamCreate(coolStream_t* stream) {
    return coolStreamCreateWithFlags(stream, coolStreamDefault);
}

extern "C" coolError_t coolStreamCreateWithFlags(coolStream_t* stream, unsigned int flags) {
    return coolStreamCreateWithPriority(stream, flags, 0);
}

extern "C" coolError_t coolStreamCreateWithPriority(coolStream_t* stream, unsigned int flags,
                                                    int priority) {
    if (stream == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_GUARD(cool::ensureInitialized());

    const coolBackend* backend = cool::activeBackend();
    if (backend == nullptr) {
        COOL_API_RETURN(coolErrorNotReady);
    }
    if (backend->streamCreate == nullptr) {
        COOL_API_RETURN(coolErrorNotSupported);
    }
    COOL_API_GUARD(backend->streamCreate(stream, flags, priority));
    return coolSuccess;
}

extern "C" coolError_t coolStreamDestroy(coolStream_t stream) {
    COOL_API_GUARD(cool::ensureInitialized());

    const coolBackend* backend = cool::activeBackend();
    if (backend == nullptr) {
        COOL_API_RETURN(coolErrorNotReady);
    }
    if (backend->streamDestroy == nullptr) {
        COOL_API_RETURN(coolErrorNotSupported);
    }
    COOL_API_GUARD(backend->streamDestroy(stream));
    return coolSuccess;
}

extern "C" coolError_t coolStreamSynchronize(coolStream_t stream) {
    COOL_API_GUARD(cool::ensureInitialized());

    const coolBackend* backend = cool::activeBackend();
    if (backend == nullptr) {
        COOL_API_RETURN(coolErrorNotReady);
    }
    if (backend->streamSynchronize == nullptr) {
        COOL_API_RETURN(coolErrorNotSupported);
    }
    COOL_API_GUARD(backend->streamSynchronize(stream));
    return coolSuccess;
}

extern "C" coolError_t coolStreamQuery(coolStream_t stream) {
    COOL_API_GUARD(cool::ensureInitialized());

    const coolBackend* backend = cool::activeBackend();
    if (backend == nullptr) {
        COOL_API_RETURN(coolErrorNotReady);
    }
    if (backend->streamQuery == nullptr) {
        COOL_API_RETURN(coolErrorNotSupported);
    }
    COOL_API_GUARD(backend->streamQuery(stream));
    return coolSuccess;
}

extern "C" coolError_t coolStreamWaitEvent(coolStream_t stream, coolEvent_t event) {
    if (event == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_GUARD(cool::ensureInitialized());

    const coolBackend* backend = cool::activeBackend();
    if (backend == nullptr) {
        COOL_API_RETURN(coolErrorNotReady);
    }
    if (backend->streamWaitEvent == nullptr) {
        COOL_API_RETURN(coolErrorNotSupported);
    }
    COOL_API_GUARD(backend->streamWaitEvent(stream, event));
    return coolSuccess;
}
