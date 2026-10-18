// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
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
    COOL_API_ENTER(streamCreate);
    COOL_API_GUARD(backend->streamCreate(stream, flags, priority));
    return coolSuccess;
}

extern "C" coolError_t coolStreamDestroy(coolStream_t stream) {
    COOL_API_ENTER(streamDestroy);
    COOL_API_GUARD(backend->streamDestroy(stream));
    return coolSuccess;
}

extern "C" coolError_t coolStreamSynchronize(coolStream_t stream) {
    COOL_API_ENTER(streamSynchronize);
    COOL_API_GUARD(backend->streamSynchronize(stream));
    return coolSuccess;
}

extern "C" coolError_t coolStreamQuery(coolStream_t stream) {
    COOL_API_ENTER(streamQuery);
    COOL_API_GUARD(backend->streamQuery(stream));
    return coolSuccess;
}

extern "C" coolError_t coolStreamWaitEvent(coolStream_t stream, coolEvent_t event) {
    if (event == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_ENTER(streamWaitEvent);
    COOL_API_GUARD(backend->streamWaitEvent(stream, event));
    return coolSuccess;
}

extern "C" coolError_t coolStreamGetPriority(int* priority, coolStream_t stream) {
    if (priority == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_ENTER(streamGetPriority);
    COOL_API_GUARD(backend->streamGetPriority(priority, stream));
    return coolSuccess;
}

extern "C" coolError_t coolStreamGetFlags(unsigned int* flags, coolStream_t stream) {
    if (flags == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_ENTER(streamGetFlags);
    COOL_API_GUARD(backend->streamGetFlags(flags, stream));
    return coolSuccess;
}
