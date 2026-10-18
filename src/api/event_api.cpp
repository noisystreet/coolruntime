// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * A 层事件管理接口：参数校验后薄透传 BAL，不持有任何后端状态。
 * 事件句柄的状态与校验由 cuda 后端内部承担（见 ADR 0003）。
 */
#include "cool/cool_runtime_api.h"

#include "core/error.hpp"
#include "core/lifecycle.hpp"

extern "C" coolError_t coolEventCreate(coolEvent_t* event) {
    return coolEventCreateWithFlags(event, coolEventDefault);
}

extern "C" coolError_t coolEventCreateWithFlags(coolEvent_t* event, unsigned int flags) {
    if (event == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_ENTER(eventCreate);
    COOL_API_GUARD(backend->eventCreate(event, flags));
    return coolSuccess;
}

extern "C" coolError_t coolEventDestroy(coolEvent_t event) {
    COOL_API_ENTER(eventDestroy);
    COOL_API_GUARD(backend->eventDestroy(event));
    return coolSuccess;
}

extern "C" coolError_t coolEventRecord(coolEvent_t event, coolStream_t stream) {
    if (event == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_ENTER(eventRecord);
    COOL_API_GUARD(backend->eventRecord(event, stream));
    return coolSuccess;
}

extern "C" coolError_t coolEventSynchronize(coolEvent_t event) {
    COOL_API_ENTER(eventSynchronize);
    COOL_API_GUARD(backend->eventSynchronize(event));
    return coolSuccess;
}

extern "C" coolError_t coolEventQuery(coolEvent_t event) {
    COOL_API_ENTER(eventQuery);
    COOL_API_GUARD(backend->eventQuery(event));
    return coolSuccess;
}

extern "C" coolError_t coolEventElapsedTime(float* ms, coolEvent_t start, coolEvent_t end) {
    if (ms == nullptr || start == nullptr || end == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_ENTER(eventElapsedTime);
    COOL_API_GUARD(backend->eventElapsedTime(ms, start, end));
    return coolSuccess;
}
