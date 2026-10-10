// SPDX-FileCopyrightText: 2026 CoolRuntime contributors
// SPDX-License-Identifier: MIT

#include "cool/cool_runtime_api.h"

#include "core/error.hpp"
#include "core/lifecycle.hpp"

extern "C" coolError_t coolMalloc(void** ptr, size_t size) {
    if (ptr == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_GUARD(cool::ensureInitialized());

    const coolBackend* backend = cool::activeBackend();
    if (backend == nullptr) {
        COOL_API_RETURN(coolErrorNotReady);
    }
    if (backend->malloc == nullptr) {
        COOL_API_RETURN(coolErrorNotSupported);
    }
    COOL_API_GUARD(backend->malloc(ptr, size));
    return coolSuccess;
}

extern "C" coolError_t coolFree(void* ptr) {
    COOL_API_GUARD(cool::ensureInitialized());

    const coolBackend* backend = cool::activeBackend();
    if (backend == nullptr || backend->free == nullptr) {
        COOL_API_RETURN(coolErrorNotReady);
    }
    COOL_API_GUARD(backend->free(ptr));
    return coolSuccess;
}

extern "C" coolError_t coolMemcpy(void* dst, const void* src, size_t count, coolMemcpyKind kind) {
    COOL_API_GUARD(cool::ensureInitialized());

    const coolBackend* backend = cool::activeBackend();
    if (backend == nullptr || backend->memcpy == nullptr) {
        COOL_API_RETURN(coolErrorNotReady);
    }
    COOL_API_GUARD(backend->memcpy(dst, src, count, kind));
    return coolSuccess;
}

extern "C" coolError_t coolMemset(void* ptr, int value, size_t count) {
    COOL_API_GUARD(cool::ensureInitialized());

    const coolBackend* backend = cool::activeBackend();
    if (backend == nullptr || backend->memset == nullptr) {
        COOL_API_RETURN(coolErrorNotReady);
    }
    COOL_API_GUARD(backend->memset(ptr, value, count));
    return coolSuccess;
}

extern "C" coolError_t coolMemGetInfo(size_t* freeBytes, size_t* totalBytes) {
    if (freeBytes == nullptr || totalBytes == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_GUARD(cool::ensureInitialized());

    const coolBackend* backend = cool::activeBackend();
    if (backend == nullptr) {
        COOL_API_RETURN(coolErrorNotReady);
    }
    if (backend->memGetInfo == nullptr) {
        COOL_API_RETURN(coolErrorNotSupported);
    }
    COOL_API_GUARD(backend->memGetInfo(freeBytes, totalBytes));
    return coolSuccess;
}

extern "C" coolError_t coolMallocHost(void** ptr, size_t size) {
    if (ptr == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_GUARD(cool::ensureInitialized());

    const coolBackend* backend = cool::activeBackend();
    if (backend == nullptr) {
        COOL_API_RETURN(coolErrorNotReady);
    }
    if (backend->mallocHost == nullptr) {
        COOL_API_RETURN(coolErrorNotSupported);
    }
    /* flags 传 0：默认页锁定分配，与 cudart 的 cudaMallocHost 语义一致。 */
    COOL_API_GUARD(backend->mallocHost(ptr, size, 0));
    return coolSuccess;
}

extern "C" coolError_t coolFreeHost(void* ptr) {
    COOL_API_GUARD(cool::ensureInitialized());

    const coolBackend* backend = cool::activeBackend();
    if (backend == nullptr || backend->freeHost == nullptr) {
        COOL_API_RETURN(coolErrorNotReady);
    }
    COOL_API_GUARD(backend->freeHost(ptr));
    return coolSuccess;
}
