// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

#include "cool/cool_runtime_api.h"

#include "core/error.hpp"
#include "core/lifecycle.hpp"

extern "C" coolError_t coolMalloc(void** ptr, size_t size) {
    if (ptr == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_ENTER(malloc);
    COOL_API_GUARD(backend->malloc(ptr, size));
    return coolSuccess;
}

extern "C" coolError_t coolFree(void* ptr) {
    COOL_API_ENTER(free);
    COOL_API_GUARD(backend->free(ptr));
    return coolSuccess;
}

extern "C" coolError_t coolMemcpy(void* dst, const void* src, size_t count, coolMemcpyKind kind) {
    COOL_API_ENTER(memcpy);
    COOL_API_GUARD(backend->memcpy(dst, src, count, kind));
    return coolSuccess;
}

extern "C" coolError_t coolMemset(void* ptr, int value, size_t count) {
    COOL_API_ENTER(memset);
    COOL_API_GUARD(backend->memset(ptr, value, count));
    return coolSuccess;
}

extern "C" coolError_t coolMemcpyAsync(void* dst, const void* src, size_t count,
                                       coolMemcpyKind kind, coolStream_t stream) {
    if (dst == nullptr || src == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_ENTER(memcpyAsync);
    COOL_API_GUARD(backend->memcpyAsync(dst, src, count, kind, stream));
    return coolSuccess;
}

extern "C" coolError_t coolMemsetAsync(void* ptr, int value, size_t count, coolStream_t stream) {
    if (ptr == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_ENTER(memsetAsync);
    COOL_API_GUARD(backend->memsetAsync(ptr, value, count, stream));
    return coolSuccess;
}

extern "C" coolError_t coolMemGetInfo(size_t* freeBytes, size_t* totalBytes) {
    if (freeBytes == nullptr || totalBytes == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_ENTER(memGetInfo);
    COOL_API_GUARD(backend->memGetInfo(freeBytes, totalBytes));
    return coolSuccess;
}

extern "C" coolError_t coolMallocHost(void** ptr, size_t size) {
    if (ptr == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_ENTER(mallocHost);
    /* flags 传 0：默认页锁定分配，与 cudart 的 cudaMallocHost 语义一致。 */
    COOL_API_GUARD(backend->mallocHost(ptr, size, 0));
    return coolSuccess;
}

extern "C" coolError_t coolFreeHost(void* ptr) {
    COOL_API_ENTER(freeHost);
    COOL_API_GUARD(backend->freeHost(ptr));
    return coolSuccess;
}

extern "C" coolError_t coolHostRegister(void* ptr, size_t size, unsigned int flags) {
    if (ptr == nullptr || size == 0) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_ENTER(memHostRegister);
    COOL_API_GUARD(backend->memHostRegister(ptr, size, flags));
    return coolSuccess;
}

extern "C" coolError_t coolHostUnregister(void* ptr) {
    if (ptr == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_ENTER(memHostUnregister);
    COOL_API_GUARD(backend->memHostUnregister(ptr));
    return coolSuccess;
}

extern "C" coolError_t coolMemcpyToSymbol(const void* symbol, const void* src, size_t count,
                                          size_t offset, coolMemcpyKind kind) {
    if (symbol == nullptr || src == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_ENTER(memcpyToSymbol);
    COOL_API_GUARD(backend->memcpyToSymbol(symbol, src, count, offset, kind));
    return coolSuccess;
}

extern "C" coolError_t coolMemcpyFromSymbol(void* dst, const void* symbol, size_t count,
                                            size_t offset, coolMemcpyKind kind) {
    if (dst == nullptr || symbol == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_ENTER(memcpyFromSymbol);
    COOL_API_GUARD(backend->memcpyFromSymbol(dst, symbol, count, offset, kind));
    return coolSuccess;
}

extern "C" coolError_t coolGetSymbolAddress(void** devPtr, const void* symbol) {
    if (devPtr == nullptr || symbol == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_ENTER(getSymbolAddress);
    COOL_API_GUARD(backend->getSymbolAddress(devPtr, symbol));
    return coolSuccess;
}

extern "C" coolError_t coolGetSymbolSize(size_t* size, const void* symbol) {
    if (size == nullptr || symbol == nullptr) {
        COOL_API_RETURN(coolErrorInvalidValue);
    }
    COOL_API_ENTER(getSymbolSize);
    COOL_API_GUARD(backend->getSymbolSize(size, symbol));
    return coolSuccess;
}
