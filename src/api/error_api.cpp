// SPDX-FileCopyrightText: 2026 CoolRuntime contributors
// SPDX-License-Identifier: MIT

#include "cool/cool_runtime_api.h"

#include "core/error.hpp"

extern "C" const char* coolGetErrorString(coolError_t error) {
    return cool::errorString(error);
}

extern "C" const char* coolGetErrorName(coolError_t error) {
    return cool::errorName(error);
}

extern "C" coolError_t coolGetLastError(void) {
    return cool::lastError();
}

extern "C" coolError_t coolPeekAtLastError(void) {
    return cool::peekLastError();
}
