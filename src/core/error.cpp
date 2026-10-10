// SPDX-FileCopyrightText: 2026 CoolRuntime contributors
// SPDX-License-Identifier: MIT

#include "core/error.hpp"

namespace cool {
namespace {

thread_local coolError_t t_lastError = coolSuccess;

struct ErrorEntry {
    coolError_t code;
    const char* name;
    const char* text;
};

/* 错误码到「枚举名 / 可读文本」的静态映射表。 */
constexpr ErrorEntry kErrors[] = {
    {coolSuccess, "coolSuccess", "no error"},
    {coolErrorInvalidValue, "coolErrorInvalidValue", "invalid argument"},
    {coolErrorOutOfMemory, "coolErrorOutOfMemory", "out of memory"},
    {coolErrorMemoryAllocation, "coolErrorMemoryAllocation", "memory allocation failed"},
    {coolErrorInvalidDevice, "coolErrorInvalidDevice", "invalid device ordinal"},
    {coolErrorInvalidResourceHandle, "coolErrorInvalidResourceHandle", "invalid resource handle"},
    {coolErrorNotReady, "coolErrorNotReady", "device not ready"},
    {coolErrorNotSupported, "coolErrorNotSupported", "operation not supported"},
    {coolErrorUnknown, "coolErrorUnknown", "unknown error"},
};

const ErrorEntry* findEntry(coolError_t err) {
    for (const ErrorEntry& entry : kErrors) {
        if (entry.code == err) {
            return &entry;
        }
    }
    return nullptr;
}

}  // namespace

void setLastError(coolError_t err) {
    t_lastError = err;
}

coolError_t lastError() {
    const coolError_t err = t_lastError;
    t_lastError = coolSuccess;
    return err;
}

coolError_t peekLastError() {
    return t_lastError;
}

const char* errorName(coolError_t err) {
    const ErrorEntry* entry = findEntry(err);
    return entry != nullptr ? entry->name : "coolErrorUnknown";
}

const char* errorString(coolError_t err) {
    const ErrorEntry* entry = findEntry(err);
    return entry != nullptr ? entry->text : "unknown error";
}

}  // namespace cool
