// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
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

    /* P6 R2 新增项。文本对齐 cudart 的 cudaGetErrorString 措辞，便于交叉阅读。 */
    {coolErrorInvalidConfiguration, "coolErrorInvalidConfiguration",
     "invalid configuration argument"},
    {coolErrorInvalidDevice, "coolErrorInvalidDevice", "invalid device ordinal"},
    {coolErrorInvalidPitchValue, "coolErrorInvalidPitchValue", "invalid pitch value"},
    {coolErrorInvalidSymbol, "coolErrorInvalidSymbol", "invalid symbol"},
    {coolErrorInvalidDevicePointer, "coolErrorInvalidDevicePointer", "invalid device pointer"},
    {coolErrorInvalidMemcpyDirection, "coolErrorInvalidMemcpyDirection",
     "invalid copy direction for memcpy"},
    {coolErrorInsufficientDriver, "coolErrorInsufficientDriver",
     "CUDA driver version is insufficient for CUDA runtime version"},
    {coolErrorInvalidDeviceFunction, "coolErrorInvalidDeviceFunction", "invalid device function"},
    {coolErrorNoDevice, "coolErrorNoDevice", "no CUDA-capable device is detected"},
    {coolErrorInvalidKernelImage, "coolErrorInvalidKernelImage", "invalid kernel image"},
    {coolErrorUnsupportedLimit, "coolErrorUnsupportedLimit", "unsupported limit"},
    {coolErrorInvalidSource, "coolErrorInvalidSource", "invalid source"},
    {coolErrorFileNotFound, "coolErrorFileNotFound", "file not found"},
    {coolErrorInvalidResourceHandle, "coolErrorInvalidResourceHandle", "invalid resource handle"},
    {coolErrorNotReady, "coolErrorNotReady", "device not ready"},
    {coolErrorIllegalAddress, "coolErrorIllegalAddress",
     "an illegal memory access was encountered"},
    {coolErrorLaunchOutOfResources, "coolErrorLaunchOutOfResources",
     "too many resources requested for launch"},
    {coolErrorLaunchTimeout, "coolErrorLaunchTimeout", "the launch timed out and was terminated"},
    {coolErrorLaunchIncompatibleTexturing, "coolErrorLaunchIncompatibleTexturing",
     "launch uses incompatible texturing mode"},
    {coolErrorHostMemoryAlreadyRegistered, "coolErrorHostMemoryAlreadyRegistered",
     "host memory already registered"},
    {coolErrorHostMemoryNotRegistered, "coolErrorHostMemoryNotRegistered",
     "host memory not registered"},
    {coolErrorMisalignedAddress, "coolErrorMisalignedAddress", "misaligned address"},
    {coolErrorInvalidAddressSpace, "coolErrorInvalidAddressSpace", "invalid address space"},
    {coolErrorInvalidPc, "coolErrorInvalidPc", "invalid program counter"},
    {coolErrorLaunchFailure, "coolErrorLaunchFailure", "unspecified launch failure"},
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
