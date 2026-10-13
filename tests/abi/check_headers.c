// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * C ABI 边界检查：公开头文件必须以 C11 独立编译。
 *
 * 本翻译单元仅参与编译（OBJECT 库），不链接，用于守住「公开层 C 兼容」约束。
 */
#include "cool/cool_runtime.h"
#include "cool/cool_runtime_api.h"
#include "cool/cool_types.h"
#include "cool/cool_version.h"
#include "cool/device_builtins.h"

int cool_abi_check_translation_unit(void) {
    coolError_t err = coolSuccess;
    coolDeviceProp prop;
    prop.maxThreadsPerBlock = (int)sizeof(coolError_t);
    return (int)err + prop.maxThreadsPerBlock + COOL_VERSION;
}
