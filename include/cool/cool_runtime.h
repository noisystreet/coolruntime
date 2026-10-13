// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * COOLRuntime 统一入口头（umbrella header）。
 *
 * 对应 cudart 的 cuda_runtime.h：聚合公开类型、版本宏与运行时 API，
 * 使用者通常只需 #include "cool_runtime.h"。
 * 本头文件保持 C 兼容，可被 C、C++ 及 CUDA 示例直接 include。
 */
#ifndef COOL_RUNTIME_H
#define COOL_RUNTIME_H

#include "cool_runtime_api.h"
#include "cool_types.h"
#include "cool_version.h"
#include "device_builtins.h"

#endif /* COOL_RUNTIME_H */
