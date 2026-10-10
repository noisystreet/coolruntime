// SPDX-FileCopyrightText: 2026 CoolRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * CoolRuntime 版本宏。
 *
 * 库版本 = 编译期宏（本文件）；运行时版本 = coolRuntimeGetVersion()。
 * 与 cudart 的 CUDA_VERSION 编码习惯保持一致：MAJOR*1000 + MINOR*10 + PATCH。
 */
#ifndef COOL_VERSION_H
#define COOL_VERSION_H

#define COOL_VERSION_MAJOR 0
#define COOL_VERSION_MINOR 1
#define COOL_VERSION_PATCH 0

#define COOL_VERSION (COOL_VERSION_MAJOR * 1000 + COOL_VERSION_MINOR * 10 + COOL_VERSION_PATCH)

#endif /* COOL_VERSION_H */
