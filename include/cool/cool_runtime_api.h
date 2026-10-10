// SPDX-FileCopyrightText: 2026 CoolRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * CoolRuntime 公开 API（A 层）。
 *
 * 与 CUDA Runtime API（cudart）命名一一对应：cudaXxx -> coolXxx。
 * 本头文件保持 C 兼容，可被 C、C++ 及 CUDA 示例直接 include。
 *
 * 阶段 P0 覆盖：版本管理（6.7）、错误处理（6.1）、设备管理（6.2）。
 */
#ifndef COOL_RUNTIME_API_H
#define COOL_RUNTIME_API_H

#include "cool_types.h"
#include "cool_version.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* 版本管理（对应 cudaRuntimeGetVersion / cudaDriverGetVersion）        */
/* ------------------------------------------------------------------ */

/* 返回运行时版本号（COOL_VERSION 组合值）。 */
coolError_t coolRuntimeGetVersion(int* runtimeVersion);

/* 返回驱动版本号（cuda 后端对应 cuDriverGetVersion）。 */
coolError_t coolDriverGetVersion(int* driverVersion);

/* ------------------------------------------------------------------ */
/* 错误处理（对应 cudaGetErrorString 等）                              */
/* ------------------------------------------------------------------ */

/* 返回错误码的可读字符串。 */
const char* coolGetErrorString(coolError_t error);

/* 返回错误码的枚举名。 */
const char* coolGetErrorName(coolError_t error);

/* 返回并清空线程局部 last error。 */
coolError_t coolGetLastError(void);

/* 返回但不清空线程局部 last error。 */
coolError_t coolPeekAtLastError(void);

/* ------------------------------------------------------------------ */
/* 设备管理（对应 cudaGetDeviceCount 等）                              */
/* ------------------------------------------------------------------ */

/* 返回逻辑设备数。 */
coolError_t coolGetDeviceCount(int* count);

/* 设置调用线程当前设备；非法设备返回 coolErrorInvalidDevice。 */
coolError_t coolSetDevice(int device);

/* 获取调用线程当前设备。 */
coolError_t coolGetDevice(int* device);

/* 填充简化版设备属性。 */
coolError_t coolGetDeviceProperties(coolDeviceProp* prop, int device);

/* 等待该设备所有流完成。 */
coolError_t coolDeviceSynchronize(void);

/* 释放运行时持有的该设备资源。 */
coolError_t coolDeviceReset(void);

/*
 * 可选：把 cudart 名称映射到 cool 名称，便于把小型 CUDA 示例
 * 源码直接改跑到 CoolRuntime 上（教学对照用途）。
 */
#ifdef COOL_ENABLE_CUDA_ALIAS
#define cudaRuntimeGetVersion coolRuntimeGetVersion
#define cudaDriverGetVersion coolDriverGetVersion
#define cudaGetErrorString coolGetErrorString
#define cudaGetErrorName coolGetErrorName
#define cudaGetLastError coolGetLastError
#define cudaPeekAtLastError coolPeekAtLastError
#define cudaGetDeviceCount coolGetDeviceCount
#define cudaSetDevice coolSetDevice
#define cudaGetDevice coolGetDevice
#define cudaGetDeviceProperties coolGetDeviceProperties
#define cudaDeviceSynchronize coolDeviceSynchronize
#define cudaDeviceReset coolDeviceReset
#endif /* COOL_ENABLE_CUDA_ALIAS */

#ifdef __cplusplus
}
#endif

#endif /* COOL_RUNTIME_API_H */
