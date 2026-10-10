// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * 后端访问器（内部头）。
 *
 * 当前仅提供 cuda 后端；生命周期的懒初始化通过 cudaBackend() 直接引用，
 * 从而在静态库场景下强制链接到后端实现。
 */
#ifndef COOL_BACKEND_REGISTRY_HPP
#define COOL_BACKEND_REGISTRY_HPP

#include "cool_backend.h"

namespace cool {

/* 返回内置的 cuda 后端描述符（静态存储，永不返回 nullptr）。 */
const coolBackend* cudaBackend();

}  // namespace cool

#endif /* COOL_BACKEND_REGISTRY_HPP */
