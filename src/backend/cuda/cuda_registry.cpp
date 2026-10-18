// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

#include "backend/cuda/cuda_registry.hpp"

namespace cool {
namespace cuda {

AllocationRegistry& allocations() {
    static AllocationRegistry instance;
    return instance;
}

HandleRegistry& handles() {
    static HandleRegistry instance;
    return instance;
}

std::uintptr_t handleAddr(const void* handle) {
    return reinterpret_cast<std::uintptr_t>(handle);
}

}  // namespace cuda
}  // namespace cool
