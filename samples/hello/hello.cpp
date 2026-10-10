// SPDX-FileCopyrightText: 2026 CoolRuntime contributors
// SPDX-License-Identifier: MIT

/* hello：最小可运行示例，打印版本与设备信息。 */
#include <cstdio>

#include "cool/cool_runtime.h"

int main() {
    int version = 0;
    if (coolRuntimeGetVersion(&version) == coolSuccess) {
        std::printf("CoolRuntime version: %d\n", version);
    }

    int count = 0;
    const coolError_t err = coolGetDeviceCount(&count);
    if (err != coolSuccess) {
        std::printf("coolGetDeviceCount failed: %s\n", coolGetErrorString(err));
        return 1;
    }
    std::printf("device count: %d\n", count);

    for (int i = 0; i < count; ++i) {
        coolDeviceProp prop{};
        if (coolGetDeviceProperties(&prop, i) == coolSuccess) {
            std::printf("device %d: %s (SMs=%d, warp=%d)\n", i, prop.name, prop.multiProcessorCount,
                        prop.warpSize);
        }
    }
    return 0;
}
