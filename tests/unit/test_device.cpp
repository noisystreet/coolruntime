// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

/* 设备管理测试：在 cuda 后端下运行（需要可用的 NVIDIA 驱动）。 */
#include "cool/cool_runtime_api.h"

#include "support/check.hpp"

int main() {
    int count = -1;
    COOL_CHECK_EQ(coolGetDeviceCount(&count), coolSuccess);
    COOL_CHECK(count >= 1);

    COOL_CHECK_EQ(coolGetDeviceCount(nullptr), coolErrorInvalidValue);

    coolDeviceProp prop{};
    COOL_CHECK_EQ(coolGetDeviceProperties(&prop, 0), coolSuccess);
    COOL_CHECK(prop.name[0] != '\0');
    COOL_CHECK(prop.maxThreadsPerBlock > 0);
    COOL_CHECK(prop.warpSize >= 1);

    COOL_CHECK_EQ(coolGetDeviceProperties(&prop, 12345), coolErrorInvalidDevice);

    COOL_CHECK_EQ(coolSetDevice(0), coolSuccess);
    int current = -1;
    COOL_CHECK_EQ(coolGetDevice(&current), coolSuccess);
    COOL_CHECK_EQ(current, 0);

    COOL_CHECK_EQ(coolSetDevice(count), coolErrorInvalidDevice);

    COOL_CHECK_EQ(coolDeviceSynchronize(), coolSuccess);
    COOL_CHECK_EQ(coolDeviceReset(), coolSuccess);

    return cooltest::summary("test_device");
}
