// SPDX-FileCopyrightText: 2026 CoolRuntime contributors
// SPDX-License-Identifier: MIT

/* 版本管理测试：coolRuntimeGetVersion / coolDriverGetVersion。 */
#include "cool/cool_runtime_api.h"
#include "cool/cool_version.h"

#include "support/check.hpp"

int main() {
    int runtimeVersion = -1;
    COOL_CHECK_EQ(coolRuntimeGetVersion(&runtimeVersion), coolSuccess);
    COOL_CHECK_EQ(runtimeVersion, COOL_VERSION);

    COOL_CHECK_EQ(coolRuntimeGetVersion(nullptr), coolErrorInvalidValue);

    int driverVersion = -1;
    COOL_CHECK_EQ(coolDriverGetVersion(&driverVersion), coolSuccess);
    COOL_CHECK(driverVersion > 0);

    COOL_CHECK_EQ(coolDriverGetVersion(nullptr), coolErrorInvalidValue);

    return cooltest::summary("test_version");
}
