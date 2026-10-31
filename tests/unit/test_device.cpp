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

    /* P6 R1：coolDeviceGetAttribute 与 coolGetDeviceProperties 必须同源同值。 */
    int attr = -1;
    COOL_CHECK_EQ(coolDeviceGetAttribute(&attr, coolDevAttrMaxThreadsPerBlock, 0), coolSuccess);
    COOL_CHECK_EQ(attr, prop.maxThreadsPerBlock);

    COOL_CHECK_EQ(coolDeviceGetAttribute(&attr, coolDevAttrWarpSize, 0), coolSuccess);
    COOL_CHECK_EQ(attr, prop.warpSize);

    COOL_CHECK_EQ(coolDeviceGetAttribute(&attr, coolDevAttrMaxBlockDimX, 0), coolSuccess);
    COOL_CHECK_EQ(attr, prop.maxThreadsDim[0]);

    COOL_CHECK_EQ(coolDeviceGetAttribute(&attr, coolDevAttrComputeCapabilityMajor, 0), coolSuccess);
    COOL_CHECK_EQ(attr, prop.major);

    COOL_CHECK_EQ(coolDeviceGetAttribute(&attr, coolDevAttrMultiProcessorCount, 0), coolSuccess);
    COOL_CHECK_EQ(attr, prop.multiProcessorCount);

    /* 新增字段必须真被填上：块共享内存与 L2 在真机上不可能为 0。 */
    COOL_CHECK(prop.sharedMemPerBlock > 0);
    COOL_CHECK(prop.maxThreadsDim[0] > 0);
    COOL_CHECK(prop.l2CacheSize > 0);

    /* 未收录的属性 → NotSupported，且不得写 *value（区别于「查到 0」）。 */
    attr = -12345;
    COOL_CHECK_EQ(coolDeviceGetAttribute(&attr, static_cast<coolDeviceAttribute>(9999), 0),
                  coolErrorNotSupported);
    COOL_CHECK_EQ(attr, -12345);

    /* 非法设备序号 → InvalidDevice。 */
    COOL_CHECK_EQ(coolDeviceGetAttribute(&attr, coolDevAttrWarpSize, 12345),
                  coolErrorInvalidDevice);

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
