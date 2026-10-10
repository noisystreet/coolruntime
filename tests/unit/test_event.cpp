// SPDX-FileCopyrightText: 2026 CoolRuntime contributors
// SPDX-License-Identifier: MIT

/* 事件管理测试：在 cuda 后端下运行（需要可用的 NVIDIA 驱动）。 */
#include "cool/cool_runtime_api.h"

#include "support/check.hpp"

int main() {
    /* 基本创建 / 记录 / 同步 / 查询 / 销毁；NULL 表示默认流。 */
    coolEvent_t event = nullptr;
    COOL_CHECK_EQ(coolEventCreate(&event), coolSuccess);
    COOL_CHECK(event != nullptr);
    COOL_CHECK_EQ(coolEventRecord(event, nullptr), coolSuccess);
    COOL_CHECK_EQ(coolEventSynchronize(event), coolSuccess);
    {
        /* 记录后事件可能已完成，故只要求不返回错误。 */
        const coolError_t q = coolEventQuery(event);
        COOL_CHECK(q == coolSuccess || q == coolErrorNotReady);
    }
    COOL_CHECK_EQ(coolEventDestroy(event), coolSuccess);

    /* 重复销毁应被句柄注册表拦截。 */
    COOL_CHECK_EQ(coolEventDestroy(event), coolErrorInvalidValue);

    /* 非法参数。 */
    COOL_CHECK_EQ(coolEventCreate(nullptr), coolErrorInvalidValue);
    COOL_CHECK_EQ(coolEventDestroy(nullptr), coolErrorInvalidValue);
    COOL_CHECK_EQ(coolEventRecord(nullptr, nullptr), coolErrorInvalidValue);
    COOL_CHECK_EQ(coolEventSynchronize(nullptr), coolErrorInvalidValue);
    COOL_CHECK_EQ(coolEventQuery(nullptr), coolErrorInvalidValue);

    /* 带标志创建（禁用计时）。 */
    coolEvent_t noTiming = nullptr;
    COOL_CHECK_EQ(coolEventCreateWithFlags(&noTiming, coolEventDisableTiming), coolSuccess);
    COOL_CHECK(noTiming != nullptr);
    COOL_CHECK_EQ(coolEventRecord(noTiming, nullptr), coolSuccess);
    COOL_CHECK_EQ(coolEventSynchronize(noTiming), coolSuccess);
    COOL_CHECK_EQ(coolEventDestroy(noTiming), coolSuccess);

    /* elapsedTime：同一流上两个事件的间隔应为非负。 */
    coolStream_t stream = nullptr;
    coolEvent_t start = nullptr;
    coolEvent_t end = nullptr;
    COOL_CHECK_EQ(coolStreamCreate(&stream), coolSuccess);
    COOL_CHECK_EQ(coolEventCreate(&start), coolSuccess);
    COOL_CHECK_EQ(coolEventCreate(&end), coolSuccess);
    COOL_CHECK_EQ(coolEventRecord(start, stream), coolSuccess);
    COOL_CHECK_EQ(coolEventRecord(end, stream), coolSuccess);
    COOL_CHECK_EQ(coolEventSynchronize(end), coolSuccess);

    float ms = -1.0f;
    COOL_CHECK_EQ(coolEventElapsedTime(&ms, start, end), coolSuccess);
    COOL_CHECK(ms >= 0.0f);

    /* elapsedTime 的空指针参数。 */
    COOL_CHECK_EQ(coolEventElapsedTime(nullptr, start, end), coolErrorInvalidValue);
    COOL_CHECK_EQ(coolEventElapsedTime(&ms, nullptr, end), coolErrorInvalidValue);
    COOL_CHECK_EQ(coolEventElapsedTime(&ms, start, nullptr), coolErrorInvalidValue);

    COOL_CHECK_EQ(coolEventDestroy(end), coolSuccess);
    COOL_CHECK_EQ(coolEventDestroy(start), coolSuccess);
    COOL_CHECK_EQ(coolStreamDestroy(stream), coolSuccess);

    return cooltest::summary("test_event");
}
