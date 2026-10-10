// SPDX-FileCopyrightText: 2026 CoolRuntime contributors
// SPDX-License-Identifier: MIT

/* 流管理测试：在 cuda 后端下运行（需要可用的 NVIDIA 驱动）。 */
#include "cool/cool_runtime_api.h"

#include "support/check.hpp"

int main() {
    /* 基本创建 / 查询 / 同步 / 销毁。 */
    coolStream_t stream = nullptr;
    COOL_CHECK_EQ(coolStreamCreate(&stream), coolSuccess);
    COOL_CHECK(stream != nullptr);
    COOL_CHECK_EQ(coolStreamQuery(stream), coolSuccess);
    COOL_CHECK_EQ(coolStreamSynchronize(stream), coolSuccess);
    COOL_CHECK_EQ(coolStreamDestroy(stream), coolSuccess);

    /* 重复销毁应被句柄注册表拦截。 */
    COOL_CHECK_EQ(coolStreamDestroy(stream), coolErrorInvalidValue);

    /* 非法参数。 */
    COOL_CHECK_EQ(coolStreamCreate(nullptr), coolErrorInvalidValue);
    COOL_CHECK_EQ(coolStreamDestroy(nullptr), coolErrorInvalidValue);

    /* 默认流（NULL）：同步与查询直接透传驱动。 */
    COOL_CHECK_EQ(coolStreamSynchronize(nullptr), coolSuccess);
    COOL_CHECK_EQ(coolStreamQuery(nullptr), coolSuccess);

    /* 带标志与优先级的创建。 */
    coolStream_t nonBlocking = nullptr;
    COOL_CHECK_EQ(coolStreamCreateWithFlags(&nonBlocking, coolStreamNonBlocking), coolSuccess);
    COOL_CHECK(nonBlocking != nullptr);
    COOL_CHECK_EQ(coolStreamDestroy(nonBlocking), coolSuccess);

    coolStream_t priority = nullptr;
    COOL_CHECK_EQ(coolStreamCreateWithPriority(&priority, coolStreamDefault, 0), coolSuccess);
    COOL_CHECK(priority != nullptr);
    COOL_CHECK_EQ(coolStreamDestroy(priority), coolSuccess);

    /* streamWaitEvent：consumer 等待 producer 上已记录的事件。 */
    coolStream_t producer = nullptr;
    coolStream_t consumer = nullptr;
    coolEvent_t event = nullptr;
    COOL_CHECK_EQ(coolStreamCreate(&producer), coolSuccess);
    COOL_CHECK_EQ(coolStreamCreate(&consumer), coolSuccess);
    COOL_CHECK_EQ(coolEventCreate(&event), coolSuccess);
    COOL_CHECK_EQ(coolEventRecord(event, producer), coolSuccess);
    COOL_CHECK_EQ(coolStreamWaitEvent(consumer, event), coolSuccess);
    {
        /* 记录后事件可能已完成，故只要求不返回错误。 */
        const coolError_t q = coolEventQuery(event);
        COOL_CHECK(q == coolSuccess || q == coolErrorNotReady);
    }
    COOL_CHECK_EQ(coolStreamSynchronize(consumer), coolSuccess);
    COOL_CHECK_EQ(coolEventSynchronize(event), coolSuccess);
    COOL_CHECK_EQ(coolEventDestroy(event), coolSuccess);
    COOL_CHECK_EQ(coolStreamDestroy(consumer), coolSuccess);
    COOL_CHECK_EQ(coolStreamDestroy(producer), coolSuccess);

    /* waitEvent 的空事件参数。 */
    coolStream_t s = nullptr;
    COOL_CHECK_EQ(coolStreamCreate(&s), coolSuccess);
    COOL_CHECK_EQ(coolStreamWaitEvent(s, nullptr), coolErrorInvalidValue);
    COOL_CHECK_EQ(coolStreamDestroy(s), coolSuccess);

    return cooltest::summary("test_stream");
}
