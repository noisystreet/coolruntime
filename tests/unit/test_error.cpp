// SPDX-FileCopyrightText: 2026 CoolRuntime contributors
// SPDX-License-Identifier: MIT

/* 错误处理测试：错误字符串、last error 的读取/清空语义。 */
#include <cstring>

#include "cool/cool_runtime_api.h"

#include "support/check.hpp"

int main() {
    /* 初始状态：无待处理错误。 */
    COOL_CHECK_EQ(coolPeekAtLastError(), coolSuccess);

    /* 枚举名与可读文本。 */
    COOL_CHECK(std::strcmp(coolGetErrorName(coolErrorInvalidDevice), "coolErrorInvalidDevice") ==
               0);
    COOL_CHECK(std::strcmp(coolGetErrorString(coolErrorOutOfMemory), "out of memory") == 0);
    COOL_CHECK(std::strcmp(coolGetErrorName(coolErrorUnknown), "coolErrorUnknown") == 0);

    /* 触发一个错误：非法设备序号。 */
    COOL_CHECK_EQ(coolSetDevice(-1), coolErrorInvalidDevice);
    COOL_CHECK_EQ(coolPeekAtLastError(), coolErrorInvalidDevice);

    /* coolGetLastError 读取后应清空。 */
    COOL_CHECK_EQ(coolGetLastError(), coolErrorInvalidDevice);
    COOL_CHECK_EQ(coolGetLastError(), coolSuccess);

    return cooltest::summary("test_error");
}
