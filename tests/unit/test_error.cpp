// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
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

    /* P6 R2：新增错误码的 name / string 必须可查（不能落到兜底文案）。 */
    COOL_CHECK(std::strcmp(coolGetErrorName(coolErrorIllegalAddress), "coolErrorIllegalAddress") ==
               0);
    COOL_CHECK(std::strcmp(coolGetErrorString(coolErrorIllegalAddress),
                           "an illegal memory access was encountered") == 0);
    COOL_CHECK(std::strcmp(coolGetErrorName(coolErrorNoDevice), "coolErrorNoDevice") == 0);
    COOL_CHECK(std::strcmp(coolGetErrorName(coolErrorLaunchFailure), "coolErrorLaunchFailure") ==
               0);
    COOL_CHECK(std::strcmp(coolGetErrorName(coolErrorInvalidConfiguration),
                           "coolErrorInvalidConfiguration") == 0);

    /* 数值锚点：新增码必须与 cudart 的 cudaError 编号一致（防抄错、防手滑改值）。 */
    COOL_CHECK_EQ(static_cast<int>(coolErrorInvalidConfiguration), 9);
    COOL_CHECK_EQ(static_cast<int>(coolErrorInvalidPitchValue), 12);
    COOL_CHECK_EQ(static_cast<int>(coolErrorInvalidSymbol), 13);
    COOL_CHECK_EQ(static_cast<int>(coolErrorInvalidDevicePointer), 17);
    COOL_CHECK_EQ(static_cast<int>(coolErrorInvalidMemcpyDirection), 21);
    COOL_CHECK_EQ(static_cast<int>(coolErrorInvalidDeviceFunction), 98);
    COOL_CHECK_EQ(static_cast<int>(coolErrorNoDevice), 100);
    COOL_CHECK_EQ(static_cast<int>(coolErrorInvalidKernelImage), 200);
    COOL_CHECK_EQ(static_cast<int>(coolErrorInvalidSource), 300);
    COOL_CHECK_EQ(static_cast<int>(coolErrorFileNotFound), 301);
    COOL_CHECK_EQ(static_cast<int>(coolErrorIllegalAddress), 700);
    COOL_CHECK_EQ(static_cast<int>(coolErrorLaunchOutOfResources), 701);
    COOL_CHECK_EQ(static_cast<int>(coolErrorLaunchTimeout), 702);
    COOL_CHECK_EQ(static_cast<int>(coolErrorMisalignedAddress), 716);
    COOL_CHECK_EQ(static_cast<int>(coolErrorInvalidAddressSpace), 717);
    COOL_CHECK_EQ(static_cast<int>(coolErrorInvalidPc), 718);
    COOL_CHECK_EQ(static_cast<int>(coolErrorLaunchFailure), 719);

    /*
     * 已知与 cudart 的数值差异（历史 ABI，本轮未改，见 cool_types.h 说明）：
     * 这里显式锁住它们，将来若真要改必须是有意为之，而不是顺手改掉。
     */
    COOL_CHECK_EQ(static_cast<int>(coolErrorInvalidDevice), 10);
    COOL_CHECK_EQ(static_cast<int>(coolErrorHostMemoryAlreadyRegistered), 711);
    COOL_CHECK_EQ(static_cast<int>(coolErrorHostMemoryNotRegistered), 712);

    /* 触发一个错误：非法设备序号。 */
    COOL_CHECK_EQ(coolSetDevice(-1), coolErrorInvalidDevice);
    COOL_CHECK_EQ(coolPeekAtLastError(), coolErrorInvalidDevice);

    /* coolGetLastError 读取后应清空。 */
    COOL_CHECK_EQ(coolGetLastError(), coolErrorInvalidDevice);
    COOL_CHECK_EQ(coolGetLastError(), coolSuccess);

    return cooltest::summary("test_error");
}
