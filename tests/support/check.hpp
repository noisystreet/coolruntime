// SPDX-FileCopyrightText: 2026 CoolRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * 极简测试断言工具（无第三方依赖）。
 *
 * 用法：
 *   COOL_CHECK(cond);
 *   COOL_CHECK_EQ(a, b);
 *   return cooltest::summary("test_name");
 */
#ifndef COOL_TESTS_SUPPORT_CHECK_HPP
#define COOL_TESTS_SUPPORT_CHECK_HPP

#include <cstdio>

namespace cooltest {

inline int& checks() {
    static int count = 0;
    return count;
}

inline int& failures() {
    static int count = 0;
    return count;
}

/* 打印结果并返回进程退出码（0 = 全部通过）。 */
inline int summary(const char* name) {
    if (failures() == 0) {
        std::printf("[PASS] %s (%d checks)\n", name, checks());
        return 0;
    }
    std::printf("[FAIL] %s (%d/%d checks failed)\n", name, failures(), checks());
    return 1;
}

}  // namespace cooltest

#define COOL_CHECK(cond)                                                                   \
    do {                                                                                   \
        ++::cooltest::checks();                                                            \
        if (!(cond)) {                                                                     \
            ++::cooltest::failures();                                                      \
            std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
        }                                                                                  \
    } while (0)

#define COOL_CHECK_EQ(a, b)                                                               \
    do {                                                                                  \
        ++::cooltest::checks();                                                           \
        if (!((a) == (b))) {                                                              \
            ++::cooltest::failures();                                                     \
            std::fprintf(stderr, "CHECK_EQ failed: %s == %s (%s:%d)\n", #a, #b, __FILE__, \
                         __LINE__);                                                       \
        }                                                                                 \
    } while (0)

#endif /* COOL_TESTS_SUPPORT_CHECK_HPP */
