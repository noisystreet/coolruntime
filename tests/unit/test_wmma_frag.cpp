// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * wmma fragment 的布局契约（C12 / MMA）。
 *
 * 纯头文件测试，不需要 CUDA 驱动。锁住三点：
 *   1. **32 字节**：m16n16k16 的 A/B/C/D 在 PTX 里都是 8 个 32 位寄存器；
 *      hacc 侧按这个大小生成 `{<2 x half> × 8}` / `{float × 8}` 的 store/load。
 *   2. **对齐 4**：hacc 目前不读类型的 aligned 属性，局部变量的对齐取自字段类型。
 *      若这里改成 `unsigned short[16]`（对齐 2），hacc 会生成 align 4 的 store
 *      打到 align 2 的槽位上 —— 未定义行为，且只在真机数值上暴露。
 *   3. **四个互不相同的类型**：出参 ABI 下 hacc 不校验 fragment 类型，
 *      「A/B/C/D 传反」只能靠 C 编译器抓，前提是四个结构体名互不相同。
 */
#include <cool/device_builtins.h>

#include <cstddef>
#include <type_traits>

#include "support/check.hpp"

namespace {

template <typename T>
void check_frag_layout(const char* name) {
    static_assert(sizeof(T) == 32, "wmma fragment 必须是 32 字节");
    static_assert(alignof(T) == 4, "wmma fragment 必须对齐 4（hacc 按 align 4 生成访存）");
    static_assert(std::is_trivially_copyable<T>::value, "fragment 必须可平凡复制");
    COOL_CHECK_EQ(sizeof(T), static_cast<std::size_t>(32));
    COOL_CHECK_EQ(alignof(T), static_cast<std::size_t>(4));
    std::printf("  %s: %zu 字节，对齐 %zu\n", name, sizeof(T), alignof(T));
}

}  // namespace

int main() {
    check_frag_layout<cool_wmma_frag_a>("cool_wmma_frag_a");
    check_frag_layout<cool_wmma_frag_b>("cool_wmma_frag_b");
    check_frag_layout<cool_wmma_frag_c>("cool_wmma_frag_c");
    check_frag_layout<cool_wmma_frag_d>("cool_wmma_frag_d");

    // 四个类型必须互不相同：相同则「传反」在 C 层也抓不到。
    COOL_CHECK((!std::is_same<cool_wmma_frag_a, cool_wmma_frag_b>::value));
    COOL_CHECK((!std::is_same<cool_wmma_frag_a, cool_wmma_frag_c>::value));
    COOL_CHECK((!std::is_same<cool_wmma_frag_a, cool_wmma_frag_d>::value));
    COOL_CHECK((!std::is_same<cool_wmma_frag_b, cool_wmma_frag_c>::value));
    COOL_CHECK((!std::is_same<cool_wmma_frag_b, cool_wmma_frag_d>::value));
    COOL_CHECK((!std::is_same<cool_wmma_frag_c, cool_wmma_frag_d>::value));

    return cooltest::summary("test_wmma_frag");
}
