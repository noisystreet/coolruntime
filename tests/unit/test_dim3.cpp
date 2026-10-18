// dim3 的构造语义与 ABI 约束（B5：dim3 归属 coolrt，hacc 只消费）。
//
// 纯头文件测试，不需要 CUDA 驱动：锁住两点——
//   1. CUDA 兼容的构造用法（dim3(x) / (x,y) / (x,y,z)，缺省维为 1）；
//   2. ABI 不变（12 字节、对齐 4、可平凡复制），因为 coolLaunchKernel 按值传递。
#include <cool/cool_types.h>

#include <cstddef>
#include <cstdio>
#include <type_traits>

namespace {

int g_failures = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

// 断言 dim3 的三维等于期望值，失败时打印实际值便于定位。
void check_dim3(const dim3& d, unsigned int ex, unsigned int ey, unsigned int ez,
                const char* what) {
    if (d.x != ex || d.y != ey || d.z != ez) {
        std::fprintf(stderr, "FAIL: %s（实际 (%u,%u,%u)，期望 (%u,%u,%u)）\n", what, d.x, d.y, d.z,
                     ex, ey, ez);
        ++g_failures;
    }
}

}  // namespace

int main() {
    // CUDA 语义：默认各维为 1（而不是未初始化）。
    check_dim3(dim3(), 1, 1, 1, "dim3 默认构造应为 (1,1,1)");

    // 缺省维补 1，对齐 CUDA 的默认参数。
    check_dim3(dim3(256), 256, 1, 1, "dim3(256) 应为 (256,1,1)");
    check_dim3(dim3(16, 16), 16, 16, 1, "dim3(16,16) 应为 (16,16,1)");
    check_dim3(dim3(1, 2, 3), 1, 2, 3, "dim3(1,2,3)");

    // 花括号写法必须继续成立：hacc 的 to_dim3 用 `dim3{v, 1, 1}`，
    // 此时是列表初始化走构造函数（而不是聚合初始化）。
    check_dim3(dim3{1, 2, 3}, 1, 2, 3, "dim3{1,2,3} 应走构造函数");
    const dim3 copied = {4, 5, 6};
    check_dim3(copied, 4, 5, 6, "dim3 = {4,5,6} 拷贝列表初始化");

    // ABI：coolLaunchKernel(kernel, dim3 grid, dim3 block, ...) 按值传递。
    check(sizeof(dim3) == 3 * sizeof(unsigned int), "dim3 应为 12 字节");
    check(alignof(dim3) == alignof(unsigned int), "dim3 对齐应为 4");
    check(std::is_trivially_copyable<dim3>::value, "dim3 必须可平凡复制");

    // 记录事实：带用户提供的构造函数后 dim3 不再是聚合类型。
    // 若哪天有人去掉构造函数，这条会失败，提醒同步复查 `dim3{}` 相关契约。
    check(!std::is_aggregate<dim3>::value, "dim3 带构造函数后不再是聚合类型");

    if (g_failures == 0) {
        std::printf("dim3 OK\n");
    }
    return g_failures == 0 ? 0 : 1;
}
