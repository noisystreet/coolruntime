cmake_minimum_required(VERSION 3.15)

project(coolruntime
    VERSION 0.1.0
    DESCRIPTION "Teaching/research reference implementation providing a CUDA-like runtime API"
    LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# Global language + warning configuration
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

add_library(cool_warnings INTERFACE)
target_compile_options(cool_warnings INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Werror>)

# ASan/UBSan only in debug builds; header-only contract carried by an INTERFACE lib.
add_library(cool_sanitizers INTERFACE)
target_compile_options(cool_sanitizers INTERFACE
    $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-fsanitize=address,undefined;-fno-omit-frame-pointer>)
target_link_options(cool_sanitizers INTERFACE
    $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-fsanitize=address,undefined>)

# ---------------------------------------------------------------------------
# CUDA Driver API (libcuda) is a hard requirement: cuda is the only backend.
# Only the driver (cuda.h + libcuda) is needed, never the CUDA Toolkit.
# ---------------------------------------------------------------------------
find_path(CUDA_INCLUDE_DIR
    NAMES cuda.h
    HINTS "$ENV{CUDA_HOME}/include" /usr/local/cuda/include)
find_library(CUDA_DRIVER_LIBRARY NAMES cuda)

if(NOT CUDA_INCLUDE_DIR OR NOT CUDA_DRIVER_LIBRARY)
    message(FATAL_ERROR
        "COOLRuntime requires the CUDA Driver API: cuda.h and libcuda were not found. "
        "Install an NVIDIA driver (no CUDA Toolkit needed) or set CUDA_HOME.")
endif()

message(STATUS "COOLRuntime: cuda backend (include: ${CUDA_INCLUDE_DIR}, "
               "driver: ${CUDA_DRIVER_LIBRARY})")

# ---------------------------------------------------------------------------
# Library targets: libcoolrt_static.a + libcoolrt.so
#
# 产物命名对齐 CUDA（libcudart_static.a / libcudart.so）：静态库带 `_static` 后缀，
# 共享库用裸名。target 名保留 coolruntime 作为内部标识，测试与 samples 无需改动。
# 静态库与共享库共用同一份对象（OBJECT 库），不重复编译。
# ---------------------------------------------------------------------------
set(COOLRUNTIME_SOURCES
    src/api/error_api.cpp
    src/api/version_api.cpp
    src/api/device_api.cpp
    src/api/mem_api.cpp
    src/api/stream_api.cpp
    src/api/event_api.cpp
    src/api/kernel_api.cpp
    src/core/error.cpp
    src/core/lifecycle.cpp
    src/backend/bootstrap.cpp
    src/backend/cuda/cuda_backend.cpp)

add_library(coolruntime_objects OBJECT ${COOLRUNTIME_SOURCES})
# 共享库需要 PIC；静态库复用同一份对象，故统一开启（x86-64 上代价可忽略）。
set_target_properties(coolruntime_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(coolruntime_objects
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CUDA_INCLUDE_DIR})
target_link_libraries(coolruntime_objects PRIVATE cool_warnings)

# bootstrap 的一次性初始化用 std::mutex / std::atomic，需要线程库。
find_package(Threads REQUIRED)

add_library(coolruntime STATIC $<TARGET_OBJECTS:coolruntime_objects>)
add_library(coolruntime_shared SHARED $<TARGET_OBJECTS:coolruntime_objects>)
set_target_properties(coolruntime PROPERTIES OUTPUT_NAME coolrt_static)
set_target_properties(coolruntime_shared PROPERTIES
    OUTPUT_NAME coolrt
    # 共享库带 soname（libcoolrt.so.0），对齐 CUDA 的 libcudart.so.12；
    # 裸名 libcoolrt.so 只是开发包里的链接，供 -lcoolrt 使用。
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})

add_library(COOLRuntime::coolruntime ALIAS coolruntime)
add_library(COOLRuntime::coolruntime_shared ALIAS coolruntime_shared)

foreach(_cool_lib coolruntime coolruntime_shared)
    target_include_directories(${_cool_lib}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
    # 共享库必须显式链接 libcuda（否则符号 undefined）；静态库此行仅记录依赖，不产生链接动作。
    target_link_libraries(${_cool_lib} PRIVATE ${CUDA_DRIVER_LIBRARY})
    # backend 层使用 std::mutex / std::atomic，对使用方是 PUBLIC 依赖。
    target_link_libraries(${_cool_lib} PUBLIC Threads::Threads)
endforeach()

# ---------------------------------------------------------------------------
# C ABI boundary check: the public headers must compile standalone as C11.
# ---------------------------------------------------------------------------
add_library(cool_c_abi_check OBJECT tests/abi/check_headers.c)
target_include_directories(cool_c_abi_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(cool_c_abi_check PRIVATE
    $<$<C_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Werror>)

enable_testing()

add_subdirectory(tests)
add_subdirectory(samples)

# ---------------------------------------------------------------------------
# Install + Debian packaging
#
# 按 Debian 惯例拆成两个包（与 cudart 的分发方式对齐）：
#   libcoolrt0    —— 运行时：libcoolrt.so.<soname>
#   libcoolrt-dev —— 开发：头文件 + libcoolrt_static.a + libcoolrt.so 链接 + coolrt.pc
#
# 打包入口：`make package`（Release 构建，独立 build-package/ 目录）。
# 不要用 Debug 目录打包——Debug 编译带 -fsanitize=address,undefined，
# 打进 deb 会让库在用户机器上强依赖 libasan。
# ---------------------------------------------------------------------------
# 布局对齐 CUDA（`/usr/local/cuda/{include,lib64}`）：默认前缀 `/usr/local/cool`，
# 库目录固定为 lib64（而非 lib 或 lib/<multiarch>）。
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "/usr/local/cool" CACHE PATH "Install prefix" FORCE)
endif()
include(GNUInstallDirs)
set(CMAKE_INSTALL_LIBDIR "lib64")

# soname 实体（libcoolrt.so.0）进运行时包；裸名链接（libcoolrt.so）进开发包。
install(TARGETS coolruntime_shared
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime
            NAMELINK_COMPONENT devel)
install(TARGETS coolruntime
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT devel)

configure_file(cmake/coolrt.pc.in ${CMAKE_CURRENT_BINARY_DIR}/coolrt.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/coolrt.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig COMPONENT devel)

set(CPACK_GENERATOR "DEB")
# CPack 的打包前缀默认硬编码为 /usr，必须显式跟随 install 前缀，
# 否则 /usr/local/cool 会被截成 /usr（deb 内路径与 pkg-config 不一致）。
set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
set(CPACK_PACKAGE_NAME "coolrt")
set(CPACK_PACKAGE_VENDOR "COOLRuntime")
set(CPACK_PACKAGE_CONTACT "noisystreet <noisystreet@users.noreply.github.com>")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "noisystreet <noisystreet@users.noreply.github.com>")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "COOLRuntime: CUDA-like runtime API (teaching / reference implementation)")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/noisystreet/coolruntime")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")

set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_COMPONENTS_ALL runtime devel)
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)

# 运行时包：包名遵循 Debian「lib + soname」惯例（libcoolrt0）。
# libcuda.so.1 由 NVIDIA 驱动提供、不在 Debian 仓库里，故不写进 Depends（与 cudart 一致），
# 只在开发包里以 Recommends 提示需要驱动。
set(CPACK_DEBIAN_RUNTIME_PACKAGE_NAME "libcoolrt0")
set(CPACK_DEBIAN_RUNTIME_PACKAGE_SECTION "libs")
set(CPACK_DEBIAN_RUNTIME_PACKAGE_SHLIBDEPS OFF)
set(CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS "libc6 (>= 2.34)")
set(CPACK_DEBIAN_RUNTIME_PACKAGE_RECOMMENDS "nvidia-driver | cuda-drivers")

set(CPACK_DEBIAN_DEVEL_PACKAGE_NAME "libcoolrt-dev")
set(CPACK_DEBIAN_DEVEL_PACKAGE_SECTION "libdevel")
set(CPACK_DEBIAN_DEVEL_PACKAGE_SHLIBDEPS OFF)
set(CPACK_DEBIAN_DEVEL_PACKAGE_DEPENDS "libcoolrt0 (= ${PROJECT_VERSION})")

include(CPack)
