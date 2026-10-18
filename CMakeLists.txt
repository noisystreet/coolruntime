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

add_library(coolruntime STATIC $<TARGET_OBJECTS:coolruntime_objects>)
add_library(coolruntime_shared SHARED $<TARGET_OBJECTS:coolruntime_objects>)
set_target_properties(coolruntime PROPERTIES OUTPUT_NAME coolrt_static)
set_target_properties(coolruntime_shared PROPERTIES OUTPUT_NAME coolrt)

add_library(COOLRuntime::coolruntime ALIAS coolruntime)
add_library(COOLRuntime::coolruntime_shared ALIAS coolruntime_shared)

foreach(_cool_lib coolruntime coolruntime_shared)
    target_include_directories(${_cool_lib}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
    # 共享库必须显式链接 libcuda（否则符号 undefined）；静态库此行仅记录依赖，不产生链接动作。
    target_link_libraries(${_cool_lib} PRIVATE ${CUDA_DRIVER_LIBRARY})
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
