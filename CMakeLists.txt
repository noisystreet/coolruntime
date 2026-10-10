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
        "CoolRuntime requires the CUDA Driver API: cuda.h and libcuda were not found. "
        "Install an NVIDIA driver (no CUDA Toolkit needed) or set CUDA_HOME.")
endif()

message(STATUS "CoolRuntime: cuda backend (include: ${CUDA_INCLUDE_DIR}, "
               "driver: ${CUDA_DRIVER_LIBRARY})")

# ---------------------------------------------------------------------------
# Library target: libcoolruntime.a
# ---------------------------------------------------------------------------
set(COOLRUNTIME_SOURCES
    src/api/error_api.cpp
    src/api/version_api.cpp
    src/api/device_api.cpp
    src/core/error.cpp
    src/core/lifecycle.cpp
    src/backend/cuda/cuda_backend.cpp)

add_library(coolruntime STATIC ${COOLRUNTIME_SOURCES})
add_library(CoolRuntime::coolruntime ALIAS coolruntime)

target_include_directories(coolruntime
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CUDA_INCLUDE_DIR})

target_link_libraries(coolruntime PRIVATE cool_warnings)
target_link_libraries(coolruntime PRIVATE ${CUDA_DRIVER_LIBRARY})

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
