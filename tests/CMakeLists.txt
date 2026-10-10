# 单元测试：每个源文件构建一个可执行文件，并注册到 CTest。
function(cool_add_unit_test name)
    add_executable(${name} unit/${name}.cpp)
    target_link_libraries(${name} PRIVATE coolruntime)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

cool_add_unit_test(test_version)
cool_add_unit_test(test_error)
cool_add_unit_test(test_device)

# 全部测试在唯一的 cuda 后端下运行，需要可用的 NVIDIA 驱动。
add_test(NAME test_version COMMAND test_version)
add_test(NAME test_error COMMAND test_error)
add_test(NAME test_device COMMAND test_device)
