cmake_minimum_required(VERSION 3.16)
project(moagi_3d_virtual_1e24 LANGUAGES CXX)

add_executable(moagi_3d_virtual_code_runtime moagi_3d_virtual_code_runtime.cpp)
target_compile_features(moagi_3d_virtual_code_runtime PRIVATE cxx_std_17)
set_target_properties(moagi_3d_virtual_code_runtime PROPERTIES CXX_EXTENSIONS OFF)
if(MSVC)
  target_compile_options(moagi_3d_virtual_code_runtime PRIVATE /W4 /WX)
else()
  target_compile_options(moagi_3d_virtual_code_runtime PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

enable_testing()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME virtual_1e24_address_and_regeneration
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_runtime.py"
          "$<TARGET_FILE:moagi_3d_virtual_code_runtime>" "${CMAKE_CXX_COMPILER}")
set_tests_properties(virtual_1e24_address_and_regeneration PROPERTIES TIMEOUT 120)
