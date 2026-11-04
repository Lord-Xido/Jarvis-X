cmake_minimum_required(VERSION 3.16)
project(jarvisx_permeate3d LANGUAGES CXX)

add_executable(jarvisx-permeate3d jarvisx_permeate.cpp)
target_compile_features(jarvisx-permeate3d PRIVATE cxx_std_20)
set_target_properties(jarvisx-permeate3d PROPERTIES CXX_EXTENSIONS OFF)
if(MSVC)
    target_compile_options(jarvisx-permeate3d PRIVATE /W4 /permissive-)
else()
    target_compile_options(jarvisx-permeate3d PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME permeate3d-smoke
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/smoke_test.py"
            "$<TARGET_FILE:jarvisx-permeate3d>")
set_tests_properties(permeate3d-smoke PROPERTIES TIMEOUT 120)
