cmake_minimum_required(VERSION 3.16)
project(JARVISX_AE3D_V4 VERSION 4.0 LANGUAGES CXX)

add_executable(jxae3d src/jx_ae3d.cpp)
target_compile_features(jxae3d PRIVATE cxx_std_17)
if(MSVC)
  target_compile_options(jxae3d PRIVATE /W4)
else()
  target_compile_options(jxae3d PRIVATE -O2 -Wall -Wextra -Wpedantic)
endif()

enable_testing()
add_test(NAME ae3d_selftest COMMAND jxae3d selftest)
if(UNIX)
  find_program(BASH_EXE bash REQUIRED)
  add_test(NAME ae3d_end_to_end
    COMMAND "${BASH_EXE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/roundtrip.sh"
            "$<TARGET_FILE:jxae3d>" "${CMAKE_CURRENT_SOURCE_DIR}/src/jx_ae3d.cpp")
endif()
