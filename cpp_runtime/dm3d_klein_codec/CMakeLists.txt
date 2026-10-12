cmake_minimum_required(VERSION 3.16)
project(jarvisx_dm3d_klein_codec LANGUAGES CXX)

add_executable(jarvisx-dm3d-klein-codec dr_moagi_3d_engine.cpp)
target_compile_features(jarvisx-dm3d-klein-codec PRIVATE cxx_std_17)
set_target_properties(jarvisx-dm3d-klein-codec PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
  target_compile_options(jarvisx-dm3d-klein-codec PRIVATE /W4 /permissive-)
else()
  target_compile_options(jarvisx-dm3d-klein-codec PRIVATE -Wall -Wextra -Wpedantic)
endif()

find_package(OpenMP QUIET COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
  target_link_libraries(jarvisx-dm3d-klein-codec PRIVATE OpenMP::OpenMP_CXX)
endif()

enable_testing()
add_test(NAME dm3d-klein-codec-self-test COMMAND jarvisx-dm3d-klein-codec --self-test)
add_test(NAME dm3d-klein-codec-smoke COMMAND jarvisx-dm3d-klein-codec --grid 16 --steps 8 --eta 0.38 --beta 0.86 --mu 0.05)
set_tests_properties(dm3d-klein-codec-self-test dm3d-klein-codec-smoke PROPERTIES TIMEOUT 120)
