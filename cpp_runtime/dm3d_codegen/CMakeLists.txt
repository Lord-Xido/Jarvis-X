cmake_minimum_required(VERSION 3.16)
project(jarvis_x_dm3d_codegen LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(dm3d_codegen dm3d_codegen.cpp)
if(MSVC)
  target_compile_options(dm3d_codegen PRIVATE /O2)
else()
  target_compile_options(dm3d_codegen PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()
