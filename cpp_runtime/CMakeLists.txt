cmake_minimum_required(VERSION 3.16)
project(jarvis_x_autopoietic_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(jarvisx-runtime
    src/jarvis_x_autopoietic_runtime.cpp
)

if(MSVC)
    target_compile_options(jarvisx-runtime PRIVATE /W4 /permissive-)
else()
    target_compile_options(jarvisx-runtime PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()
add_test(
    NAME inward-runtime-smoke
    COMMAND jarvisx-runtime
        --generations 2
        --population 3
        --state-dir ${CMAKE_CURRENT_BINARY_DIR}/smoke-state
        --text "Jarvis X inward runtime smoke test"
        --quiet
        --reset
)
set_tests_properties(inward-runtime-smoke PROPERTIES TIMEOUT 120)
