cmake_minimum_required(VERSION 3.16)
project(jarvisx_self_editor3d LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(JARVISX_SELFEDIT_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

function(jarvisx_selfedit_harden target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
        if(JARVISX_SELFEDIT_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(${target} PRIVATE -fsanitize=address,undefined)
        endif()
    endif()
endfunction()

add_library(jarvisx-self-editor3d-core STATIC
    src/self_editor3d.cpp
)
target_include_directories(jarvisx-self-editor3d-core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
jarvisx_selfedit_harden(jarvisx-self-editor3d-core)

add_executable(jarvisx-self-editor3d
    src/main.cpp
)
target_link_libraries(jarvisx-self-editor3d PRIVATE jarvisx-self-editor3d-core)
jarvisx_selfedit_harden(jarvisx-self-editor3d)

enable_testing()

add_executable(jarvisx-self-editor3d-tests
    tests/self_editor3d_tests.cpp
)
target_link_libraries(jarvisx-self-editor3d-tests PRIVATE jarvisx-self-editor3d-core)
jarvisx_selfedit_harden(jarvisx-self-editor3d-tests)

add_test(NAME self-editor3d-regressions COMMAND jarvisx-self-editor3d-tests)
set_tests_properties(self-editor3d-regressions PROPERTIES TIMEOUT 120)
