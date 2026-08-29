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

add_library(jarvisx-symmetry-loop3d-core STATIC
    src/symmetry_loop3d.cpp
)
target_include_directories(jarvisx-symmetry-loop3d-core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
jarvisx_selfedit_harden(jarvisx-symmetry-loop3d-core)

add_executable(jarvisx-symmetry-loop3d
    src/symmetry_loop3d_main.cpp
)
target_link_libraries(jarvisx-symmetry-loop3d PRIVATE jarvisx-symmetry-loop3d-core)
jarvisx_selfedit_harden(jarvisx-symmetry-loop3d)

add_library(jarvisx-symmetry-benchmark3d-core STATIC
    src/symmetry_benchmark3d.cpp
)
target_include_directories(jarvisx-symmetry-benchmark3d-core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(jarvisx-symmetry-benchmark3d-core PUBLIC jarvisx-symmetry-loop3d-core)
jarvisx_selfedit_harden(jarvisx-symmetry-benchmark3d-core)

add_executable(jarvisx-symmetry-benchmark3d
    src/symmetry_benchmark3d_main.cpp
)
target_link_libraries(jarvisx-symmetry-benchmark3d PRIVATE jarvisx-symmetry-benchmark3d-core)
jarvisx_selfedit_harden(jarvisx-symmetry-benchmark3d)

enable_testing()

add_executable(jarvisx-self-editor3d-tests
    tests/self_editor3d_tests.cpp
)
target_link_libraries(jarvisx-self-editor3d-tests PRIVATE jarvisx-self-editor3d-core)
jarvisx_selfedit_harden(jarvisx-self-editor3d-tests)

add_test(NAME self-editor3d-regressions COMMAND jarvisx-self-editor3d-tests)
set_tests_properties(self-editor3d-regressions PROPERTIES TIMEOUT 120)

add_executable(jarvisx-symmetry-loop3d-tests
    tests/symmetry_loop3d_tests.cpp
)
target_link_libraries(jarvisx-symmetry-loop3d-tests PRIVATE jarvisx-symmetry-loop3d-core)
jarvisx_selfedit_harden(jarvisx-symmetry-loop3d-tests)

add_test(NAME symmetry-loop3d-regressions COMMAND jarvisx-symmetry-loop3d-tests)
set_tests_properties(symmetry-loop3d-regressions PROPERTIES TIMEOUT 120)

add_executable(jarvisx-symmetry-benchmark3d-tests
    tests/symmetry_benchmark3d_tests.cpp
)
target_link_libraries(jarvisx-symmetry-benchmark3d-tests PRIVATE jarvisx-symmetry-benchmark3d-core)
jarvisx_selfedit_harden(jarvisx-symmetry-benchmark3d-tests)

add_test(NAME symmetry-benchmark3d-regressions COMMAND jarvisx-symmetry-benchmark3d-tests)
set_tests_properties(symmetry-benchmark3d-regressions PROPERTIES TIMEOUT 120)
