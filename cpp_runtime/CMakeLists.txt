cmake_minimum_required(VERSION 3.16)
project(jarvis_x_autopoietic_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(JARVISX_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

function(jarvisx_harden target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
        if(JARVISX_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(${target} PRIVATE -fsanitize=address,undefined)
        endif()
    endif()
endfunction()

add_executable(jarvisx-runtime
    src/main.cpp
)
target_include_directories(jarvisx-runtime PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
jarvisx_harden(jarvisx-runtime)

add_executable(jarvisx-autoencoder3d
    src/autoencoder3d_main.cpp
)
target_include_directories(jarvisx-autoencoder3d PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
jarvisx_harden(jarvisx-autoencoder3d)

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

add_executable(jarvisx-processor-tests
    tests/processor_tests.cpp
)
target_include_directories(jarvisx-processor-tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
jarvisx_harden(jarvisx-processor-tests)

add_test(NAME processor-regressions COMMAND jarvisx-processor-tests)
set_tests_properties(processor-regressions PROPERTIES TIMEOUT 120)

add_executable(jarvisx-autoencoder3d-tests
    tests/autoencoder3d_tests.cpp
)
target_include_directories(jarvisx-autoencoder3d-tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
jarvisx_harden(jarvisx-autoencoder3d-tests)

add_test(NAME autoencoder3d-regressions COMMAND jarvisx-autoencoder3d-tests)
set_tests_properties(autoencoder3d-regressions PROPERTIES TIMEOUT 120)
