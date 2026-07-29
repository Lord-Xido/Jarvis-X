cmake_minimum_required(VERSION 3.16)
project(jarvis_x_autopoietic_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(JARVISX_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(JARVISX_BUILD_GL_VISUALIZER "Build the OpenGL/GLUT 3D autoencoder visualizer when dependencies are available" ON)

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

if(JARVISX_BUILD_GL_VISUALIZER)
    find_package(OpenGL QUIET)
    find_package(GLUT QUIET)

    if(OpenGL_FOUND AND GLUT_FOUND)
        add_executable(jarvisx-autoencoder3d-gl
            src/autoencoder3d_glut.cpp
        )
        target_include_directories(jarvisx-autoencoder3d-gl PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )

        if(TARGET OpenGL::GL)
            target_link_libraries(jarvisx-autoencoder3d-gl PRIVATE OpenGL::GL)
        else()
            target_include_directories(jarvisx-autoencoder3d-gl PRIVATE ${OPENGL_INCLUDE_DIR})
            target_link_libraries(jarvisx-autoencoder3d-gl PRIVATE ${OPENGL_gl_LIBRARY})
        endif()

        if(TARGET GLUT::GLUT)
            target_link_libraries(jarvisx-autoencoder3d-gl PRIVATE GLUT::GLUT)
        else()
            target_include_directories(jarvisx-autoencoder3d-gl PRIVATE ${GLUT_INCLUDE_DIRS})
            target_link_libraries(jarvisx-autoencoder3d-gl PRIVATE ${GLUT_LIBRARIES})
        endif()

        jarvisx_harden(jarvisx-autoencoder3d-gl)
    else()
        message(STATUS "OpenGL/GLUT not found; skipping jarvisx-autoencoder3d-gl")
    endif()
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
