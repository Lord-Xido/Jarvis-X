cmake_minimum_required(VERSION 3.16)
project(jarvis_x_autopoietic_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(JARVISX_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(JARVISX_BUILD_GL_VISUALIZER "Build the OpenGL/GLUT ANN visualizers when dependencies are available" ON)

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

function(jarvisx_include_runtime target)
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
endfunction()

add_executable(jarvisx-runtime
    src/main.cpp
)
jarvisx_include_runtime(jarvisx-runtime)
jarvisx_harden(jarvisx-runtime)

add_executable(jarvisx-autoencoder3d
    src/autoencoder3d_main.cpp
)
jarvisx_include_runtime(jarvisx-autoencoder3d)
jarvisx_harden(jarvisx-autoencoder3d)

add_executable(jarvisx-multimedia4d
    src/multimedia4d_main.cpp
)
jarvisx_include_runtime(jarvisx-multimedia4d)
jarvisx_harden(jarvisx-multimedia4d)

add_executable(jarvisx-riof3d
    src/riof3d_main.cpp
)
jarvisx_include_runtime(jarvisx-riof3d)
jarvisx_harden(jarvisx-riof3d)

if(JARVISX_BUILD_GL_VISUALIZER)
    find_package(OpenGL QUIET)
    find_package(GLUT QUIET)

    if(OpenGL_FOUND AND GLUT_FOUND)
        function(jarvisx_link_glut target)
            jarvisx_include_runtime(${target})

            if(TARGET OpenGL::GL)
                target_link_libraries(${target} PRIVATE OpenGL::GL)
            else()
                target_include_directories(${target} PRIVATE ${OPENGL_INCLUDE_DIR})
                target_link_libraries(${target} PRIVATE ${OPENGL_gl_LIBRARY})
            endif()

            if(TARGET GLUT::GLUT)
                target_link_libraries(${target} PRIVATE GLUT::GLUT)
            else()
                target_include_directories(${target} PRIVATE ${GLUT_INCLUDE_DIRS})
                target_link_libraries(${target} PRIVATE ${GLUT_LIBRARIES})
            endif()

            jarvisx_harden(${target})
        endfunction()

        add_executable(jarvisx-autoencoder3d-gl
            src/autoencoder3d_glut.cpp
        )
        jarvisx_link_glut(jarvisx-autoencoder3d-gl)

        add_executable(jarvisx-multimedia4d-gl
            src/multimedia4d_glut.cpp
        )
        jarvisx_link_glut(jarvisx-multimedia4d-gl)
    else()
        message(STATUS "OpenGL/GLUT not found; skipping ANN visualizers")
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

add_test(
    NAME multimedia4d-runtime-smoke
    COMMAND jarvisx-multimedia4d
        --cycles 2
        --edge 8
        --channels 3
        --temporal-depth 3
        --output-dir ${CMAKE_CURRENT_BINARY_DIR}/multimedia4d-smoke
        --quiet
)
set_tests_properties(multimedia4d-runtime-smoke PROPERTIES TIMEOUT 120)

add_test(
    NAME riof3d-runtime-smoke
    COMMAND jarvisx-riof3d
        --cycles 4
        --edge 8
        --pattern sphere
        --quiet
)
set_tests_properties(riof3d-runtime-smoke PROPERTIES TIMEOUT 120)

add_executable(jarvisx-processor-tests
    tests/processor_tests.cpp
)
jarvisx_include_runtime(jarvisx-processor-tests)
jarvisx_harden(jarvisx-processor-tests)

add_test(NAME processor-regressions COMMAND jarvisx-processor-tests)
set_tests_properties(processor-regressions PROPERTIES TIMEOUT 120)

add_executable(jarvisx-autoencoder3d-tests
    tests/autoencoder3d_tests.cpp
)
jarvisx_include_runtime(jarvisx-autoencoder3d-tests)
jarvisx_harden(jarvisx-autoencoder3d-tests)

add_test(NAME autoencoder3d-regressions COMMAND jarvisx-autoencoder3d-tests)
set_tests_properties(autoencoder3d-regressions PROPERTIES TIMEOUT 120)

add_executable(jarvisx-multimedia4d-tests
    tests/multimedia4d_tests.cpp
)
jarvisx_include_runtime(jarvisx-multimedia4d-tests)
jarvisx_harden(jarvisx-multimedia4d-tests)

add_test(NAME multimedia4d-regressions COMMAND jarvisx-multimedia4d-tests)
set_tests_properties(multimedia4d-regressions PROPERTIES TIMEOUT 120)

add_executable(jarvisx-riof3d-tests
    tests/riof3d_tests.cpp
)
jarvisx_include_runtime(jarvisx-riof3d-tests)
jarvisx_harden(jarvisx-riof3d-tests)

add_test(NAME riof3d-regressions COMMAND jarvisx-riof3d-tests)
set_tests_properties(riof3d-regressions PROPERTIES TIMEOUT 120)

add_executable(jarvisx-riof-permeation-tests
    tests/riof_permeation_tests.cpp
)
jarvisx_include_runtime(jarvisx-riof-permeation-tests)
jarvisx_harden(jarvisx-riof-permeation-tests)

add_test(NAME riof-permeation-regressions COMMAND jarvisx-riof-permeation-tests)
set_tests_properties(riof-permeation-regressions PROPERTIES TIMEOUT 120)
