# Standalone DM3D 4:1 / 1:4 Mandelbulb lossless reference codec.
add_executable(jarvisx-mandelbulb4 src/mandelbulb4_main.cpp)
jarvisx_harden(jarvisx-mandelbulb4)

add_test(NAME mandelbulb4-self-test COMMAND jarvisx-mandelbulb4 --self-test)
set_tests_properties(mandelbulb4-self-test PROPERTIES TIMEOUT 60)

add_test(
    NAME mandelbulb4-end-to-end
    COMMAND jarvisx-mandelbulb4
        --side 16 --iterations 18
        --force-hierarchy
        --out ${CMAKE_CURRENT_BINARY_DIR}/mandelbulb4-ci.mb4
        --decoded ${CMAKE_CURRENT_BINARY_DIR}/mandelbulb4-ci.raw
)
set_tests_properties(mandelbulb4-end-to-end PROPERTIES TIMEOUT 60)
