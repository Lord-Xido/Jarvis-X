# Bounded 8K UHD toroidal framebuffer pipeline (CPU reference).
add_executable(jarvisx-8k-inward-framebuffer
    src/streaming_framebuffer3d_main.cpp
)
set_target_properties(jarvisx-8k-inward-framebuffer PROPERTIES
    OUTPUT_NAME "DrMoagi-8K-Inward-Framebuffer"
)
jarvisx_include_runtime(jarvisx-8k-inward-framebuffer)
jarvisx_harden(jarvisx-8k-inward-framebuffer)

add_executable(jarvisx-8k-inward-framebuffer-tests
    tests/streaming_framebuffer3d_tests.cpp
)
jarvisx_include_runtime(jarvisx-8k-inward-framebuffer-tests)
jarvisx_harden(jarvisx-8k-inward-framebuffer-tests)
add_test(NAME 8k-inward-framebuffer-regressions COMMAND jarvisx-8k-inward-framebuffer-tests)
set_tests_properties(8k-inward-framebuffer-regressions PROPERTIES TIMEOUT 30)
add_test(NAME 8k-inward-framebuffer-smoke
    COMMAND jarvisx-8k-inward-framebuffer --width 128 --height 96 --frames 2)
set_tests_properties(8k-inward-framebuffer-smoke PROPERTIES TIMEOUT 30)
