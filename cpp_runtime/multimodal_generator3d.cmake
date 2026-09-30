add_executable(jarvisx-multimodal-generator3d
    src/multimodal_generator3d_main.cpp
)
set_target_properties(jarvisx-multimodal-generator3d PROPERTIES
    OUTPUT_NAME "DrMoagi-Multimodal-3D"
)
jarvisx_include_runtime(jarvisx-multimodal-generator3d)
jarvisx_harden(jarvisx-multimodal-generator3d)

add_test(
    NAME multimodal-generator3d-runtime-smoke
    COMMAND jarvisx-multimodal-generator3d
        --source text
        --target image
        --text "Jarvis X MM3D smoke test"
        --prompt "bounded deterministic generated geometry"
        --edge 8
        --channels 3
        --output-dir ${CMAKE_CURRENT_BINARY_DIR}/multimodal-generator3d-smoke
        --quiet
)
set_tests_properties(multimodal-generator3d-runtime-smoke PROPERTIES TIMEOUT 120)

add_executable(jarvisx-multimodal-generator3d-tests
    tests/multimodal_generator3d_tests.cpp
)
jarvisx_include_runtime(jarvisx-multimodal-generator3d-tests)
jarvisx_harden(jarvisx-multimodal-generator3d-tests)

add_test(
    NAME multimodal-generator3d-regressions
    COMMAND jarvisx-multimodal-generator3d-tests
)
set_tests_properties(multimodal-generator3d-regressions PROPERTIES TIMEOUT 120)
