add_executable(jarvisx-moagi3d-engine
    src/moagi3d_engine_main.cpp
)
set_target_properties(jarvisx-moagi3d-engine PROPERTIES OUTPUT_NAME "DrMoagi-3D-Engine")
jarvisx_include_runtime(jarvisx-moagi3d-engine)
jarvisx_harden(jarvisx-moagi3d-engine)

add_test(
    NAME moagi3d-engine-smoke
    COMMAND jarvisx-moagi3d-engine
        --edge 8
        --channels 3
        --latent-channels 3
        --max-iterations 4
        --epsilon 0.00001
        --pattern sphere
        --quiet
)
set_tests_properties(moagi3d-engine-smoke PROPERTIES TIMEOUT 120)

add_executable(jarvisx-moagi3d-engine-tests
    tests/moagi3d_engine_tests.cpp
)
jarvisx_include_runtime(jarvisx-moagi3d-engine-tests)
jarvisx_harden(jarvisx-moagi3d-engine-tests)

add_test(NAME moagi3d-engine-regressions COMMAND jarvisx-moagi3d-engine-tests)
set_tests_properties(moagi3d-engine-regressions PROPERTIES TIMEOUT 120)
