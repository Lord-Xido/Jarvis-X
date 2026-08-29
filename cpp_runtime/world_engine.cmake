add_executable(jarvisx-world-engine
    src/world_engine_main.cpp
)
set_target_properties(jarvisx-world-engine PROPERTIES OUTPUT_NAME "DrMoagi-World-Engine")
jarvisx_include_runtime(jarvisx-world-engine)
jarvisx_harden(jarvisx-world-engine)

add_test(
    NAME world-engine-runtime-smoke
    COMMAND jarvisx-world-engine
        --state-dir ${CMAKE_CURRENT_BINARY_DIR}/world-engine-smoke
        --quiet
)
set_tests_properties(world-engine-runtime-smoke PROPERTIES TIMEOUT 120)

add_executable(jarvisx-world-engine-tests
    tests/world_engine_vmad_tests.cpp
)
jarvisx_include_runtime(jarvisx-world-engine-tests)
jarvisx_harden(jarvisx-world-engine-tests)

add_test(NAME world-engine-regressions COMMAND jarvisx-world-engine-tests)
set_tests_properties(world-engine-regressions PROPERTIES TIMEOUT 120)
