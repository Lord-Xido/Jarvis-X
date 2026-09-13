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

add_executable(jarvisx-recursive-cube
    src/recursive_cube_main.cpp
)
set_target_properties(jarvisx-recursive-cube PROPERTIES OUTPUT_NAME "DrMoagi-Recursive-Cube")
jarvisx_include_runtime(jarvisx-recursive-cube)
jarvisx_harden(jarvisx-recursive-cube)

add_test(
    NAME recursive-cube-runtime-smoke
    COMMAND jarvisx-recursive-cube
        --tiles 4
        --levels 1
        --state-dir ${CMAKE_CURRENT_BINARY_DIR}/recursive-cube-smoke
        --quiet
)
set_tests_properties(recursive-cube-runtime-smoke PROPERTIES TIMEOUT 120)

add_executable(jarvisx-recursive-cube-tests
    tests/recursive_cube_interpreter_tests.cpp
)
jarvisx_include_runtime(jarvisx-recursive-cube-tests)
jarvisx_harden(jarvisx-recursive-cube-tests)

add_test(NAME recursive-cube-regressions COMMAND jarvisx-recursive-cube-tests)
set_tests_properties(recursive-cube-regressions PROPERTIES TIMEOUT 120)

add_executable(jarvisx-compile-interpret-accelerator
    src/compile_interpret_accelerator_main.cpp
)
set_target_properties(jarvisx-compile-interpret-accelerator PROPERTIES
    OUTPUT_NAME "DrMoagi-Compile-Interpret-Accelerator")
jarvisx_include_runtime(jarvisx-compile-interpret-accelerator)
jarvisx_harden(jarvisx-compile-interpret-accelerator)

add_test(
    NAME compile-interpret-accelerator-runtime-smoke
    COMMAND jarvisx-compile-interpret-accelerator
        --tiles 4
        --levels 1
        --state-dir ${CMAKE_CURRENT_BINARY_DIR}/compile-interpret-accelerator-smoke
        --quiet
)
set_tests_properties(compile-interpret-accelerator-runtime-smoke PROPERTIES TIMEOUT 120)

add_executable(jarvisx-compile-interpret-accelerator-tests
    tests/compile_interpret_accelerator_tests.cpp
)
jarvisx_include_runtime(jarvisx-compile-interpret-accelerator-tests)
jarvisx_harden(jarvisx-compile-interpret-accelerator-tests)

add_test(NAME compile-interpret-accelerator-regressions COMMAND jarvisx-compile-interpret-accelerator-tests)
set_tests_properties(compile-interpret-accelerator-regressions PROPERTIES TIMEOUT 120)

add_executable(jarvisx-volumetric-rom-ann
    src/volumetric_rom_ann_main.cpp
)
set_target_properties(jarvisx-volumetric-rom-ann PROPERTIES
    OUTPUT_NAME "DrMoagi-Volumetric-ROM-ANN")
jarvisx_include_runtime(jarvisx-volumetric-rom-ann)
jarvisx_harden(jarvisx-volumetric-rom-ann)

add_test(
    NAME volumetric-rom-ann-runtime-smoke
    COMMAND jarvisx-volumetric-rom-ann
        --cycles 4
        --active-tiles 8
        --quiet
)
set_tests_properties(volumetric-rom-ann-runtime-smoke PROPERTIES TIMEOUT 120)

add_executable(jarvisx-volumetric-rom-ann-tests
    tests/volumetric_rom_ann_tests.cpp
)
jarvisx_include_runtime(jarvisx-volumetric-rom-ann-tests)
jarvisx_harden(jarvisx-volumetric-rom-ann-tests)

add_test(NAME volumetric-rom-ann-regressions COMMAND jarvisx-volumetric-rom-ann-tests)
set_tests_properties(volumetric-rom-ann-regressions PROPERTIES TIMEOUT 120)
