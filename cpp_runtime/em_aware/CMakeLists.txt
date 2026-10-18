add_executable(jarvisx-em-aware-3d main.cpp)
jarvisx_harden(jarvisx-em-aware-3d)
add_test(NAME em-aware-3d-regressions COMMAND jarvisx-em-aware-3d --self-test)
set_tests_properties(em-aware-3d-regressions PROPERTIES TIMEOUT 30)
add_test(NAME em-aware-3d-smoke COMMAND jarvisx-em-aware-3d
    --edge 8 --iterations 40 --gamma 0.5 --tol 1e-6
    --receipt ${CMAKE_CURRENT_BINARY_DIR}/em-aware-smoke.json
    --slice ${CMAKE_CURRENT_BINARY_DIR}/em-aware-smoke.pgm)
set_tests_properties(em-aware-3d-smoke PROPERTIES TIMEOUT 30)
add_test(NAME em-aware-3d-rejects-negative-steps COMMAND jarvisx-em-aware-3d --iterations -1)
set_tests_properties(em-aware-3d-rejects-negative-steps PROPERTIES WILL_FAIL TRUE TIMEOUT 10)
