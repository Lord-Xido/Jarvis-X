add_executable(jarvisx-master-equation
    src/master_equation_main.cpp
)
jarvisx_include_runtime(jarvisx-master-equation)
jarvisx_harden(jarvisx-master-equation)

add_executable(jarvisx-master-equation-tests
    tests/master_equation_tests.cpp
)
jarvisx_include_runtime(jarvisx-master-equation-tests)
jarvisx_harden(jarvisx-master-equation-tests)

add_test(
    NAME dr-moagi-master-equation-smoke
    COMMAND jarvisx-master-equation
)
set_tests_properties(dr-moagi-master-equation-smoke PROPERTIES TIMEOUT 30)

add_test(
    NAME dr-moagi-master-equation-regressions
    COMMAND jarvisx-master-equation-tests
)
set_tests_properties(dr-moagi-master-equation-regressions PROPERTIES TIMEOUT 30)
