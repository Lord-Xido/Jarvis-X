option(JARVISX_OMNI_ENABLE_OPENMP "Use OpenMP for the self-verifying inward engine when available" ON)
if(JARVISX_OMNI_ENABLE_OPENMP)
    find_package(OpenMP QUIET COMPONENTS CXX)
endif()

foreach(target IN ITEMS jarvisx-self-verifying-omni jarvisx-self-verifying-omni-tests)
    if(target STREQUAL "jarvisx-self-verifying-omni")
        add_executable(${target} src/self_verifying_omni_main.cpp)
    else()
        add_executable(${target} tests/self_verifying_omni_tests.cpp)
    endif()
    jarvisx_include_runtime(${target})
    jarvisx_harden(${target})
    if(JARVISX_OMNI_ENABLE_OPENMP AND TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endforeach()

add_test(NAME self-verifying-omni-regressions COMMAND jarvisx-self-verifying-omni-tests)
add_test(NAME self-verifying-omni-million-smoke COMMAND jarvisx-self-verifying-omni --threads 2 --json)
add_test(NAME self-verifying-omni-convergence-smoke COMMAND jarvisx-self-verifying-omni
    --pathways 257 --depth 1024 --threads 2 --require-convergence --json)
add_test(NAME self-verifying-omni-invalid-input COMMAND jarvisx-self-verifying-omni --pathways 0)
add_test(NAME self-verifying-omni-incomplete-convergence COMMAND jarvisx-self-verifying-omni
    --pathways 257 --depth 16 --threads 2 --require-convergence)
set_tests_properties(self-verifying-omni-invalid-input self-verifying-omni-incomplete-convergence
    PROPERTIES WILL_FAIL TRUE)
set_tests_properties(self-verifying-omni-regressions self-verifying-omni-million-smoke
    self-verifying-omni-convergence-smoke self-verifying-omni-invalid-input
    self-verifying-omni-incomplete-convergence PROPERTIES TIMEOUT 120)
