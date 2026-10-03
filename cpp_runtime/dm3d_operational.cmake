option(JARVISX_DM3D_ENABLE_OPENMP "Use OpenMP for the DM3D operational engine when available" ON)

if(JARVISX_DM3D_ENABLE_OPENMP)
    find_package(OpenMP QUIET COMPONENTS CXX)
endif()

add_executable(jarvisx-dm3d-operational
    src/dm3d_operational_main.cpp
)
set_target_properties(jarvisx-dm3d-operational PROPERTIES
    OUTPUT_NAME "DrMoagi-DM3D-Operational"
)
jarvisx_include_runtime(jarvisx-dm3d-operational)
jarvisx_harden(jarvisx-dm3d-operational)

if(OpenMP_CXX_FOUND)
    target_link_libraries(jarvisx-dm3d-operational PRIVATE OpenMP::OpenMP_CXX)
endif()

add_test(
    NAME dm3d-operational-self-test
    COMMAND jarvisx-dm3d-operational --self-test
)
set_tests_properties(dm3d-operational-self-test PROPERTIES TIMEOUT 120)
