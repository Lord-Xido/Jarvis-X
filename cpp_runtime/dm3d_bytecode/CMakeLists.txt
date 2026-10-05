cmake_minimum_required(VERSION 3.16)
project(dm3d_bytecode_runtime LANGUAGES CXX)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_executable(dm3d_vm dm3d_vm.cpp)
target_compile_features(dm3d_vm PRIVATE cxx_std_17)

if(MSVC)
  target_compile_options(dm3d_vm PRIVATE /W4)
else()
  target_compile_options(dm3d_vm PRIVATE -Wall -Wextra -Wpedantic)
endif()

set(DM3D_BYTECODE "${CMAKE_CURRENT_BINARY_DIR}/dm3d_system.bc")

add_custom_command(
  OUTPUT "${DM3D_BYTECODE}"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/assemble.py" "${DM3D_BYTECODE}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/assemble.py"
  VERBATIM
)

add_custom_target(dm3d_bytecode ALL DEPENDS "${DM3D_BYTECODE}")
add_dependencies(dm3d_vm dm3d_bytecode)

enable_testing()
add_test(NAME dm3d_vm_smoke COMMAND dm3d_vm "${DM3D_BYTECODE}")
set_tests_properties(dm3d_vm_smoke PROPERTIES
  PASS_REGULAR_EXPRESSION "\\[EOS\\]"
)
