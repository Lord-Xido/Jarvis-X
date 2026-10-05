cmake_minimum_required(VERSION 3.16)
project(dm3d_rpl LANGUAGES CXX)

add_executable(dm3d_rpl dm3d_rpl.cpp)
target_compile_features(dm3d_rpl PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(dm3d_rpl PRIVATE /W4)
else()
  target_compile_options(dm3d_rpl PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()
add_test(
  NAME dm3d_source
  COMMAND dm3d_rpl run ${CMAKE_CURRENT_SOURCE_DIR}/examples/multimodal.dm3d
)
set_tests_properties(dm3d_source PROPERTIES PASS_REGULAR_EXPRESSION "\\[HALT\\]")
