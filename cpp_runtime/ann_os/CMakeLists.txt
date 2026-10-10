cmake_minimum_required(VERSION 3.16)
project(jarvisx_ann_os LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(ANN_OS_SANITIZERS "Enable ASan and UBSan on supported compilers" OFF)

add_executable(jarvisx-ann-os ann_os.cpp)
add_executable(jarvisx-ann-os-tests tests.cpp)
set_target_properties(jarvisx-ann-os PROPERTIES OUTPUT_NAME ann_os)
set_target_properties(jarvisx-ann-os-tests PROPERTIES OUTPUT_NAME ann_os_tests)

foreach(target IN ITEMS jarvisx-ann-os jarvisx-ann-os-tests)
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  if(COMMAND jarvisx_harden)
    jarvisx_harden(${target})
  elseif(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive-)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  endif()
  if(ANN_OS_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  endif()
endforeach()

# Regression tests read ann_os.cpp relative to their working directory.
# Keep source files pristine: copy fixtures into the build directory.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/ann_os.cpp"
               "${CMAKE_CURRENT_BINARY_DIR}/ann_os.cpp" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/ann.hpp"
               "${CMAKE_CURRENT_BINARY_DIR}/ann.hpp" COPYONLY)

enable_testing()
add_test(NAME ann-os-regressions COMMAND jarvisx-ann-os-tests)
add_test(NAME ann-os-batch-smoke COMMAND jarvisx-ann-os --self ann_os.cpp --batch)
foreach(t IN ITEMS ann-os-regressions ann-os-batch-smoke)
  set_tests_properties(${t} PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    TIMEOUT 120)
endforeach()
