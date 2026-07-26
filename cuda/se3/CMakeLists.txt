cmake_minimum_required(VERSION 3.24)
project(jarvis_x_se3 LANGUAGES CXX CUDA)

option(JARVIS_X_SE3_FAST_MATH "Enable CUDA fast-math approximations" OFF)

add_executable(jarvis_x_se3 jarvis_x_se3.cu)
target_compile_features(jarvis_x_se3 PRIVATE cxx_std_17 cuda_std_17)
target_compile_options(
    jarvis_x_se3
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

if(JARVIS_X_SE3_FAST_MATH)
    target_compile_options(jarvis_x_se3 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()
