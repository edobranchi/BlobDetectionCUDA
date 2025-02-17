cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(BlobDetectionCUDA LANGUAGES CUDA CXX)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV configuration
#set(OpenCV_DIR ~/opencv_install/lib/cmake/opencv4)
#include_directories(${OpenCV_INCLUDE_DIRS}/opencv2/core)
find_package(OpenCV REQUIRED COMPONENTS
        core
        highgui
        cudaimgproc
        cudaarithm
        cudafilters
        cudaimgproc
        cudaoptflow
        cudawarping
)

# CUDA configuration
find_package(CUDAToolkit REQUIRED)  # Use modern CMake CUDA finding

# Add executable
add_executable(BlobDetectionCUDA main.cu)

# Set CUDA specific properties
set_target_properties(BlobDetectionCUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# Modern way to link CUDA libraries
target_link_libraries(BlobDetectionCUDA PRIVATE
        ${OpenCV_LIBS}
        CUDA::cudart
        CUDA::cuda_driver
)
set_property(TARGET BlobDetectionCUDA PROPERTY CUDA_ARCHITECTURES 86)

# Include directories using modern CMake
target_include_directories(BlobDetectionCUDA PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
)