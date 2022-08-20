# CMakeList.txt : CMake project for SDAonCUDA, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)
project (SDAonCUDA LANGUAGES CUDA CXX)


# Add source to this project's executable.
add_executable (SDAonCUDA "main.cpp" "main.h" "cuda.cu")

# TODO: Add tests and install targets if needed.

# TinyTIFF #
find_package(TinyTIFF REQUIRED)
include_directories(${TinyTIFF_DIR})
target_link_libraries(SDAonCUDA TinyTIFF)

# CUDA #
enable_language(CUDA) 
find_package(CUDA REQUIRED)
include_directories( ${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_INCLUDE_DIRS}/../lib/x64)
set_target_properties(SDAonCUDA PROPERTIES CUDA_ARCHITECTURES "86")