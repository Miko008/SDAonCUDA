# CMakeList.txt : CMake project for SDAonCUDA, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Add source to this project's executable.
add_executable (SDAonCUDA "SDAonCUDA.cpp" "SDAonCUDA.h")

# TODO: Add tests and install targets if needed.

# TinyTIFF #
include_directories("C:/Program Files (x86)/libTinyTIFF/lib/cmake/TinyTIFF")

set(TinyTIFF_DIR "C:/Program Files (x86)/libTinyTIFF/lib/cmake/TinyTIFF")

find_package(TinyTIFF REQUIRED)

target_link_libraries(SDAonCUDA TinyTIFF)

# CUDA #
enable_language(CUDA) 
FIND_PACKAGE(CUDA REQUIRED)
INCLUDE_DIRECTORIES( ${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_INCLUDE_DIRS}/../lib/x64)
