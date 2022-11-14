# CMakeList.txt : CMake project for SDAonCUDA, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)
project (${PROJECT_NAME} LANGUAGES CUDA CXX)


# Add source to this project's executable.
add_executable (${PROJECT_NAME} "main.cpp" "main.h" "cuda.cu")

# TODO: Add tests and install targets if needed.

# TinyTIFF #
find_package(TinyTIFF REQUIRED)
include_directories(${TinyTIFF_DIR})
target_link_libraries(${PROJECT_NAME} TinyTIFF)

# CUDA #
enable_language(CUDA) 
find_package(CUDA REQUIRED)
include_directories( ${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_INCLUDE_DIRS}/../lib/x64)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "86")

# Sweet #

find_package(Git REQUIRED)
if(GIT_FOUND AND EXISTS "$PROJECT_SOURCE_DIR}/.git")
	option(GIT_SUBMODULE "Check submodules during build" ON)
	if(GIT_SUBMODULE)
		message(STATUS "Submodule update")
		execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
						WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
						RESULT_VARIABLE GIT_SUBMOD_RESULT)
		if(NOT GIT_SUBMOD_RESULT EQUAL "0")
			message(FATAL_ERROR "git submodule update failed with ${GIT_SUBMOD_RESULT}")
		endif()
	endif()
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sweet/CMakeLists.txt")
	message(FATAL_ERROR "Sweet.hpp was not cloned! GIT_SUBMODULE was turned off or failed.") 
endif()

add_subdirectory(sweet)
target_include_directories(${PROJECT_NAME} PUBLIC sweet)