# ************************* STANDARDS *************************
cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-std=c++17 -W")
set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 80 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(
        barco
        VERSION 1.0
        DESCRIPTION "BARCO Test"
        LANGUAGES CXX CUDA
)


# ************************* VARIABLES *************************

set(EXECUTABLE_NAME "barco")
set(SHARED_LIBRARY_NAME "libcuda_transformations")

# Set the output directory for the executable
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

# Set the output directory for the version.h file
set(VERSION_HEADER_OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Set the output directory for the shared library
set(LIBRARY_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/lib")

# Path to the 'utils' directory
set(UTILS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

# Path to the 'utils' directory
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Path to the 'src' directory
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

# Path to the 'src' directory
set(CUDA_INCLUDE_DIR "/usr/local/cuda/include")

message(STATUS "CMake version: ${CMAKE_VERSION}")
message(STATUS "Executable: '${EXECUTABLE_NAME}' is built in: '${CMAKE_CURRENT_SOURCE_DIR}'")
message(STATUS "Library Path: '${LIBRARY_OUTPUT_PATH}'")

# ************************* CUDA *************************

# Add the CUDA source file
add_library(
        ${SHARED_LIBRARY_NAME}
        SHARED
        ${SRC_DIR}/cuda_transformations.cu
        ${SRC_DIR}/cuda_utils.cu
)

# Add the include directory for the CUDA source file
target_include_directories(${SHARED_LIBRARY_NAME} PUBLIC ${INCLUDE_DIR})

# Set the output name of the shared library
set_target_properties(${SHARED_LIBRARY_NAME} PROPERTIES
        PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
)

# ************************* MAIN *************************

# Add the executable
add_executable(
        ${EXECUTABLE_NAME}
        main.cpp
        include/version.h
        src/Tint_Transformation.cpp
        src/Colour_Space_Transformation.cpp
        include/cxxopts.hpp

)

# ************* INCLUDE *************

# Add the working directory
target_include_directories(${EXECUTABLE_NAME} PUBLIC "${PROJECT_BINARY_DIR}" "${UTILS_DIR}" "${INCLUDE_DIR}" "${CUDA_INCLUDE_DIR}")

configure_file(include/version.h.in "${VERSION_HEADER_OUTPUT}/version.h")

# ************ PACKAGES ************

find_package(OpenCV REQUIRED)
target_include_directories(${EXECUTABLE_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${EXECUTABLE_NAME} ${OpenCV_LIBS})

# Link the shared library with the executable
target_link_libraries(${EXECUTABLE_NAME} ${LIBRARY_OUTPUT_PATH}/${SHARED_LIBRARY_NAME}.so)
