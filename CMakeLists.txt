cmake_minimum_required(VERSION 3.10)
project(vectorization)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenMP REQUIRED)

add_subdirectory(libs)

add_executable(vectorization src/main.cpp)

# Specify libraries or flags to use when linking a given target and/or its dependents. 
# Usage requirements from linked library targets will be propagated. 
# Usage requirements of a target's dependencies affect compilation of its own sources
target_link_libraries(vectorization
    PUBLIC
    spmv_benchmarks
    bc_matvec
    bcsr
    OpenMP::OpenMP_CXX
) # where are the dependencies for the main executable