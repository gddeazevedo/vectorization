cmake_minimum_required(VERSION 3.10)
project(vectorization)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenMP REQUIRED)

add_subdirectory(libs)

add_executable(vectorization src/main.cpp)

target_link_libraries(vectorization
    PUBLIC
    spmv_benchmarks
    bc_matvec
    bcsr
    OpenMP::OpenMP_CXX
)