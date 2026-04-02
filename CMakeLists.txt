cmake_minimum_required(VERSION 3.10)
project(vectorization)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenMP REQUIRED)

# Google Highway
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
add_subdirectory(external/highway)

add_subdirectory(libs)

add_executable(vectorization src/main.cpp)

target_link_libraries(vectorization
    PUBLIC
    spmv_benchmarks
    bc_matvec
    bcsr
    OpenMP::OpenMP_CXX
    hwy
)