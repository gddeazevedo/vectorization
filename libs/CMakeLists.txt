# bcsr
add_library(bcsr
    bcsr/bcsr.cpp
)
target_include_directories(bcsr PUBLIC bcsr) # where are the headers for bcsr

# Add a library to the project using the specified source files
# bc_matvec
add_library(bc_matvec
    bc_matvec/bc_matvec.cpp
)
target_include_directories(bc_matvec PUBLIC bc_matvec) # where are the headers for bc_matvec
target_link_libraries(bc_matvec PUBLIC bcsr) # where are the dependencies for bc_matvec

# benchmarks/spmv
add_library(spmv_benchmarks
    benchmarks/spmv/spmv_benchmarks.cpp
)
target_include_directories(spmv_benchmarks PUBLIC benchmarks/spmv) # where are the headers for spmv_benchmarks
target_link_libraries(spmv_benchmarks PUBLIC bcsr bc_matvec) # where are the dependencies for spmv_benchmarks