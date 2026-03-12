# bcsr
add_library(bcsr
    bcsr/bcsr.cpp
)
target_include_directories(bcsr PUBLIC bcsr)

# bc_matvec
add_library(bc_matvec
    bc_matvec/bc_matvec.cpp
)
target_include_directories(bc_matvec PUBLIC bc_matvec)
target_link_libraries(bc_matvec PUBLIC bcsr)

# benchmarks/spmv
add_library(spmv_benchmarks
    benchmarks/spmv/spmv_benchmarks.cpp
)
target_include_directories(spmv_benchmarks PUBLIC benchmarks/spmv)
target_link_libraries(spmv_benchmarks PUBLIC bcsr bc_matvec)