find_package(benchmark REQUIRED)

add_executable(bench_axpy_lwtricolwise
  bench_axpy_lwtricolwise.cpp
)
target_link_libraries(bench_axpy_lwtricolwise
  PRIVATE
    stokes_coeffs
    benchmark::benchmark
)

add_executable(bench_axpy2_lwtricolwise
  bench_axpy2_lwtricolwise.cpp
)
target_link_libraries(bench_axpy2_lwtricolwise
  PRIVATE
    stokes_coeffs
    benchmark::benchmark
)

add_executable(bench_sh2gradient
  bench_sh2gradient.cpp
)

target_link_libraries(bench_sh2gradient
  PRIVATE
    stokes_coeffs
    benchmark::benchmark
    benchmark::benchmark_main
)