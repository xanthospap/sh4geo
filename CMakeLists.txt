cmake_minimum_required(VERSION 3.16)

project(stokes_coeffs LANGUAGES CXX)

# ---- Build settings ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Options ----
option(DSO_SIMD "Enable SIMD kernels" ON)
option(ENABLE_SANITIZERS "Enable Address/Undefined sanitizers (Debug only)" OFF)
option(ENABLE_BENCHMARKS "Build benchmarks" ON)

# ---- Common compile options ----
if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(
    -Wall -Wextra -Werror -pedantic -W -Wshadow
    $<$<CONFIG:Release>:-O2>
    $<$<CONFIG:Release>:-march=native>
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Debug>:-Wdisabled-optimization>
    $<$<CONFIG:RelWithDebInfo>:-fno-omit-frame-pointer>
    $<$<CONFIG:RelWithDebInfo>:-g>
    $<$<CONFIG:RelWithDebInfo>:-O2>
    $<$<CONFIG:RelWithDebInfo>:-march=native>
  )
endif()

add_compile_definitions($<$<CONFIG:Debug>:DEBUG>)

if(ENABLE_SANITIZERS)
  add_compile_options(
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
  )
  add_link_options(
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>
  )
endif()

# ---- Library target ----
add_library(stokes_coeffs STATIC
  src/stokes_coefficients.cpp
)

target_include_directories(stokes_coeffs
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ---- SIMD flags ----
if(DSO_SIMD)
  target_compile_definitions(stokes_coeffs PUBLIC DSO_SIMD)

  # PUBLIC because templates / headers may contain SIMD code
  if(MSVC)
    target_compile_options(stokes_coeffs PUBLIC /arch:AVX2)
  else()
    target_compile_options(stokes_coeffs PUBLIC -mavx2 -mfma)
  endif()
endif()

# ---- Tests ----
include(CTest)
if(BUILD_TESTING)
  enable_testing()

  file(GLOB UNIT_TEST_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/test/unit_test/*.cpp
  )

  if(NOT UNIT_TEST_SOURCES)
    message(WARNING "No unit tests found in test/unit_test/*.cpp")
  endif()

  foreach(test_src ${UNIT_TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)

    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE stokes_coeffs)

    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
endif()

# ---- Benchmarks ----
if(ENABLE_BENCHMARKS)
  find_package(benchmark REQUIRED)

  add_executable(bench_axpy_lwtricolwise
    bench/bench_axpy_lwtricolwise.cpp
  )

  add_executable(bench_axpy2_lwtricolwise
    bench/bench_axpy2_lwtricolwise.cpp
  )

  foreach(bench_target
      bench_axpy_lwtricolwise
      bench_axpy2_lwtricolwise)
    target_link_libraries(${bench_target}
      PRIVATE
        stokes_coeffs
        benchmark::benchmark
    )
  endforeach()
endif()