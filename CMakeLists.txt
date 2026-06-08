cmake_minimum_required(VERSION 3.20)

project(sh4geo VERSION 0.1.0 LANGUAGES CXX)

# ---- Options ----
option(DSO_SIMD "Enable SIMD kernels" ON)
option(ENABLE_SANITIZERS "Enable Address/Undefined sanitizers (Debug only)" OFF)
option(ENABLE_TESTING "Build unit tests" ON)
option(ENABLE_BENCHMARKS "Build benchmarks" OFF)

# ---- C++ standard ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Global compile options ----
add_compile_options(
  -Wall -Wextra -Werror -pedantic -Wshadow
  $<$<CONFIG:Release>:-O2>
  $<$<CONFIG:Release>:-march=native>
  $<$<CONFIG:Debug>:-g>
  $<$<CONFIG:Debug>:-Wdisabled-optimization>
  $<$<CONFIG:RelWithDebInfo>:-fno-omit-frame-pointer>
  $<$<CONFIG:RelWithDebInfo>:-g>
  $<$<CONFIG:RelWithDebInfo>:-O2>
  $<$<CONFIG:RelWithDebInfo>:-march=native>
)

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

# ---- Dependencies ----
find_package(Eigen3 REQUIRED)

# ---- Subdirectories ----
add_subdirectory(src)

if(ENABLE_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()

if(ENABLE_BENCHMARKS)
  add_subdirectory(bench)
endif()