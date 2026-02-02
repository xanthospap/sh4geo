cmake_minimum_required(VERSION 3.16)

project(stokes_coeffs LANGUAGES CXX)

# ---- Build settings ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optional: nicer warnings
if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---- Library target ----
add_library(stokes_coeffs STATIC
  src/stokes_coefficients.cpp
)

target_include_directories(stokes_coeffs
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ---- Tests ----
include(CTest)  # defines BUILD_TESTING option and enables testing infra
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
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
endif()

