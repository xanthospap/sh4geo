add_library(stokes_coeffs
  stokes_coefficients.cpp
  sh_basis_cs_exterior.cpp
  sh_cunningham_normalized_gradient.cpp
  normalized_legendre_factors.cpp
)

target_include_directories(stokes_coeffs
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(stokes_coeffs
  PUBLIC
    Eigen3::Eigen
)

if(DSO_SIMD)
  target_compile_definitions(stokes_coeffs PUBLIC DSO_SIMD)
  target_compile_options(stokes_coeffs PUBLIC -mavx2 -mfma)
endif()

target_compile_features(stokes_coeffs
  PUBLIC cxx_std_20
)

if(DSO_SIMD)
  target_compile_definitions(stokes_coeffs PUBLIC DSO_SIMD)
endif()