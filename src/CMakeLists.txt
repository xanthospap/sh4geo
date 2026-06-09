add_library(stokes_coeffs
  stokes_coefficients.cpp
  sh_basis_cs_exterior.cpp
  sh_normalized_gradient.cpp
  sh_normalized_potential.cpp
  sh_normalized_deformation.cpp
  cunningham_weights.cpp
  normalized_legendre_factors.cpp
  load_love_numbers.cpp
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

if(PRECOMPUTED_SQRT_SHFACS)
  target_compile_definitions(stokes_coeffs PUBLIC PRECOMPUTED_SQRT_SHFACS)
endif()

target_compile_features(stokes_coeffs
  PUBLIC cxx_std_20
)