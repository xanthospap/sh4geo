file(GLOB TEST_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/unit_test/*.cpp
)

foreach(test_src IN LISTS TEST_SOURCES)
  get_filename_component(test_name ${test_src} NAME_WE)

  add_executable(${test_name} ${test_src})
  target_link_libraries(${test_name} PRIVATE stokes_coeffs)

  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()