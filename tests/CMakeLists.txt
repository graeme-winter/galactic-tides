function(galactic_tides_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE galactic_tides)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

galactic_tides_add_test(test_params)
galactic_tides_add_test(test_octree)
galactic_tides_add_test(test_integrator)
galactic_tides_add_test(test_snapshot_io)
galactic_tides_add_test(test_hernquist_ics)
galactic_tides_add_test(test_gpu_kernels)
if(GALACTIC_TIDES_USE_METAL)
  galactic_tides_add_test(test_metal)
endif()

# The Metal kernel source compiled as C++ against a mock metal_stdlib and run
# with GPU-style threads: checks gravity_kernels.metal on any platform.
find_package(Threads REQUIRED)
add_executable(test_metal_mock test_metal_mock.cpp metal_mock_kernels.cpp)
target_link_libraries(test_metal_mock PRIVATE galactic_tides Threads::Threads)
set_source_files_properties(metal_mock_kernels.cpp PROPERTIES INCLUDE_DIRECTORIES
  "${CMAKE_CURRENT_SOURCE_DIR}/mock_metal;${PROJECT_SOURCE_DIR}/src/metal;${PROJECT_SOURCE_DIR}/include/galactic_tides/kernels")
add_test(NAME test_metal_mock COMMAND test_metal_mock)
