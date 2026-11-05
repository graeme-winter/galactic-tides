cmake_minimum_required(VERSION 3.16)
project(galactic-tides VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(GALACTIC_TIDES_USE_OPENMP "Parallelise the tree walk with OpenMP" ON)
option(GALACTIC_TIDES_BUILD_TESTS "Build the test programs" ON)

add_library(galactic_tides
  src/particles.cpp
  src/params.cpp
  src/octree.cpp
  src/gravity.cpp
  src/integrator.cpp
  src/snapshot_io.cpp
  src/hernquist_ics.cpp
  src/gravity_backends.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/generated/kernel_source.cpp
)
target_include_directories(galactic_tides PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(galactic_tides PRIVATE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-Wall -Wextra -Wpedantic>)

# The Metal kernels are compiled at run time from source embedded in the
# library: the shared arithmetic header followed by the kernel wrappers. This
# is generated on every platform, so tests can check what would be compiled.
set(GT_KERNEL_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/galactic_tides/kernels/gravity_kernels.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/gravity_kernels.metal)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/include/galactic_tides/kernels/gravity_kernels.h GT_KERNEL_HEADER)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/gravity_kernels.metal GT_KERNEL_METAL)
configure_file(src/metal/kernel_source.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/generated/kernel_source.cpp @ONLY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${GT_KERNEL_FILES}
  ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/kernel_source.cpp.in)

# Metal: Apple only. Without it a stub keeps the API and reports clearly.
if(APPLE)
  option(GALACTIC_TIDES_USE_METAL "Build the Metal GPU gravity backends" ON)
else()
  set(GALACTIC_TIDES_USE_METAL OFF)
endif()
if(GALACTIC_TIDES_USE_METAL)
  enable_language(OBJCXX)
  target_sources(galactic_tides PRIVATE src/metal/metal_backend.mm)
  set_source_files_properties(src/metal/metal_backend.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
  target_link_libraries(galactic_tides PUBLIC "-framework Metal" "-framework Foundation")
  target_compile_definitions(galactic_tides PUBLIC GALACTIC_TIDES_HAVE_METAL=1)
  message(STATUS "Metal backends: on")

  # Optional offline syntax check of the kernels (needs the Metal compiler,
  # which comes with full Xcode rather than the command line tools).
  execute_process(COMMAND xcrun -sdk macosx -find metal
                  RESULT_VARIABLE GT_METAL_FOUND OUTPUT_QUIET ERROR_QUIET)
  if(GT_METAL_FOUND EQUAL 0)
    add_custom_target(metal_check
      COMMAND xcrun -sdk macosx metal -c
              -I ${CMAKE_CURRENT_SOURCE_DIR}/include/galactic_tides/kernels
              ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/gravity_kernels.metal
              -o ${CMAKE_CURRENT_BINARY_DIR}/gravity_kernels.air
      COMMENT "Compiling the Metal kernels offline"
      VERBATIM)
  endif()
else()
  target_sources(galactic_tides PRIVATE src/metal/metal_stub.cpp)
endif()

find_package(HDF5 REQUIRED COMPONENTS C)
target_include_directories(galactic_tides PUBLIC ${HDF5_INCLUDE_DIRS})
target_link_libraries(galactic_tides PUBLIC ${HDF5_C_LIBRARIES})
target_compile_definitions(galactic_tides PUBLIC ${HDF5_C_DEFINITIONS})

if(GALACTIC_TIDES_USE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(galactic_tides PUBLIC OpenMP::OpenMP_CXX)
  else()
    message(STATUS "OpenMP not found: building serial")
  endif()
endif()

add_executable(galactic-tides-ics apps/galactic_tides_ics.cpp)
target_link_libraries(galactic-tides-ics PRIVATE galactic_tides)
add_executable(galactic-tides-run apps/galactic_tides_run.cpp)
target_link_libraries(galactic-tides-run PRIVATE galactic_tides)
install(TARGETS galactic-tides-ics galactic-tides-run RUNTIME DESTINATION bin)

if(GALACTIC_TIDES_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
