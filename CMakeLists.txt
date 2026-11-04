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
)
target_include_directories(galactic_tides PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(galactic_tides PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>)

if(GALACTIC_TIDES_USE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(galactic_tides PUBLIC OpenMP::OpenMP_CXX)
  else()
    message(STATUS "OpenMP not found: building serial")
  endif()
endif()

if(GALACTIC_TIDES_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
