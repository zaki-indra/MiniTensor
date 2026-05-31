cmake_minimum_required(VERSION 4.0)
project(minitensor VERSION 0.0.1 LANGUAGES CXX)

# ─────────────────────────────────────────────
# Build Profiles & Compiler Flags
# ─────────────────────────────────────────────
# Default to Debug if no build type is specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
    message(STATUS "Setting build type to 'Debug' as none was specified.")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set platform-specific optimization and debug flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-O0 -g -Wall -Wextra)
    else()
        add_compile_options(-O3 -DNDEBUG -march=native)
    endif()
elseif(MSVC)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/Od /Zi /W4)
    else()
        add_compile_options(/O2 /DNDEBUG)
    endif()
endif()

# ─────────────────────────────────────────────
# 1. Main Library Target
# ─────────────────────────────────────────────
add_library(minitensor
    minitensor/core/Array.cpp
    minitensor/core/ArrayIterator.cpp
    minitensor/core/Storage.cpp
    minitensor/core/IO.cpp
    minitensor/ops/Elementwise.cpp
    minitensor/ops/Math.cpp
    minitensor/ops/Initializers.cpp
)

target_include_directories(minitensor
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/minitensor
)

# ─────────────────────────────────────────────
# 2. Examples
# ─────────────────────────────────────────────
add_executable(create_example example/create.cpp)
target_link_libraries(create_example PRIVATE minitensor)

# ─────────────────────────────────────────────
# 3. Unit Tests (Using GoogleTest)
# ─────────────────────────────────────────────
enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows MSVC compatibility
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# CONFIGURE_DEPENDS ensures CMake re-runs automatically if you add a new test file!
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "tests/test_*.cpp")

if(TEST_SOURCES)
    add_executable(minitensor_tests ${TEST_SOURCES})

    target_link_libraries(minitensor_tests
        PRIVATE
            minitensor
            GTest::gtest_main
    )

    # Tests are part of the project and may include internal headers
    # (e.g. core/ArrayIterator.hpp). Mirrors the library's own PRIVATE
    # include path.
    target_include_directories(minitensor_tests
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/minitensor
    )

    include(GoogleTest)
    gtest_discover_tests(minitensor_tests)
else()
    message(WARNING "No test files found in the tests/ directory matching test_*.cpp")
endif()

# ─────────────────────────────────────────────
# 4. Benchmarks (Using Google Benchmark)
# ─────────────────────────────────────────────
# Opt-in: the common build/test path should not pay for fetching and
# compiling Google Benchmark. Enable with -DMT_BUILD_BENCHMARKS=ON.
option(MT_BUILD_BENCHMARKS "Build the MiniTensor benchmark suite" OFF)

if(MT_BUILD_BENCHMARKS)
    # Benchmarks measure optimized code. A Debug (-O0) build reports noise.
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
        message(WARNING
            "MT_BUILD_BENCHMARKS is ON but CMAKE_BUILD_TYPE is '${CMAKE_BUILD_TYPE}'. "
            "Benchmark numbers are only meaningful in a Release build "
            "(configure with -DCMAKE_BUILD_TYPE=Release).")
    endif()

    # Google Benchmark pulls in its own test suite and install rules by
    # default; we want neither.
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        benchmark
        URL https://github.com/google/benchmark/archive/refs/tags/v1.8.5.zip
    )
    FetchContent_MakeAvailable(benchmark)

    file(GLOB_RECURSE BENCHMARK_SOURCES CONFIGURE_DEPENDS "benchmarks/bench_*.cpp")

    if(BENCHMARK_SOURCES)
        add_executable(minitensor_benchmarks ${BENCHMARK_SOURCES})

        target_link_libraries(minitensor_benchmarks
            PRIVATE
                minitensor
                benchmark::benchmark_main
        )

        # Benchmarks may reach into internal headers, same as tests.
        target_include_directories(minitensor_benchmarks
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/minitensor
        )
    else()
        message(WARNING "MT_BUILD_BENCHMARKS is ON but no benchmarks/bench_*.cpp files were found.")
    endif()
endif()