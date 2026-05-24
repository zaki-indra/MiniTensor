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
    minitensor/core/Storage.cpp
    minitensor/core/Allocator.cpp
    minitensor/core/utilities.cpp
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

    include(GoogleTest)
    gtest_discover_tests(minitensor_tests)
else()
    message(WARNING "No test files found in the tests/ directory matching test_*.cpp")
endif()