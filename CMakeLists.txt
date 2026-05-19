cmake_minimum_required(VERSION 4.0)
project(minitensor VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 1. Main Library Target ---
add_library(minitensor
    minitensor/core/Tensor.cpp
    minitensor/core/TensorImpl.cpp
)

# Enforce our clean Pitchfork layout boundaries
target_include_directories(minitensor
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/minitensor
)

# --- 2. Examples ---
add_executable(create_example example/create.cpp)
target_link_libraries(create_example PRIVATE minitensor)

# --- 3. Unit Tests (Using GoogleTest) ---
enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows MSVC compatibility
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(minitensor_tests tests/main.cpp)

# GTest::gtest_main automatically provides the int main() for us!
target_link_libraries(minitensor_tests
    PRIVATE
        minitensor
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(minitensor_tests)