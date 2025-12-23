cmake_minimum_required(VERSION 3.22)
project(urop LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(include)

add_library(core STATIC src/MCE.cpp src/Payoff.cpp src/RNG.cpp src/RunStats.cpp src/Timer.cpp src/GAsianPricer.cpp)

target_include_directories(core PUBLIC include)

add_executable(test tests/test_main.cpp)
target_link_libraries(test PRIVATE core)
