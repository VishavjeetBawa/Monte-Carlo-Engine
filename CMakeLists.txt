cmake_minimum_required(VERSION 3.22)
project(urop LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(include)

add_library(core STATIC src/MCE.cpp src/Payoff.cpp src/RNG.cpp src/RunStats.cpp src/Timer.cpp src/GAsianPricer.cpp)

target_include_directories(core PUBLIC include)

add_executable(test1 tests/test_OMCE.cpp)
add_executable(test2 tests/test_AVMCE.cpp)
add_executable(test3 tests/test_CMCE.cpp)
add_executable(test4 tests/test_QMCE.cpp)

target_link_libraries(test1 PRIVATE core)
target_link_libraries(test2 PRIVATE core)
target_link_libraries(test3 PRIVATE core)
target_link_libraries(test4 PRIVATE core)
