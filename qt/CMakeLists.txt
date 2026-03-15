find_package(Qt6 REQUIRED COMPONENTS Widgets)

set(CMAKE_AUTOMOC ON)

add_executable(qt_app
    main.cpp
    Window.cpp
    Window.hpp
)

target_link_libraries(qt_app
    Qt6::Widgets
    core
    cpu_mc
    cuda_mc
)
