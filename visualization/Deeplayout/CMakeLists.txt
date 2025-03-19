cmake_minimum_required(VERSION 3.14)

# Set the project name
project(Deeplayout)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the Qt libraries
set(CMAKE_PREFIX_PATH /opt/homebrew/opt/qt)

find_package(Qt6 REQUIRED COMPONENTS Widgets Core Gui)

# Enable AUTOMOC
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Add the include directories
include_directories(${Qt6Widgets_INCLUDE_DIRS} ${Qt6Core_INCLUDE_DIRS} ${Qt6Gui_INCLUDE_DIRS})

# Add the source files
set(SOURCES
    deeplayout.cpp
    main.cpp
    polygon.cpp
    structure.cpp
    GeneratedFiles/qrc_deeplayout.cpp
)

# Add the header files
set(HEADERS
    deeplayout.h
    polygon.h
    structure.h
    GeneratedFiles/ui_deeplayout.h
)

# Add the UI files
set(UIS
    deeplayout.ui
)

# Add the executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS} ${UIS})

# Link the Qt libraries
target_link_libraries(${PROJECT_NAME} Qt6::Widgets Qt6::Core Qt6::Gui)