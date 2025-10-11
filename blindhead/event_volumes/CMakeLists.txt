cmake_minimum_required(VERSION 3.5)
project(RealTimeEventVisualizer)

set(CMAKE_CXX_STANDARD 11)

# Find packages (adjust these if using different providers)
find_package(GLM REQUIRED)
find_package(OpenGL REQUIRED)

# Adjust these if you're pointing to your local GLFW, GLEW folders.
find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
pkg_search_module(GLEW REQUIRED glew)

include_directories(${GLM_INCLUDE_DIRS} ${GLFW_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS})
link_directories(${GLFW_LIBRARY_DIRS} ${GLEW_LIBRARY_DIRS})
add_definitions(${GLFW_CFLAGS_OTHER} ${GLEW_CFLAGS_OTHER})

add_executable(RealTimeEventVisualizer main.cpp)

target_link_libraries(RealTimeEventVisualizer ${OPENGL_LIBRARIES} ${GLFW_LIBRARIES} ${GLEW_LIBRARIES})

find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS})
target_link_libraries(RealTimeEventVisualizer ${HDF5_LIBRARIES})