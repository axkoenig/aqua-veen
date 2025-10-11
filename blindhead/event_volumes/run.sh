#!/bin/bash
set -e

# Define directories
BUILD_DIR="build"
IMGUI_DIR="imgui"

# Download and setup ImGui if it doesn't exist
if [ ! -d "$IMGUI_DIR" ]; then
    echo "Downloading ImGui..."
    git clone https://github.com/ocornut/imgui.git "$IMGUI_DIR"
fi

# Create the build directory if it doesn't exist
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory: $BUILD_DIR"
    mkdir "$BUILD_DIR"
fi

# Get Homebrew prefix for include and library paths
HOMEBREW_PREFIX=$(brew --prefix)

# Compile the program with ImGui
echo "Compiling main.cpp..."
g++ -std=c++17 \
    main.cpp \
    $IMGUI_DIR/imgui.cpp \
    $IMGUI_DIR/imgui_demo.cpp \
    $IMGUI_DIR/imgui_draw.cpp \
    $IMGUI_DIR/imgui_tables.cpp \
    $IMGUI_DIR/imgui_widgets.cpp \
    $IMGUI_DIR/backends/imgui_impl_glfw.cpp \
    $IMGUI_DIR/backends/imgui_impl_opengl3.cpp \
    -I$IMGUI_DIR \
    -I$IMGUI_DIR/backends \
    -I. \
    -I$HOMEBREW_PREFIX/include \
    -L$HOMEBREW_PREFIX/lib \
    -framework OpenGL \
    -framework Cocoa \
    -framework IOKit \
    -framework CoreVideo \
    -lglfw \
    -lGLEW \
    -lhdf5 \
    -lhdf5_cpp \
    -o "$BUILD_DIR/event_volumes"

echo "Compilation complete."

# Run the program
"$BUILD_DIR/event_volumes"