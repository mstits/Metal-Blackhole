cmake_minimum_required(VERSION 3.10)
project(MetalBlackhole LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Paths
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs/imgui)

# Dependencies
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
find_library(COCOA_FRAMEWORK Cocoa REQUIRED)
find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

# Executable
add_executable(MetalBlackhole
    src/main.mm
    include/ShaderCommon.h
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_metal.mm
)

target_include_directories(MetalBlackhole PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)

target_link_libraries(MetalBlackhole PRIVATE
    glfw
    glm::glm
    ${METAL_FRAMEWORK}
    ${QUARTZCORE_FRAMEWORK}
    ${COCOA_FRAMEWORK}
    ${METALKIT_FRAMEWORK}
    ${MPS_FRAMEWORK}
)

# Shader copying
add_custom_command(TARGET MetalBlackhole POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/geodesic.metal
    $<TARGET_FILE_DIR:MetalBlackhole>/geodesic.metal
)
