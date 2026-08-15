cmake_minimum_required(VERSION 3.16)   # OBJCXX became a first-class language in 3.16
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

# The Objective-C++ sources are written in ARC style; without -fobjc-arc every
# implied release is a no-op and the app leaks per frame and per resize.
# Per-source (not target-wide) so the plain C++ ImGui units are untouched.
set_source_files_properties(src/main.mm ${IMGUI_DIR}/backends/imgui_impl_metal.mm
    PROPERTIES COMPILE_OPTIONS "-fobjc-arc")

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

# Compiler hardening
target_compile_options(MetalBlackhole PRIVATE
    -Wall -Wextra -Wformat-security
    -fstack-protector-strong
)

# Shader pipeline: dependency-tracked so `make` after editing the shader
# actually refreshes the metallib AND the runtime-compile fallback copies.
# (POST_BUILD commands only ran on relink, silently serving stale shaders.)
set(SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/shaders/geodesic.metal)
set(SHADER_HDR ${CMAKE_CURRENT_SOURCE_DIR}/include/ShaderCommon.h)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/geodesic.metal ${CMAKE_BINARY_DIR}/ShaderCommon.h
    BYPRODUCTS ${CMAKE_BINARY_DIR}/geodesic.metallib
    COMMAND ${CMAKE_COMMAND} -E copy ${SHADER_SRC} ${CMAKE_BINARY_DIR}/geodesic.metal
    COMMAND ${CMAKE_COMMAND} -E copy ${SHADER_HDR} ${CMAKE_BINARY_DIR}/ShaderCommon.h
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_metallib.sh
        ${SHADER_HDR} ${SHADER_SRC} ${CMAKE_BINARY_DIR}
    DEPENDS ${SHADER_SRC} ${SHADER_HDR} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_metallib.sh
    COMMENT "Compiling Metal shaders (metallib + runtime fallback copies)"
)
add_custom_target(shaders ALL
    DEPENDS ${CMAKE_BINARY_DIR}/geodesic.metal ${CMAKE_BINARY_DIR}/ShaderCommon.h)
add_dependencies(MetalBlackhole shaders)
