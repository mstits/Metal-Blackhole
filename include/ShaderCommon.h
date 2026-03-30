#ifndef SHADER_COMMON_H
#define SHADER_COMMON_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
typedef float4 packed_float4;
typedef float3 packed_float3;
#else
#include <glm/glm.hpp>
typedef glm::vec4 packed_float4;
typedef glm::vec3 packed_float3;
#endif

struct SimObject {
    packed_float4 posRadius;
    packed_float4 color;
    float  mass;
    float  _pad0[3];
    packed_float4 velocity;
};

struct CameraData {
    packed_float4 camPos;
    packed_float4 camRight;
    packed_float4 camUp;
    packed_float4 camForward;
    float  tanHalfFov;
    float  aspect;
    int    moving;
    int    _pad4;
};

struct SystemUniforms {
    float time;
    float spin;
    float star_scint;
    float nebula_int;
    float charge;
    float dt_sim;
    float bloom_threshold;
    float flare_int;
    float motion_blur;
    float film_grain;
    float disk_density;
    float disk_height;
    float shadow_int;
    float gw_amp;
    float _pad[2]; // Ensure 16-byte alignment
};

struct ObjectsUniform {
    int count;
};

struct GridUniforms {
#ifdef __METAL_VERSION__
    float4x4 viewProj;
#else
    glm::mat4 viewProj;
#endif
};

#endif
