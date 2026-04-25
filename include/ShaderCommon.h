#ifndef SHADER_COMMON_H
#define SHADER_COMMON_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#define VEC4 float4
#define VEC3 float3
#define MAT4 float4x4
#else
#include <glm/glm.hpp>
#define VEC4 glm::vec4
#define VEC3 glm::vec3
#define MAT4 glm::mat4
#endif

struct SimObject {
    VEC4   posRadius;
    VEC4   color;
    float  mass;
    float  _pad0[3];
    VEC4   velocity;
};

struct CameraData {
    VEC4   camPos;
    VEC4   camRight;
    VEC4   camUp;
    VEC4   camForward;
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
    float exposure;
    float jet_int;
    float r_horizon;        // Kerr event horizon (rs units), CPU-precomputed from spin
    float r_isco;           // Kerr ISCO prograde (rs units), CPU-precomputed from spin
    int   enable_bloom;     // 0 = bloom pipeline contributes nothing; 1 = active
    int   _pad_sys;
};

struct ObjectsUniform {
    int count;
    int bh_index;           // Index into objects[] of the black hole; -1 if absent
    int _pad_obj[2];
};

struct GridUniforms {
    MAT4 viewProj;
};

#undef VEC4
#undef VEC3
#undef MAT4

#endif
