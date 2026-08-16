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
    VEC4   acceleration;    // velocity-Verlet leapfrog scratch; xyz used, w unused
};

struct CameraData {
    VEC4   camPos;
    VEC4   camRight;
    VEC4   camUp;
    VEC4   camForward;
    float  tanHalfFov;
    float  aspect;
    float  _pad[2];
};

// Lens modes (SystemUniforms::lens_mode). Non-Standard lenses are mutually
// exclusive false-color remappings; overlays and physics switches compose.
#define LENS_STANDARD    0
#define LENS_RING_ORDER  1   // categorical color by image order n (equatorial crossings)
#define LENS_REDSHIFT    2   // diverging colormap of the g-factor at the first disk hit
#define LENS_CHECKER     3   // procedural lat-long checkerboard sky (lensing map)
#define LENS_EHT         4   // telescope-beam Gaussian blur + radio colormap

// Beaming modes (SystemUniforms::beaming_mode).
#define BEAM_FULL        0   // g^4 intensity + temperature shift (physical)
#define BEAM_NO_BOOST    1   // temperature shift only (color, no intensity change)
#define BEAM_NONE        2   // no shift, no boost (Luminet-style bolometric render)

// Accretion-flow models (SystemUniforms::disk_model).
#define DISK_THIN        0   // optically-thick Novikov-Thorne surface at theta = pi/2
#define DISK_VOLUMETRIC  1   // optically-thin plasma torus, covariant radiative transfer

struct SystemUniforms {
    float time;
    float spin;             // a/M in [-1, 1]; negative = retrograde disk
    float star_scint;
    float nebula_int;
    float charge;           // Q/M in [0, 1], jointly clamped so a^2 + Q^2 <= 1
    float dt_phys;          // N-body physics substep in seconds (fixed, host-substepped)
    float bloom_threshold;
    float flare_int;
    float motion_blur;
    float film_grain;
    float disk_density;
    float disk_r_out;       // disk outer radius in rs units
    float disk_temp;        // Novikov-Thorne peak temperature in Kelvin
    float vignette_int;     // 0 = off
    float gw_amp;
    float exposure;
    float jet_int;
    float r_horizon;        // Kerr-Newman outer horizon (rs units), from spin AND charge
    float r_isco;           // ISCO (rs units): prograde for spin >= 0, retrograde for spin < 0
    float photon_ring_boost;// 0 = physical; >0 amplifies image orders n >= 1
    float accum_alpha;      // temporal blend weight: 1 = replace, <1 = accumulate
    float jitter_x;         // subpixel jitter in [-0.5, 0.5] (Halton, per frame)
    float jitter_y;
    float edr_headroom;     // display EDR headroom: 1 = SDR, >1 = highlights above
                            // reference white (Liquid Retina XDR reports ~4-16)
    // Volumetric torus (EHT code-comparison parameterization, M units).
    float torus_r0;         // radial density scale
    float torus_h;          // vertical compression (0 = spherical, larger = thinner)
    float torus_l0;         // specific-angular-momentum normalization
    float torus_alpha;      // emissivity spectral index alpha_s: j_nu ~ nu^-alpha_s
    float torus_absorb;     // absorption scale A (0 = optically thin)
    float torus_temp;       // visible-light color temperature at the density peak
    float torus_density;    // overall emission scale
    float _padf1;
    int   enable_bloom;     // 0 = bloom pipeline contributes nothing; 1 = active
    int   lens_mode;        // LENS_* above
    int   beaming_mode;     // BEAM_* above
    int   star_bodies;      // 1 = render + lens the N-body companion stars
    int   disk_model;       // DISK_* above
    int   _padi[3];
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
