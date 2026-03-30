#include <metal_stdlib>
using namespace metal;

// Structs (SimObject, CameraData, SystemUniforms, ObjectsUniform, GridUniforms)
// are provided by ShaderCommon.h, prepended at compile time.

// --- HIGH-FIDELITY NOISE ENGINE ---

static inline float hash13(float3 p3) {
    p3  = fract(p3 * 0.1031f);
    p3 += dot(p3, p3.yzx + 33.33f);
    return fract((p3.x + p3.y) * p3.z);
}

static inline half noise_half(float3 x) {
    float3 i = floor(x);
    float3 f = fract(x);
    f = f*f*(3.0f-2.0f*f);
    float h000 = hash13(i+float3(0,0,0));
    float h100 = hash13(i+float3(1,0,0));
    float h010 = hash13(i+float3(0,1,0));
    float h110 = hash13(i+float3(1,1,0));
    float h001 = hash13(i+float3(0,0,1));
    float h101 = hash13(i+float3(1,0,1));
    float h011 = hash13(i+float3(0,1,1));
    float h111 = hash13(i+float3(1,1,1));
    return (half)mix(mix(mix(h000, h100, f.x), mix(h010, h110, f.x), f.y),
                    mix(mix(h001, h101, f.x), mix(h011, h111, f.x), f.y), f.z);
}

static inline half fbm_half(float3 p, float t) {
    half v = 0.0h;
    half a = 0.5h;
    [[unroll]]
    for (int i=0; i<5; i++) {
        v += a * noise_half(p + t * 0.3f);
        p = p * 2.02f + float3(10.0f);
        a *= 0.5h;
    }
    return v;
}

// 3-octave variant for raytrace disk sampling (cheaper, visually close)
static inline half fbm_half3(float3 p, float t) {
    half v = 0.0h;
    half a = 0.5h;
    [[unroll]]
    for (int i=0; i<3; i++) {
        v += a * noise_half(p + t * 0.3f);
        p = p * 2.02f + float3(10.0f);
        a *= 0.5h;
    }
    return v;
}

// --- FLUID DYNAMICS ---

kernel void simulate_disk_fluid(texture2d<float, access::sample> inTex [[texture(0)]],
                                texture2d<float, access::write> outTex [[texture(1)]],
                                constant SystemUniforms& sys [[buffer(0)]],
                                uint2 pix [[thread_position_in_grid]]) {
    uint w = outTex.get_width(); uint h = outTex.get_height();
    if (pix.x >= w || pix.y >= h) return;

    float2 uv = (float2(pix) + 0.5f) / float2(w, h);
    float2 centered = uv - 0.5f;
    float r = length(centered);
    float phi = atan2(centered.y, centered.x);

    float v_ang = 0.6f / max(0.01f, pow(r, 1.5f));
    float delta_phi = v_ang * sys.dt_sim * 0.2f;
    float2 prev_uv = float2(r * cos(phi - delta_phi), r * sin(phi - delta_phi)) + 0.5f;
    
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float4 val = inTex.sample(s, prev_uv);
    
    float noise_val = (float)fbm_half(float3(uv * 15.0f, sys.time * 0.1f), sys.time * 0.05f);
    if (r > 0.08f && r < 0.48f) {
        val.x = mix(val.x, noise_val, 0.04f);
        val.y = mix(val.y, (float)noise_half(float3(uv * 30.0f, -sys.time * 0.2f)), 0.02f);
    } else {
        val.xy *= 0.94f;
    }
    outTex.write(val, pix);
}

// --- RAYTRACING ---

static inline half3 sampleBackground_half(float3 rd, float time, float scint, float nebula_int) {
    float n1 = (float)noise_half(rd * 2.5f + time * 0.04f);
    float n2 = (float)noise_half(rd * 4.5f - time * 0.02f);
    half3 nebula = mix(half3(0.002h, 0.005h, 0.015h), half3(0.02h, 0.01h, 0.04h), (half)n1 * 0.5h + 0.5h);
    nebula += half3(0.04h, 0.015h, 0.06h) * pow(max(0.0f, n2), 4.0f) * (half)nebula_int;
    
    float3 p = rd * 600.0f;
    float3 id = floor(p);
    float h = hash13(id);
    half3 col = nebula;
    if (h > 0.9988f) {
        float pulse = 1.0f + scint * 0.5f * sin(time * 4.0f + h * 100.0f);
        col += half3(1.5h, 1.8h, 2.5h) * (half)(pulse * pow(hash13(id + 0.5f), 20.0f) * 15.0f);
    }
    return col;
}

static inline float3 get_acc(float3 p, float3 v, float a, float Q) {
    float r2 = dot(p, p);
    float r = sqrt(r2);
    float3 h_vec = cross(p, v);
    float3 acc = -1.5f * dot(h_vec, h_vec) * p / (r2 * r2 * r);
    acc += (Q * Q) * p / (r2 * r2);
    float3 J = float3(0.0f, a * 0.5f, 0.0f);
    float3 B = (3.0f * p * dot(p, J) / r2 - J) / (r2 * r);
    return acc + 4.0f * cross(v, B);
}

static inline void stepRK4(thread float3& pos, thread float3& vel, float dt, float a, float Q) {
    float3 k1p = vel;
    float3 k1v = get_acc(pos, vel, a, Q);
    float3 k2p = vel + k1v * (dt * 0.5f);
    float3 k2v = get_acc(pos + k1p * (dt * 0.5f), k2p, a, Q);
    float3 k3p = vel + k2v * (dt * 0.5f);
    float3 k3v = get_acc(pos + k2p * (dt * 0.5f), k3p, a, Q);
    float3 k4p = vel + k3v * dt;
    float3 k4v = get_acc(pos + k3p * dt, k4p, a, Q);
    pos += (k1p + 2.0f*k2p + 2.0f*k3p + k4p) * (dt / 6.0f);
    vel = normalize(vel + (k1v + 2.0f*k2v + 2.0f*k3v + k4v) * (dt / 6.0f));
}

// Euler integrator for weak-field regions (r > 8) — 1 force eval vs 4
static inline void stepEuler(thread float3& pos, thread float3& vel, float dt, float a, float Q) {
    float3 acc = get_acc(pos, vel, a, Q);
    pos += vel * dt;
    vel = normalize(vel + acc * dt);
}

kernel void raytrace(texture2d<float, access::write> out [[texture(0)]],
                     texture2d<float, access::sample> fluidTex [[texture(1)]],
                     constant CameraData& cam [[buffer(0)]],
                     const device SimObject* objs [[buffer(1)]],
                     constant ObjectsUniform& u_obj [[buffer(2)]],
                     constant SystemUniforms& sys [[buffer(3)]],
                     uint2 pix [[thread_position_in_grid]]) {
    uint w = out.get_width(); uint h = out.get_height();
    if (pix.x >= w || pix.y >= h) return;
    
    float ur = (2.0f * (float(pix.x) + 0.5f) / float(w) - 1.0f) * cam.aspect * cam.tanHalfFov;
    float vr = (1.0f - 2.0f * (float(pix.y) + 0.5f) / float(h)) * cam.tanHalfFov;
    float3 rd = normalize(ur * cam.camRight.xyz + vr * cam.camUp.xyz + cam.camForward.xyz);
    float3 ro = cam.camPos.xyz;

    const device SimObject* bh = nullptr;
    for (int i=0; i<u_obj.count; i++) if (objs[i].mass > 1e35f) { bh = &objs[i]; break; }
    if (!bh) return;

    float rs = bh->posRadius.w;
    float3 bhPos = bh->posRadius.xyz;
    float3 pos = (ro - bhPos) / rs;
    float3 vel = rd;
    
    float3 col_accum = float3(0.0f);
    float trans = 1.0f;
    
    const float r_in = 2.6f;
    const float r_out = 22.0f;
    const float disk_h = sys.disk_height;
    const float disk_h3 = disk_h * 3.0f;

    for (int i=0; i<1200; i++) {
        float r = length(pos);
        if (r < 1.005f) { trans = 0.0f; break; }
        if (r > 500.0f) break;
        // Early exit: ray heading outward past all disk/object influence
        if (r > 30.0f && dot(pos, vel) > 0.0f) break;
        
        // INVARIANT: disk step clamp must stay at 0.06 — do NOT reduce.
        // Lower values exhaust iteration budget for edge-on rays.
        // INVARIANT: raytrace must run at FULL resolution — no half-res.
        // Half-res causes sub-pixel thin disk to bleed stars through.
        float dt = max(r * 0.06f, 0.001f);
        if (abs(pos.y) < disk_h3) dt = min(dt, 0.06f);
        
        // Weak-field: Euler (1 force eval). Strong-field: RK4 (4 force evals).
        if (r > 8.0f) stepEuler(pos, vel, dt, sys.spin, sys.charge);
        else          stepRK4(pos, vel, dt, sys.spin, sys.charge);
        
        if (simd_any(r < 150.0f)) {
            float3 p_world = pos * rs + bhPos;
            for (int j=0; j<u_obj.count; j++) {
                if (objs[j].mass <= 1e35f) {
                    float3 delta = p_world - objs[j].posRadius.xyz;
                    if (dot(delta, delta) < objs[j].posRadius.w * objs[j].posRadius.w) {
                        col_accum += trans * objs[j].color.xyz * 25.0f;
                        trans = 0.0f; break;
                    }
                }
            }
        }
        if (trans <= 0.0f) break;

        if (abs(pos.y) < disk_h) {
            float rh = length(pos.xz);
            if (rh > r_in && rh < r_out) {
                float d_norm = (rh - r_in) / (r_out - r_in);
                float2 fluid_uv = pos.xz / 50.0f + 0.5f;
                constexpr sampler s(filter::linear);
                float2 f_val = fluidTex.sample(s, fluid_uv).xy;
                float noise_val = (float)fbm_half3(float3(rh * 1.8f, atan2(pos.z, pos.x) * 5.0f, sys.time * 0.1f), sys.time * 0.02f);
                noise_val = mix(noise_val, f_val.x, 0.6f + 0.4f * f_val.y);

                float y_frac = 1.0f - abs(pos.y) / disk_h;
                float density = smoothstep(0.0f, 0.05f, d_norm) * (1.0f - smoothstep(0.45f, 1.0f, d_norm)) * (y_frac * y_frac * y_frac) * (0.1f + 0.9f * noise_val) * sys.disk_density;
                
                half h_dnorm = half(d_norm);
                half3 dCol = mix(half3(0.4h, 0.8h, 1.0h), half3(1.0h, 0.7h, 0.2h), smoothstep(half(0.0h), half(0.2h), h_dnorm));
                dCol = mix(dCol, half3(0.8h, 0.1h, 0.0h), smoothstep(half(0.2h), half(0.6h), h_dnorm));
                dCol = mix(dCol, half3(0.1h, 0.0h, 0.3h), smoothstep(half(0.6h), half(1.0h), h_dnorm));
                
                float inv_rh = 1.0f / rh;
                float2 tang = float2(-pos.z, pos.x) * rsqrt(pos.z*pos.z + pos.x*pos.x);
                float cos_v = vel.x * tang.x + vel.z * tang.y;
                float beaming_base = max(0.01f, 1.0f - 0.98f * cos_v * sqrt(max(0.0f, inv_rh)));
                float beaming = 1.0f / (beaming_base * beaming_base * beaming_base * beaming_base);
                float redshift = sqrt(max(0.01f, 1.0f - inv_rh));
                
                float shadow = 1.0f;
                if (sys.shadow_int > 0.0f) {
                    float3 shadow_p = pos;
                    float shadow_accum = 0.0f;
                    [[unroll]]
                    for (int s=0; s<2; s++) {
                        shadow_p += float3(0, 0.2f, 0); 
                        float s_rh = length(shadow_p.xz);
                        if (s_rh > r_in && s_rh < r_out && abs(shadow_p.y) < disk_h) shadow_accum += 0.4f;
                    }
                    shadow = exp(-shadow_accum * sys.shadow_int * 5.0f);
                }

                float step_opacity = density * 0.7f;
                col_accum += trans * float3(dCol) * 35.0f * beaming * redshift * step_opacity * shadow;
                trans *= (1.0f - step_opacity);
            }
        }
        
        // Glow only matters close to the BH — at r=10 it's 0.000006, negligible
        if (r < 5.0f) {
            float glow = 0.0006f / (r*r);
            col_accum += trans * float3(0.12f, 0.07f, 0.04f) * glow * 60.0f;
            trans *= (1.0f - glow);
        }

        if (trans < 0.005f) break;
    }
    
    // Gravitational lensing of distant stars: check if the deflected exit
    // direction `vel` points toward any star. The geodesic bending IS the lens.
    if (trans > 0.005f) {
        float3 bg = (float3)sampleBackground_half(vel, sys.time, sys.star_scint, sys.nebula_int);
        for (int j = 0; j < u_obj.count; j++) {
            if (objs[j].mass <= 1e35f) {
                float3 star_dir = normalize(objs[j].posRadius.xyz - bhPos);
                float star_dist = length(objs[j].posRadius.xyz - bhPos);
                float ang_radius = objs[j].posRadius.w / star_dist;
                float cos_angle = dot(vel, star_dir);
                float threshold = 1.0f - ang_radius * ang_radius * 0.5f;
                if (cos_angle > threshold) {
                    // Inverse-square dimming: star brightness scales with (R/d)²
                    float solid_angle = ang_radius * ang_radius;
                    float brightness = min(solid_angle * 800.0f, 4.0f);  // cap to prevent flare overdrive
                    float limb = smoothstep(threshold, 1.0f, cos_angle);
                    bg = mix(bg, objs[j].color.xyz * brightness, limb);
                }
            }
        }
        col_accum += trans * bg;
    }
    out.write(float4(col_accum, 1.0f), pix);
}

// --- POST-PROCESSING ---

kernel void post_process_suite(texture2d<float, access::read> inTex [[texture(0)]],
                               texture2d<float, access::sample> accumTex [[texture(1)]],
                               texture2d<float, access::write> outTex [[texture(2)]],
                               constant SystemUniforms& sys [[buffer(0)]],
                               uint2 pix [[thread_position_in_grid]]) {
    uint w = outTex.get_width(); uint h = outTex.get_height();
    if (pix.x >= w || pix.y >= h) return;

    float3 col = inTex.read(pix).rgb;

    // Anamorphic Flare
    float flare = 0.0f;
    for(int i=-10; i<=10; i++) {
        uint2 p = uint2(clamp(int(pix.x) + i*4, 0, int(w)-1), pix.y);
        flare += max(0.0f, dot(inTex.read(p).rgb, float3(0.2126f, 0.7152f, 0.0722f)) - sys.bloom_threshold);
    }
    col += float3(0.1f, 0.3f, 1.0f) * (flare / 21.0f) * sys.flare_int;

    float2 uv = (float2(pix) + 0.5f) / float2(w, h);
    col = mix(col, accumTex.sample(sampler(filter::linear), uv).rgb, sys.motion_blur);
    col += (hash13(float3(uv * 100.0f, sys.time)) - 0.5f) * sys.film_grain;

    // ACES
    float a = 2.51f; float b = 0.03f; float c = 2.43f; float d = 0.59f; float e = 0.14f;
    col = clamp((col*(a*col+b))/(col*(c*col+d)+e), 0.0f, 1.0f);
    outTex.write(float4(pow(col, 0.4545f), 1.0f), pix);
}

kernel void update_physics(device SimObject* objects [[buffer(0)]], constant ObjectsUniform& u [[buffer(1)]], constant SystemUniforms& sys [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id >= (uint)u.count || objects[id].mass > 1e35f) return;
    float dt = 2500000.0f * sys.dt_sim;
    float G = 6.67430e-11f;
    float eps = 1e9f;
    float3 p = objects[id].posRadius.xyz;
    float3 v = objects[id].velocity.xyz;
    float3 f = float3(0.0f);
    for (int i=0; i<u.count; i++) {
        if (i == (int)id) continue;
        float3 r_v = objects[i].posRadius.xyz - p;
        float r2 = dot(r_v, r_v);
        f += normalize(r_v) * (G * objects[i].mass / (r2 + eps * eps));
    }
    v += f * dt;
    p += v * dt;
    objects[id].velocity.xyz = v;
    objects[id].posRadius.xyz = p;
}

struct VertexOut { float4 position [[position]]; float depth; };
struct VertexIn { float3 position [[attribute(0)]]; };

vertex VertexOut grid_vertex(VertexIn in [[stage_in]], constant GridUniforms& u [[buffer(1)]], const device SimObject* o [[buffer(2)]], constant ObjectsUniform& uo [[buffer(3)]], constant SystemUniforms& sys [[buffer(4)]]) {
    VertexOut out; float3 p = in.position;
    const device SimObject* bh = nullptr;
    for (int i=0; i<uo.count; i++) if (o[i].mass > 1e35f) { bh = &o[i]; break; }
    if (bh) {
        float3 delta = p - bh->posRadius.xyz;
        float d = length(delta.xz);
        float ripple = 0.0f;
        if (sys.gw_amp > 0.0f) {
            float phase = d * 1e-11f - sys.time * 2.0f;
            ripple = sin(phase) * (sys.gw_amp * 2e11f) / (1.0f + d * 1e-12f);
            ripple *= cos(2.0f * atan2(delta.z, delta.x));
        }
        p.y += ripple - (bh->posRadius.w * 10.0f) / (1.0f + d / (bh->posRadius.w * 3.0f));
    }
    float grid_drop = bh ? bh->posRadius.w * 24.0f : 3.0e11f;
    p.y -= grid_drop;
    out.position = u.viewProj * float4(p, 1.0f);
    return out;
}
fragment float4 grid_fragment(VertexOut in [[stage_in]]) { return float4(0.0f, 0.5f, 1.0f, 0.4f); }
