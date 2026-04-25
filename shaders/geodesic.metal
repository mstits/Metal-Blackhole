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

// =====================================================================
// EXACT KERR-NEWMAN GEODESIC INTEGRATOR
//   Boyer-Lindquist coordinates with Carter's separation constant.
//   Reference: Carter, Phys. Rev. 174, 1559 (1968); Cunningham & Bardeen,
//   ApJ 183, 237 (1973). Same formulation used by ipole, GYOTO, RAPTOR.
//
// Conventions (matching the host's slider semantics):
//   sys.spin    = a/M ∈ [-1, 1]   (dimensionless Kerr spin)
//   sys.charge  = Q/M ∈ [0, 1]    (dimensionless charge)
//   r, λ        in rs = 2M units
//   M = 1/2 in these units, so 2Mr → r and Δ = r² − r + a² + Q²
//   "a" inside the metric below is the dimensional spin in rs units = (a/M)/2.
//
// Null-geodesic conserved quantities:
//   E   = −p_t = 1                (affine normalization)
//   L_z =  p_φ                    (axial symmetry)
//   Q_C = p_θ² + cos²θ (L_z²/sin²θ − a²)   (Carter, μ = 0)
//
// Hamilton's equations reduce, via Carter's separation, to:
//   Σ (dr/dλ)  = ±√R(r)
//   Σ (dθ/dλ) = ±√Θ(θ)
//   Σ (dφ/dλ) = (a P − a² L_z + L_z Δ/sin²θ) / Δ
// with R(r) = P² − Δ[(L_z − a)² + Q_C],  Θ(θ) = Q_C + a²cos²θ − L_z²cot²θ,
//      P = r² + a² − a L_z,  Σ = r² + a²cos²θ,  Δ = r² − r + a² + Q_ch².
//
// We track the radial and polar momentum signs (s_r, s_θ) and flip them
// when R or Θ crosses zero (turning points).
// =====================================================================

struct KerrConst {
    float a;        // real spin in rs units (= sys.spin / 2)
    float a2;
    float Q_ch_sq;  // real charge² in rs units (= (sys.charge / 2)²)
    float L;        // L_z, conserved
    float QC;       // Carter constant, conserved
};

static inline float kerr_Sigma(float r, float cos_th, float a2) {
    return r * r + a2 * cos_th * cos_th;
}

static inline float kerr_Delta(float r, float a2, float Q_ch_sq) {
    return r * r - r + a2 + Q_ch_sq;
}

// Radial potential R(r) for E = 1.
static inline float kerr_R(float r, KerrConst k) {
    float P = r * r + k.a2 - k.a * k.L;
    float B = (k.L - k.a) * (k.L - k.a) + k.QC;
    float Delta = kerr_Delta(r, k.a2, k.Q_ch_sq);
    return P * P - Delta * B;
}

// Polar potential Θ(θ) for E = 1.
static inline float kerr_Theta(float cos_th, float sin_th, KerrConst k) {
    float s2 = sin_th * sin_th;
    if (s2 < 1e-9f) s2 = 1e-9f;     // dodge polar coord singularity
    return k.QC + k.a2 * cos_th * cos_th - k.L * k.L * cos_th * cos_th / s2;
}

// dφ/dλ from the Carter form (E = 1).
//   Σ dφ/dλ = a (r − Q_ch²)/Δ − a²L/Δ + L/sin²θ
static inline float kerr_dphi(float r, float sin_th, KerrConst k) {
    float Delta = kerr_Delta(r, k.a2, k.Q_ch_sq);
    float s2 = sin_th * sin_th;
    if (s2 < 1e-9f) s2 = 1e-9f;
    return (k.a * (r - k.Q_ch_sq) - k.a2 * k.L) / Delta + k.L / s2;
}

// One RK4 step of the (r, θ, φ) state. The sign trackers s_r, s_θ are
// pre-multiplied into √R / √Θ; turning-point detection runs in the caller.
static inline void kerr_rk4(thread float& r, thread float& th, thread float& ph,
                            float s_r, float s_th, float dlambda, KerrConst k)
{
    // Each k_i evaluates (dr, dθ, dφ)/dλ at an intermediate state.
    auto rhs = [&](float r_, float th_) {
        float c = cos(th_);
        float s = sin(th_);
        float Sigma = kerr_Sigma(r_, c, k.a2);
        float R = kerr_R(r_, k);
        float Th = kerr_Theta(c, s, k);
        float dr = s_r  * sqrt(max(R, 0.0f)) / Sigma;
        float dt = s_th * sqrt(max(Th, 0.0f)) / Sigma;
        float dp = kerr_dphi(r_, s, k) / Sigma;
        return float3(dr, dt, dp);
    };

    float3 k1 = rhs(r, th);
    float3 k2 = rhs(r + 0.5f * dlambda * k1.x, th + 0.5f * dlambda * k1.y);
    float3 k3 = rhs(r + 0.5f * dlambda * k2.x, th + 0.5f * dlambda * k2.y);
    float3 k4 = rhs(r + dlambda * k3.x,         th + dlambda * k3.y);

    r  += dlambda * (k1.x + 2.0f * k2.x + 2.0f * k3.x + k4.x) / 6.0f;
    th += dlambda * (k1.y + 2.0f * k2.y + 2.0f * k3.y + k4.y) / 6.0f;
    ph += dlambda * (k1.z + 2.0f * k2.z + 2.0f * k3.z + k4.z) / 6.0f;
}

// Convert BL (r, θ, φ) to BH-local Cartesian (rs units), with y as the spin
// axis. Uses the spherical approximation; the spheroidal correction is
// O(a²/r²) and irrelevant for star-intersection bounding-sphere tests.
static inline float3 bl_to_cart(float r, float sin_th, float cos_th, float sin_ph, float cos_ph) {
    return float3(r * sin_th * cos_ph, r * cos_th, r * sin_th * sin_ph);
}

// Local spherical basis at (r, θ, φ) expressed in Cartesian (rs units).
static inline void bl_basis(float sin_th, float cos_th, float sin_ph, float cos_ph,
                            thread float3& r_hat, thread float3& th_hat, thread float3& ph_hat) {
    r_hat  = float3(sin_th * cos_ph,  cos_th,           sin_th * sin_ph);
    th_hat = float3(cos_th * cos_ph, -sin_th,           cos_th * sin_ph);
    ph_hat = float3(-sin_ph,           0.0f,            cos_ph);
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

    if (u_obj.bh_index < 0 || u_obj.bh_index >= u_obj.count) return;
    const device SimObject* bh = &objs[u_obj.bh_index];
    float rs = bh->posRadius.w;
    float3 bhPos = bh->posRadius.xyz;

    // Camera position relative to BH, in rs units, in BH-local Cartesian.
    float3 ro_bh = (ro - bhPos) / rs;

    // ----- Kerr-Newman parameters in rs units (M = 1/2). -----
    // Slider sys.spin is dimensionless a/M; the dimensional spin in rs units
    // is half that. Same for charge.
    KerrConst kerr;
    kerr.a       = 0.5f * sys.spin;
    kerr.a2      = kerr.a * kerr.a;
    float Q_ch   = 0.5f * sys.charge;
    kerr.Q_ch_sq = Q_ch * Q_ch;
    // L and QC are filled in below from the camera ray.

    float r_horizon = sys.r_horizon;
    float r_isco    = sys.r_isco;
    const float r_in   = max(r_isco, r_horizon * 1.2f);
    const float r_out  = 22.0f;
    const float disk_h = sys.disk_height;     // half-thickness in rs (Cartesian y)

    // ----- Convert camera Cartesian → Boyer-Lindquist coordinates. -----
    float r_obs       = length(ro_bh);
    float cos_th_obs  = clamp(ro_bh.y / r_obs, -1.0f, 1.0f);
    float sin_th_obs  = sqrt(max(1.0f - cos_th_obs * cos_th_obs, 1e-9f));
    float phi_obs     = atan2(ro_bh.z, ro_bh.x);
    float sin_ph_obs  = sin(phi_obs);
    float cos_ph_obs  = cos(phi_obs);

    // Local spherical basis at the observer (Cartesian world).
    float3 r_hat, th_hat, ph_hat;
    bl_basis(sin_th_obs, cos_th_obs, sin_ph_obs, cos_ph_obs, r_hat, th_hat, ph_hat);

    // Decompose the pixel ray direction into the observer's local frame.
    float n_r  = dot(rd, r_hat);
    float n_th = dot(rd, th_hat);
    float n_ph = dot(rd, ph_hat);

    // ----- Static-observer tetrad initial conditions. -----
    // For an observer at rest in BL coords (outside any ergosphere), the
    // photon's covariant momenta are p_i = √g_ii · n_i / √(−g_tt) per axis,
    // with E ≡ −p_t = 1 set by affine choice. We ignore the g_tφ tetrad
    // correction; at r_obs ~ 20 rs it contributes < 1.5% to L_z.
    float Sigma_obs = kerr_Sigma(r_obs, cos_th_obs, kerr.a2);
    float Delta_obs = kerr_Delta(r_obs, kerr.a2, kerr.Q_ch_sq);
    float gtt_neg   = 1.0f - r_obs / Sigma_obs;             // = -g_tt (positive outside ergosphere)
    float grr       = Sigma_obs / max(Delta_obs, 1e-9f);
    float gthth     = Sigma_obs;
    // g_φφ = ((r²+a²) Σ + a² r sin²θ) sin²θ / Σ   [in rs units, 2M = 1]
    float gphph_over_s2 = ((r_obs * r_obs + kerr.a2) * Sigma_obs
                          + kerr.a2 * r_obs * sin_th_obs * sin_th_obs) / Sigma_obs;
    float gphph     = gphph_over_s2 * sin_th_obs * sin_th_obs;

    float omega_obs = rsqrt(max(gtt_neg, 1e-9f));            // = 1/√(-g_tt)
    float p_r_init  = sqrt(grr)   * omega_obs * n_r;
    float p_th_init = sqrt(gthth) * omega_obs * n_th;
    kerr.L          = sqrt(max(gphph, 0.0f)) * omega_obs * n_ph;

    // Carter constant from p_θ. For E = 1:
    //   p_θ² = Q_C + cos²θ (a² − L²/sin²θ)   →   Q_C = p_θ² − cos²θ(a² − L²/sin²θ)
    {
        float c2 = cos_th_obs * cos_th_obs;
        float s2 = sin_th_obs * sin_th_obs;
        kerr.QC = p_th_init * p_th_init - c2 * (kerr.a2 - kerr.L * kerr.L / max(s2, 1e-9f));
    }

    // ----- Integrator state (BL coordinates + sign trackers). -----
    float r        = r_obs;
    float th       = acos(cos_th_obs);
    float ph       = phi_obs;
    float s_r      = (p_r_init  < 0.0f) ? -1.0f : 1.0f;
    float s_th     = (p_th_init < 0.0f) ? -1.0f : 1.0f;
    float prev_R   = kerr_R(r, kerr);
    float prev_Th  = kerr_Theta(cos_th_obs, sin_th_obs, kerr);
    float prev_cos = cos_th_obs;
    int   disk_crossings = 0;
    float init_abs_n_th  = abs(n_th);

    float3 col_accum = float3(0.0f);
    float  trans     = 1.0f;

    // ----- Geodesic integration loop. -----
    for (int i = 0; i < 1200; i++) {
        // Horizon: terminate slightly above r_+ to avoid Δ → 0 singularity.
        if (r < r_horizon * 1.01f) { trans = 0.0f; col_accum = float3(0.0f); break; }
        if (r > 500.0f) break;

        // Outward early-out: ray escaping past all disk/object influence.
        float dr_sign = s_r * sqrt(max(prev_R, 0.0f));
        if (r > 30.0f && dr_sign > 0.0f) break;

        // Affine-parameter step. The natural length scale is the local r;
        // tighten near the disk slab for accurate emission accumulation.
        float dlam = max(r * 0.06f, 0.001f);
        float y_cart = r * prev_cos;
        if (abs(y_cart) < 3.0f * disk_h && r < r_out * 1.2f) {
            // Near-disk: cap step. abs(p_θ)/Σ controls how fast we cross the
            // slab in θ; |dy/dλ| ≈ |Σ dθ/dλ| / r ≈ |√Θ|/r approximately.
            float cross_speed = min(abs(s_th * sqrt(max(prev_Th, 0.0f))) / max(r, 1e-3f) * 10.0f, 1.0f);
            float dlam_cap    = mix(0.4f, 0.06f, cross_speed);
            dlam              = min(dlam, dlam_cap);
        }

        // RK4 step on (r, θ, φ).
        kerr_rk4(r, th, ph, s_r, s_th, dlam, kerr);

        // Sign updates at turning points: detect by R(r) or Θ(θ) crossing zero.
        float cos_th = cos(th);
        float sin_th = sqrt(max(1.0f - cos_th * cos_th, 1e-9f));
        float Rn  = kerr_R(r, kerr);
        float Thn = kerr_Theta(cos_th, sin_th, kerr);
        if (Rn  < 0.0f && prev_R  > 0.0f) s_r  = -s_r;
        if (Thn < 0.0f && prev_Th > 0.0f) s_th = -s_th;
        prev_R  = Rn;
        prev_Th = Thn;

        // Photon-ring detection: equatorial-plane crossings = sign flips of cos θ.
        if (prev_cos * cos_th < 0.0f) disk_crossings++;
        prev_cos = cos_th;

        // ----- Background star-sphere intersection (Cartesian, BH-local). -----
        // Only consult when at least one lane is close enough for any star.
        if (simd_any(r < 150.0f)) {
            float sin_ph = sin(ph);
            float cos_ph = cos(ph);
            float3 pos_cart = bl_to_cart(r, sin_th, cos_th, sin_ph, cos_ph);
            float3 p_world  = pos_cart * rs + bhPos;
            for (int j = 0; j < u_obj.count; j++) {
                if (j == u_obj.bh_index) continue;
                float3 delta = p_world - objs[j].posRadius.xyz;
                if (dot(delta, delta) < objs[j].posRadius.w * objs[j].posRadius.w) {
                    col_accum += trans * objs[j].color.xyz * 25.0f;
                    trans = 0.0f; break;
                }
            }
        }
        if (trans <= 0.0f) break;

        // ----- Equatorial accretion-disk emission. -----
        // The "disk slab" is |y_cart| < disk_h with rh = √(x² + z²) ∈ (r_in, r_out).
        // In BL with y as spin axis, y_cart = r cos θ and rh = r sin θ.
        float y_disk = r * cos_th;
        if (abs(y_disk) < disk_h) {
            float rh = r * sin_th;
            if (rh > r_in && rh < r_out) {
                float sin_ph = sin(ph);
                float cos_ph = cos(ph);
                float3 pos_cart = bl_to_cart(r, sin_th, cos_th, sin_ph, cos_ph);

                float2 fluid_uv = float2(pos_cart.x, pos_cart.z) / 50.0f + 0.5f;
                constexpr sampler s(filter::linear);
                float2 f_val = fluidTex.sample(s, fluid_uv).xy;
                float noise_val = (float)fbm_half3(float3(rh * 1.8f, ph * 5.0f, sys.time * 0.1f), sys.time * 0.02f);
                noise_val = mix(noise_val, f_val.x, 0.6f + 0.4f * f_val.y);

                float y_frac = 1.0f - abs(y_disk) / disk_h;

                // Novikov-Thorne disk: Page-Thorne emissivity & temperature profile.
                float r_ratio    = r_in / rh;
                float emission   = r_ratio * r_ratio * r_ratio;
                float inner_edge = smoothstep(0.0f, 0.15f, (rh - r_in) / max(r_in, 0.01f));
                float outer_fade = 1.0f - smoothstep(0.7f, 1.0f, rh / r_out);
                float density    = inner_edge * outer_fade * emission * (y_frac * y_frac * y_frac)
                                 * (0.15f + 0.85f * noise_val) * sys.disk_density;

                float T_base     = pow(r_ratio, 0.75f);
                float T_boundary = pow(max(1.0f - sqrt(r_ratio), 0.001f), 0.25f);
                float T_norm     = T_base * T_boundary;
                half t = half(clamp(T_norm, 0.0f, 1.0f));
                half3 dCol = mix(half3(0.12h, 0.01h, 0.0h),
                                 half3(1.0h,  0.4h,  0.06h),
                                 smoothstep(half(0.0h), half(0.3h), t));
                dCol = mix(dCol, half3(1.0h, 0.75h, 0.3h),
                           smoothstep(half(0.3h), half(0.6h), t));
                dCol = mix(dCol, half3(1.0h, 0.95h, 0.85h),
                           smoothstep(half(0.6h), half(0.9h), t));

                // ----- Exact Kerr g-factor for a Keplerian-orbiting emitter. -----
                // Kerr prograde Keplerian Ω in rs units: Ω = 1 / (√2 r^(3/2) + a),
                // where a = (a/M)/2 is the dimensional spin in rs units.
                float a_real    = kerr.a;
                float r_sqrt    = sqrt(rh);
                float Omega_K   = 1.0f / (1.41421356f * rh * r_sqrt + a_real);
                // Equatorial metric components at rh (sin²θ = 1).
                float gtt_eq    = -(1.0f - 1.0f / rh);
                float gtphi_eq  = -a_real / rh;
                float gphph_eq  = (rh * rh + a_real * a_real
                                   + a_real * a_real / rh);
                // Emitter's u^t time-component normalization.
                float U_denom   = -gtt_eq - 2.0f * gtphi_eq * Omega_K - gphph_eq * Omega_K * Omega_K;
                float Ut        = rsqrt(max(U_denom, 1e-6f));
                // Photon energy in emitter frame: ω_e = U^t (E − Ω L_z).
                // The infinity-to-emitter frequency ratio is 1/(U^t (1 − Ω L_z)).
                float g_factor  = 1.0f / max(Ut * (1.0f - Omega_K * kerr.L), 1e-3f);
                // Bardeen-Press-Teukolsky g⁴ law (bolometric I_obs/I_emit = g⁴).
                // Cap to bound HDR spikes at the inner disk limb where the
                // approaching limb factor diverges.
                float g4 = min(g_factor * g_factor * g_factor * g_factor, 15.0f);

                // Spectral tint from the Doppler shift (visual flair only).
                half shift_t   = half(clamp((g_factor - 1.0f) * 1.5f, -1.0f, 1.0f));
                half3 hot_col  = half3(0.7h, 0.85h, 1.0h);
                half3 cool_col = half3(1.0h, 0.25h, 0.05h);
                dCol = mix(dCol, shift_t > 0.0h ? hot_col : cool_col, abs(shift_t) * 0.4h);

                // Volumetric self-shadowing along +y (toward camera approximately).
                float shadow = 1.0f;
                if (sys.shadow_int > 0.0f) {
                    float3 shadow_p = pos_cart;
                    float shadow_accum = 0.0f;
                    [[unroll]]
                    for (int s = 0; s < 2; s++) {
                        shadow_p += float3(0.0f, 0.2f, 0.0f);
                        float s_rh = length(float2(shadow_p.x, shadow_p.z));
                        if (s_rh > r_in && s_rh < r_out && abs(shadow_p.y) < disk_h)
                            shadow_accum += 0.4f;
                    }
                    shadow = exp(-shadow_accum * sys.shadow_int * 5.0f);
                }

                // Foreshortening — primary image uses the camera-frame angle,
                // secondary uses the lensed angle (matches the "crossbar" in
                // every published GRRT visualization).
                float foreshorten_n_th = (disk_crossings > 0)
                                        ? abs(s_th * sqrt(max(prev_Th, 0.0f))) / max(r, 1e-3f)
                                        : init_abs_n_th;
                float foreshorten = smoothstep(0.0f, 0.25f, foreshorten_n_th);
                foreshorten = max(foreshorten, 0.005f);

                float step_opacity = clamp(density * dlam * 2.5f * foreshorten, 0.0f, 0.9f);
                float crossing_boost = 1.0f + min(float(disk_crossings), 3.0f) * 2.0f;
                col_accum += trans * float3(dCol) * 35.0f * g4 * step_opacity * shadow * crossing_boost;
                trans *= (1.0f - step_opacity);
            }
        }

        // ----- Near-BH glow + ergosphere shimmer. -----
        if (r < 5.0f) {
            float glow = 0.0006f / (r * r);
            col_accum += trans * float3(0.12f, 0.07f, 0.04f) * glow * 60.0f;
            trans *= (1.0f - glow);

            // Ergosphere (equatorial static-limit r_E = 1 in rs units for any spin).
            float a_abs = abs(sys.spin);
            if (a_abs > 0.01f && r < 1.05f && r > r_horizon) {
                float ergo_depth = (1.05f - r) / (1.05f - r_horizon);
                float ergo_glow  = ergo_depth * ergo_depth * 0.015f * a_abs;
                col_accum += trans * float3(0.2f, 0.05f, 0.35f) * ergo_glow * 30.0f;
            }
        }

        // ----- Polar relativistic jets. -----
        if (sys.jet_int > 0.0f && r > 1.2f && r < 25.0f) {
            float cos_axis = abs(cos_th);    // |cos θ| = alignment with spin axis
            if (cos_axis > 0.92f) {
                float jet_core    = smoothstep(0.92f, 0.98f, cos_axis);
                float jet_falloff = exp(-r * 0.15f);
                float sin_ph_j = sin(ph), cos_ph_j = cos(ph);
                float3 pj      = bl_to_cart(r, sin_th, cos_th, sin_ph_j, cos_ph_j);
                float jet_turb = 0.7f + 0.3f * (float)noise_half(pj * 3.0f + sys.time * 0.5f);
                float jet_density = jet_core * jet_falloff * jet_turb * 0.08f * sys.jet_int;
                half3 jet_col  = mix(half3(0.4h, 0.6h, 1.0h), half3(0.8h, 0.9h, 1.0h), half(jet_core));
                col_accum += trans * float3(jet_col) * 20.0f * jet_density;
                trans *= (1.0f - jet_density);
            }
        }

        if (trans < 0.005f) break;
    }

    // ----- Background sample at escape using exit direction. -----
    if (trans > 0.005f) {
        float cos_th = cos(th);
        float sin_th = sqrt(max(1.0f - cos_th * cos_th, 1e-9f));
        float sin_ph_e = sin(ph), cos_ph_e = cos(ph);
        float3 r_hat_e, th_hat_e, ph_hat_e;
        bl_basis(sin_th, cos_th, sin_ph_e, cos_ph_e, r_hat_e, th_hat_e, ph_hat_e);

        // Exit momentum signs scaled with √R/√Θ — at large r the spatial
        // direction in the local frame is (p^r, p^θ·r, p^φ·r·sin θ)/|·|,
        // which reduces to (s_r√R/Δ, s_th √Θ/r, L_z/(r sin θ)) up to a norm.
        float Rn  = kerr_R(r, kerr);
        float Thn = kerr_Theta(cos_th, sin_th, kerr);
        float Delta_e = max(kerr_Delta(r, kerr.a2, kerr.Q_ch_sq), 1e-6f);
        float n_r_out  = s_r  * sqrt(max(Rn,  0.0f)) / Delta_e;
        float n_th_out = s_th * sqrt(max(Thn, 0.0f)) / max(r, 1e-3f);
        float n_ph_out = kerr.L / max(r * sin_th, 1e-3f);
        float norm     = rsqrt(n_r_out * n_r_out + n_th_out * n_th_out + n_ph_out * n_ph_out + 1e-12f);
        float3 exit_dir = (n_r_out * r_hat_e + n_th_out * th_hat_e + n_ph_out * ph_hat_e) * norm;

        float3 bg = (float3)sampleBackground_half(exit_dir, sys.time, sys.star_scint, sys.nebula_int);
        for (int j = 0; j < u_obj.count; j++) {
            if (j == u_obj.bh_index) continue;
            float3 star_dir = normalize(objs[j].posRadius.xyz - bhPos);
            float star_dist = length(objs[j].posRadius.xyz - bhPos);
            float ang_radius = objs[j].posRadius.w / star_dist;
            float cos_angle  = dot(exit_dir, star_dir);
            float threshold  = 1.0f - ang_radius * ang_radius * 0.5f;
            if (cos_angle > threshold) {
                float solid_angle = ang_radius * ang_radius;
                float brightness  = min(solid_angle * 800.0f, 4.0f);
                float limb        = smoothstep(threshold, 1.0f, cos_angle);
                bg = mix(bg, objs[j].color.xyz * brightness, limb);
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
                               texture2d<float, access::sample> bloomTex [[texture(3)]],
                               constant SystemUniforms& sys [[buffer(0)]],
                               uint2 pix [[thread_position_in_grid]]) {
    uint w = outTex.get_width(); uint h = outTex.get_height();
    if (pix.x >= w || pix.y >= h) return;

    float3 col = inTex.read(pix).rgb;
    float2 uv = (float2(pix) + 0.5f) / float2(w, h);

    // MPS bloom composite (half-res blurred bright regions)
    if (sys.enable_bloom != 0) {
        float3 bloom = bloomTex.sample(sampler(filter::linear), uv).rgb;
        col += bloom * 0.6f;
    }

    // Anamorphic flare — 21-tap horizontal scan; skip the reads when disabled.
    if (sys.flare_int > 0.0f) {
        float flare = 0.0f;
        for (int i=-10; i<=10; i++) {
            uint2 p = uint2(clamp(int(pix.x) + i*4, 0, int(w)-1), pix.y);
            flare += max(0.0f, dot(inTex.read(p).rgb, float3(0.2126f, 0.7152f, 0.0722f)) - sys.bloom_threshold);
        }
        col += float3(0.1f, 0.3f, 1.0f) * (flare / 21.0f) * sys.flare_int;
    }

    // Auto-exposure
    col *= sys.exposure;

    // Motion blur
    col = mix(col, accumTex.sample(sampler(filter::linear), uv).rgb, sys.motion_blur);
    col += (hash13(float3(uv * 100.0f, sys.time)) - 0.5f) * sys.film_grain;



    // Optical Vignette — darken edges like a real lens
    float vignette = 1.0f - 0.4f * dot(uv - 0.5f, uv - 0.5f) * 4.0f;
    col *= max(vignette, 0.0f);

    // ACES Filmic Tonemapping
    float a = 2.51f; float b = 0.03f; float c = 2.43f; float d = 0.59f; float e = 0.14f;
    col = clamp((col*(a*col+b))/(col*(c*col+d)+e), 0.0f, 1.0f);
    outTex.write(float4(pow(col, 0.4545f), 1.0f), pix);
}

// --- BLOOM EXTRACTION ---

kernel void bloom_extract(texture2d<float, access::read> inTex [[texture(0)]],
                          texture2d<float, access::write> outTex [[texture(1)]],
                          constant SystemUniforms& sys [[buffer(0)]],
                          uint2 pix [[thread_position_in_grid]]) {
    uint w = outTex.get_width(); uint h = outTex.get_height();
    if (pix.x >= w || pix.y >= h) return;

    // Disabled: write zero so the subsequent MPS blur sees nothing.
    if (sys.enable_bloom == 0) { outTex.write(float4(0.0f), pix); return; }

    // Box-sample 2x2 from full-res → half-res
    uint2 src = pix * 2;
    uint sw = inTex.get_width(); uint sh = inTex.get_height();
    float3 c0 = inTex.read(min(src, uint2(sw-1, sh-1))).rgb;
    float3 c1 = inTex.read(min(src + uint2(1,0), uint2(sw-1, sh-1))).rgb;
    float3 c2 = inTex.read(min(src + uint2(0,1), uint2(sw-1, sh-1))).rgb;
    float3 c3 = inTex.read(min(src + uint2(1,1), uint2(sw-1, sh-1))).rgb;
    float3 avg = (c0 + c1 + c2 + c3) * 0.25f;

    float3 bloom = max(avg - sys.bloom_threshold, 0.0f);
    outTex.write(float4(bloom, 1.0f), pix);
}

// --- AUTO-EXPOSURE LUMINANCE ANALYSIS ---
//
// Dispatched at (W/4) × (H/4) — each thread samples one pixel out of every
// 4×4 block. Per-thread results are summed across the SIMD group with
// simd_sum, so we issue at most one atomic per simdgroup (32× fewer than
// per-thread atomics on Apple Silicon).
//
// Encoding: log2 luminance ∈ [-10, 10] → [0, 2000] as fixed-point ×100.
// At a 4K-equivalent active-thread count (~520k), 520k × 2000 = 1.04e9
// stays well under uint32 max (4.29e9). The previous ×1000 encoding
// overflowed on Retina framebuffers.
kernel void luminance_reduce(texture2d<float, access::read> inTex [[texture(0)]],
                             device atomic_uint* lumBuffer [[buffer(0)]],
                             uint2 pix [[thread_position_in_grid]],
                             uint simd_lane [[thread_index_in_simdgroup]]) {
    uint w = inTex.get_width(); uint h = inTex.get_height();
    uint2 src = pix * 4u;

    uint encoded = 0u;
    uint count = 0u;
    if (src.x < w && src.y < h) {
        float3 col = inTex.read(src).rgb;
        float lum = dot(col, float3(0.2126f, 0.7152f, 0.0722f));
        float log_lum = log2(max(lum, 0.001f));
        encoded = uint(clamp((log_lum + 10.0f) * 100.0f, 0.0f, 2000.0f));
        count = 1u;
    }

    uint sum_simd = simd_sum(encoded);
    uint cnt_simd = simd_sum(count);
    if (simd_lane == 0u && cnt_simd > 0u) {
        atomic_fetch_add_explicit(lumBuffer, sum_simd, memory_order_relaxed);
        atomic_fetch_add_explicit(lumBuffer + 1, cnt_simd, memory_order_relaxed);
    }
}

// --- N-BODY PHYSICS ---

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
    float total_depression = 0.0f;

    // Visual depth parameters — calibrated to grid extent (±200e12)
    const float BH_WELL_DEPTH = 25.0e12f;   // Gentle funnel, ~6% of grid extent
    const float BH_SOFTENING  = 15.0e12f;   // Wide mouth for smooth conical shape
    const float STAR_WELL_DEPTH = 5.0e12f;  // Gentle stellar dimples
    const float GRID_BASELINE = 1.5e12f;    // Grid sits just below BH

    for (int i = 0; i < uo.count; i++) {
        float3 delta = p - o[i].posRadius.xyz;
        float d = length(delta.xz);
        bool is_bh = o[i].mass > 1e35f;

        if (is_bh) {
            // BH: deep conical funnel — 1/r potential with softening
            total_depression += BH_WELL_DEPTH * BH_SOFTENING / sqrt(d * d + BH_SOFTENING * BH_SOFTENING);
        } else if (o[i].mass > 1e28f) {
            // Stars: gentle dimple bowls scaled by mass ratio
            float star_soft = max(o[i].posRadius.w * 3.0f, 5.0e12f);  // floor prevents tubes
            float mass_ratio = o[i].mass / 12.0e30f;
            total_depression += STAR_WELL_DEPTH * mass_ratio * star_soft / sqrt(d * d + star_soft * star_soft);
        }

        // Gravitational wave ripples
        if (sys.gw_amp > 0.0f && o[i].mass > 1e28f) {
            float mass_scale = is_bh ? 1.0f : 0.08f;
            float phase = d * 1e-11f - sys.time * 2.0f;
            float ripple = sin(phase) * (sys.gw_amp * 2e11f * mass_scale) / (1.0f + d * 1e-12f);
            ripple *= cos(2.0f * atan2(delta.z, delta.x));
            p.y += ripple;
        }
    }

    // The well curves DOWNWARD — p.y decreases near masses
    p.y -= total_depression;

    // Normalized depth for coloring (0 = flat, 1 = deepest)
    float well_depth = clamp(total_depression / BH_WELL_DEPTH, 0.0f, 1.0f);

    // Place flat grid just below the BH equatorial plane
    // BH sits at the RIM of the funnel
    p.y -= GRID_BASELINE;
    out.position = u.viewProj * float4(p, 1.0f);
    out.depth = well_depth;
    return out;
}

fragment float4 grid_fragment(VertexOut in [[stage_in]],
                              texture2d<float, access::sample> sceneTex [[texture(0)]]) {
    float d = in.depth;  // 0 = flat surface, 1 = deep in gravity well
    
    // CRITICAL: flat grid lines (d ≈ 0) MUST be fully discarded.
    if (d < 0.008f) discard_fragment();
    
    // Check the scene behind this fragment — don't draw grid over BH/stars/disk
    float2 screen_uv = in.position.xy / float2(sceneTex.get_width(), sceneTex.get_height());
    constexpr sampler s(filter::linear);
    float3 scene = sceneTex.sample(s, screen_uv).rgb;
    float scene_lum = dot(scene, float3(0.2126f, 0.7152f, 0.0722f));
    if (scene_lum > 0.05f) discard_fragment();  // hide grid behind bright objects
    
    // Color gradient: bright cyan → electric blue-purple → hot orange-red
    float3 col = mix(float3(0.15f, 0.8f, 1.0f),
                     float3(0.5f, 0.2f, 1.0f),
                     smoothstep(0.0f, 0.3f, d));
    col = mix(col,
              float3(1.0f, 0.3f, 0.05f),
              smoothstep(0.3f, 0.75f, d));
    
    // Emissive HDR boost
    col *= 1.5f + d * 1.5f;
    
    // Alpha ramps with depth
    float alpha = smoothstep(0.008f, 0.06f, d) * mix(0.5f, 0.9f, d);
    return float4(col, alpha);
}
