#include <metal_stdlib>
using namespace metal;

// Structs (SimObject, CameraData, SystemUniforms, ObjectsUniform, GridUniforms)
// are provided by ShaderCommon.h, prepended at compile time.

// --- NOISE (background nebula + jets) ---

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

// --- BLACKBODY COLOR ---
// Luminance-normalized linear-sRGB chromaticity of a Planck spectrum,
// indexed by log10(T) in [3.0, 4.6] (1000 K .. ~40000 K, 32 entries).
// Baked offline: Planck x CIE-1931 CMFs (Wyman-Sloan-Shirley fits) -> XYZ -> sRGB.
// A Doppler/gravitationally shifted blackbody is exactly another blackbody at
// T_obs = g * T_emit, so disk color = BB_LUT(g * T) is the physically correct
// treatment (an RGB tint cannot move along the Planckian locus).
constant float3 BB_LUT[32] = {
    float3(4.28961f, 0.12308f, 0.00000f),  //    1000 K
    float3(3.95561f, 0.22237f, 0.00000f),  //    1126 K
    float3(3.61761f, 0.32284f, 0.00000f),  //    1268 K
    float3(3.28964f, 0.42033f, 0.00000f),  //    1428 K
    float3(2.98092f, 0.51210f, 0.00000f),  //    1609 K
    float3(2.69679f, 0.59656f, 0.00000f),  //    1812 K
    float3(2.43484f, 0.67163f, 0.02771f),  //    2040 K
    float3(2.19622f, 0.73653f, 0.08751f),  //    2298 K
    float3(1.98397f, 0.79219f, 0.16113f),  //    2588 K
    float3(1.79676f, 0.83907f, 0.24797f),  //    2914 K
    float3(1.63279f, 0.87784f, 0.34675f),  //    3282 K
    float3(1.49008f, 0.90927f, 0.45562f),  //    3696 K
    float3(1.36662f, 0.93420f, 0.57227f),  //    4163 K
    float3(1.26041f, 0.95347f, 0.69411f),  //    4688 K
    float3(1.16955f, 0.96793f, 0.81842f),  //    5279 K
    float3(1.09224f, 0.97838f, 0.94254f),  //    5946 K
    float3(1.02678f, 0.98558f, 1.06399f),  //    6696 K
    float3(0.97158f, 0.99021f, 1.18065f),  //    7541 K
    float3(0.92519f, 0.99288f, 1.29083f),  //    8492 K
    float3(0.88628f, 0.99410f, 1.39333f),  //    9564 K
    float3(0.85368f, 0.99429f, 1.48745f),  //   10771 K
    float3(0.82635f, 0.99378f, 1.57291f),  //   12130 K
    float3(0.80342f, 0.99284f, 1.64980f),  //   13661 K
    float3(0.78415f, 0.99163f, 1.71846f),  //   15385 K
    float3(0.76790f, 0.99031f, 1.77943f),  //   17326 K
    float3(0.75417f, 0.98895f, 1.83332f),  //   19513 K
    float3(0.74252f, 0.98762f, 1.88081f),  //   21975 K
    float3(0.73261f, 0.98635f, 1.92256f),  //   24748 K
    float3(0.72415f, 0.98517f, 1.95922f),  //   27872 K
    float3(0.71690f, 0.98407f, 1.99138f),  //   31389 K
    float3(0.71067f, 0.98308f, 2.01958f),  //   35350 K
    float3(0.70530f, 0.98218f, 2.04432f),  //   39811 K
};

static inline float3 blackbody_color(float T) {
    float u = (log10(max(T, 1.0f)) - 3.0f) / 1.6f * 31.0f;
    u = clamp(u, 0.0f, 30.999f);
    int   i = int(u);
    float f = u - float(i);
    return mix(BB_LUT[i], BB_LUT[i + 1], f);
}

// Diverging blue-white-red colormap for the redshift lens, centered at t=0.5.
static inline float3 diverging_map(float t) {
    t = clamp(t, 0.0f, 1.0f);
    float3 blue  = float3(0.13f, 0.30f, 0.89f);
    float3 white = float3(0.95f, 0.95f, 0.95f);
    float3 red   = float3(0.85f, 0.10f, 0.09f);
    return (t < 0.5f) ? mix(red, white, t * 2.0f)
                      : mix(white, blue, t * 2.0f - 1.0f);
}

// "Hot" colormap (black -> red -> orange -> white) for the EHT lens.
static inline float3 hot_map(float t) {
    t = clamp(t, 0.0f, 1.0f);
    return float3(clamp(3.0f * t,        0.0f, 1.0f),
                  clamp(3.0f * t - 1.0f, 0.0f, 1.0f),
                  clamp(3.0f * t - 2.0f, 0.0f, 1.0f));
}

// --- BACKGROUND ---

static inline half3 sampleBackground_half(float3 rd, float time, float scint, float nebula_int) {
    // Default: dark, smooth sky. The procedural starfield is a moiré factory
    // under single-sample raycasting through curved spacetime; the cinematic
    // toggles bring it back, and temporal accumulation (jittered rays) tames it.
    float n1 = (float)noise_half(rd * 2.5f + time * 0.04f);
    float n2 = (float)noise_half(rd * 4.5f - time * 0.02f);
    half3 nebula = mix(half3(0.001h, 0.002h, 0.006h),
                      half3(0.012h, 0.006h, 0.024h),
                      (half)n1 * 0.5h + 0.5h) * (half)nebula_int;
    nebula += half3(0.04h, 0.015h, 0.06h) * pow(max(0.0f, n2), 4.0f) * (half)nebula_int;

    half3 col = nebula;
    if (scint > 0.0f) {
        float3 p = rd * 600.0f;
        float3 id = floor(p);
        float h = hash13(id);
        if (h > 0.99995f) {
            float pulse = 1.0f + scint * 0.5f * sin(time * 4.0f + h * 100.0f);
            col += half3(0.9h, 1.0h, 1.2h)
                 * (half)(pulse * pow(hash13(id + 0.5f), 30.0f) * 4.0f);
        }
    }
    return col;
}

// Lat-long checkerboard sky for the lensing-map lens: makes light deflection
// legible with all astrophysics stripped out (DNGR Fig. 13 convention: distinct
// quadrant tints + a marked antipode patch so multiple images are identifiable).
static inline float3 checker_sky(float3 rd, float3 cam_fwd) {
    float th = acos(clamp(rd.y, -1.0f, 1.0f));
    float ph = atan2(rd.z, rd.x);
    const float cell = 0.2617994f;             // 15 degrees
    float parity = fmod(floor(th / cell) + floor((ph + 3.14159265f) / cell), 2.0f);
    float3 base = (parity < 0.5f) ? float3(0.05f, 0.06f, 0.09f) : float3(0.55f, 0.58f, 0.65f);
    // Quadrant tint: hemisphere (up/down) x (front/back relative to phi=0).
    float3 tint = float3(1.0f);
    if (rd.y >  0.0f && cos(ph) >  0.0f) tint = float3(1.0f, 0.75f, 0.75f);
    if (rd.y >  0.0f && cos(ph) <= 0.0f) tint = float3(0.75f, 1.0f, 0.75f);
    if (rd.y <= 0.0f && cos(ph) >  0.0f) tint = float3(0.75f, 0.80f, 1.0f);
    if (rd.y <= 0.0f && cos(ph) <= 0.0f) tint = float3(1.0f, 1.0f, 0.72f);
    float3 col = base * tint;
    // Antipode patch (directly behind the camera): every appearance of this red
    // spot in the image is one more image order of the same sky point.
    if (dot(rd, -cam_fwd) > 0.9986f) col = float3(0.9f, 0.08f, 0.08f);
    return col;
}

// =====================================================================
// EXACT KERR-NEWMAN NULL GEODESICS — super-Hamiltonian first-order form.
//
//   Boyer-Lindquist coordinates, Carter-separated potentials, but the
//   integrator evolves the MOMENTA (p_r, p_theta) with Hamilton's
//   equations instead of the +/-sqrt(R), +/-sqrt(Theta) root-tracking
//   form. The root form freezes at turning points (sqrt(max(R,0)) has
//   zero derivative in the forbidden region); the Hamiltonian form
//   passes through them smoothly — the same reason DNGR (James et al.
//   2015, CQG 32 065001, Eq. A.15) chose it.
//
// Conventions (matching the host's slider semantics):
//   sys.spin    = a/M in [-1, 1]   (dimensionless Kerr spin)
//   sys.charge  = Q/M in [0, 1]    (dimensionless charge)
//   r, lambda   in rs = 2M units, so M = 1/2 and
//   Delta = r^2 - r + a^2 + Q^2,   "a" below = (a/M)/2,  Q = (Q/M)/2.
//
// Null-geodesic conserved quantities (E = -p_t = 1 by affine choice):
//   L  = p_phi
//   QC = p_theta^2 + cos^2(th) (L^2/sin^2(th) - a^2)      (Carter, mu = 0)
//
// On-shell (H = 0) Hamilton equations:
//   dr/dl      = Delta p_r / Sigma
//   dth/dl     = p_theta / Sigma
//   dphi/dl    = [a (r - Q^2) - a^2 L] / (Delta Sigma) + L / (Sigma sin^2 th)
//   dp_r/dl    = [ d/dr (R/Delta) - Delta' p_r^2 ] / (2 Sigma)
//   dp_th/dl   = [ -2 a^2 sin th cos th + 2 L^2 cos th / sin^3 th ] / (2 Sigma)
// with R(r) = P^2 - Delta [(L-a)^2 + QC],  P = r^2 + a^2 - a L,
//      Sigma = r^2 + a^2 cos^2 th,  Delta' = 2r - 1.
// The null condition Delta p_r^2 + p_th^2 = R/Delta + Theta(th) is enforced
// by the initial conditions and preserved to integrator accuracy.
// =====================================================================

struct KerrConst {
    float a;        // spin in rs units (= sys.spin / 2), signed
    float a2;
    float Q2;       // charge^2 in rs units (= (sys.charge / 2)^2)
    float E;        // conserved energy of the traced ray: -1 (past-directed).
                    // The pixel's light ARRIVES at the camera; its backward
                    // extension is the past-directed continuation of the
                    // arriving momentum. Tracing a future-directed E=+1 ray
                    // along the view direction instead integrates the TIME-
                    // REVERSED photon — harmless in Schwarzschild (t -> -t is
                    // an isometry) but mirrored in Kerr, which pairs the
                    // shadow's flattened side with the WRONG Doppler side.
    float L;        // traced ray's conserved L_z (for E = -1)
    float QC;       // Carter constant
    float L_arr;    // the photon's impact ratio xi = L/E = -L: used by the
                    // emitter g-factor g = 1/(U^t (1 - Omega xi)).
};

static inline float kerr_Sigma(float r, float cos_th, float a2) {
    return r * r + a2 * cos_th * cos_th;
}

static inline float kerr_Delta(float r, float a2, float Q2) {
    return r * r - r + a2 + Q2;
}

static inline float kerr_R(float r, KerrConst k) {
    float P = k.E * (r * r + k.a2) - k.a * k.L;
    float B = (k.L - k.a * k.E) * (k.L - k.a * k.E) + k.QC;
    return P * P - kerr_Delta(r, k.a2, k.Q2) * B;
}

static inline float kerr_Theta(float cos_th, float sin_th, KerrConst k) {
    float s2 = max(sin_th * sin_th, 1e-4f);
    return k.QC + k.a2 * cos_th * cos_th - k.L * k.L * cos_th * cos_th / s2;
}

// Geodesic state and its lambda-derivative.
struct GeoState { float r, th, ph, pr, pth; };

static inline GeoState kerr_rhs(GeoState s, KerrConst k) {
    float c = cos(s.th);
    float sn = sin(s.th);
    float sn_g   = (sn >= 0.0f) ? max(sn, 1e-4f) : min(sn, -1e-4f);  // pole guard
    float s2     = sn_g * sn_g;
    float Sigma  = kerr_Sigma(s.r, c, k.a2);
    float Delta  = kerr_Delta(s.r, k.a2, k.Q2);
    float Dp     = 2.0f * s.r - 1.0f;                                 // Delta'
    float P      = k.E * (s.r * s.r + k.a2) - k.a * k.L;
    float B      = (k.L - k.a * k.E) * (k.L - k.a * k.E) + k.QC;
    float R      = P * P - Delta * B;
    float Rp     = 4.0f * s.r * P * k.E - Dp * B;                     // R'
    float D2     = Delta * Delta;
    float RoDp   = (Rp * Delta - R * Dp) / max(D2, 1e-12f);           // (R/Delta)'

    GeoState d;
    float inv_Sigma = 1.0f / Sigma;
    d.r   = Delta * s.pr * inv_Sigma;
    d.th  = s.pth * inv_Sigma;
    d.ph  = ((k.a * (k.E * (s.r - k.Q2)) - k.a2 * k.L) / max(Delta, 1e-9f) + k.L / s2) * inv_Sigma;
    d.pr  = (RoDp - Dp * s.pr * s.pr) * 0.5f * inv_Sigma;
    d.pth = (-2.0f * k.a2 * sn * c + 2.0f * k.L * k.L * c / (s2 * sn_g)) * 0.5f * inv_Sigma;
    return d;
}

static inline GeoState geo_add(GeoState s, GeoState d, float h) {
    GeoState o;
    o.r = s.r + h * d.r;  o.th = s.th + h * d.th;  o.ph = s.ph + h * d.ph;
    o.pr = s.pr + h * d.pr;  o.pth = s.pth + h * d.pth;
    return o;
}

// One RK4 step. No sign tracking: p_r, p_theta cross zero smoothly.
static inline GeoState kerr_rk4(GeoState s, float h, KerrConst k) {
    GeoState k1 = kerr_rhs(s, k);
    GeoState k2 = kerr_rhs(geo_add(s, k1, 0.5f * h), k);
    GeoState k3 = kerr_rhs(geo_add(s, k2, 0.5f * h), k);
    GeoState k4 = kerr_rhs(geo_add(s, k3, h), k);
    GeoState o;
    o.r   = s.r   + h * (k1.r   + 2.0f * (k2.r   + k3.r)   + k4.r)   / 6.0f;
    o.th  = s.th  + h * (k1.th  + 2.0f * (k2.th  + k3.th)  + k4.th)  / 6.0f;
    o.ph  = s.ph  + h * (k1.ph  + 2.0f * (k2.ph  + k3.ph)  + k4.ph)  / 6.0f;
    o.pr  = s.pr  + h * (k1.pr  + 2.0f * (k2.pr  + k3.pr)  + k4.pr)  / 6.0f;
    o.pth = s.pth + h * (k1.pth + 2.0f * (k2.pth + k3.pth) + k4.pth) / 6.0f;
    return o;
}

// BL (r, th, ph) -> BH-local Cartesian (rs units), y = spin axis.
static inline float3 bl_to_cart(float r, float sin_th, float cos_th, float sin_ph, float cos_ph) {
    return float3(r * sin_th * cos_ph, r * cos_th, r * sin_th * sin_ph);
}

// Local spherical basis at (r, th, ph) in Cartesian (rs units).
static inline void bl_basis(float sin_th, float cos_th, float sin_ph, float cos_ph,
                            thread float3& r_hat, thread float3& th_hat, thread float3& ph_hat) {
    r_hat  = float3(sin_th * cos_ph,  cos_th,           sin_th * sin_ph);
    th_hat = float3(cos_th * cos_ph, -sin_th,           cos_th * sin_ph);
    ph_hat = float3(-sin_ph,           0.0f,            cos_ph);
}

// =====================================================================
// Optically-thick Novikov-Thorne surface emission with the physical
// temperature-shift color pipeline:
//
//   F(r)  ∝ (r_in/r)^3 (1 − sqrt(r_in/r))     (Page-Thorne flux; zero-torque
//                                              BC: F = 0 at r_in, peak at
//                                              r = (49/36) r_in)
//   T(r)  = T_peak · Fn(r)^{1/4}               (Fn = F normalized to peak 1)
//   g     = 1 / [U^t (1 − Ω L_arr)]            (covariant redshift+Doppler,
//                                              KN metric, ARRIVING photon L)
//   T_obs = g · T                              (shifted blackbody is a
//                                              blackbody at gT)
//   I_obs = Fn · g^4 (capped)                  (bolometric Liouville)
//
// The Kerr-Newman circular-orbit frequency (rs units, M = 1/2):
//   Ω = sqrt(M r − Q^2) / (r^2 + a sqrt(M r − Q^2)),  signed a: for
//   spin < 0 the flow (always +phi) is retrograde relative to the hole
//   and the host supplies the RETROGRADE ISCO as r_in.
// =====================================================================
static float3 disk_surface_emission(
    float rh,
    KerrConst kerr,
    constant SystemUniforms& sys,
    float r_in,
    float r_out,
    thread float& g_out)
{
    float r_ratio = r_in / rh;
    float flux    = r_ratio * r_ratio * r_ratio * max(1.0f - sqrt(r_ratio), 0.0f);
    // Normalize: peak of x^3(1-sqrt(x)) over x = r_in/r is at x = 36/49.
    const float x_pk = 36.0f / 49.0f;
    const float f_pk = x_pk * x_pk * x_pk * (1.0f /* - sqrt(36/49) = 1/7 */) / 7.0f;
    float Fn = flux / f_pk;

    float outer_fade = 1.0f - smoothstep(r_out * 0.7f, r_out, rh);

    // Kerr-Newman equatorial metric + circular-orbit frequency (charge-aware).
    float m_kn   = (rh - kerr.Q2) / (rh * rh);        // (2Mr - Q^2)/r^2 in rs units
    float gtt_eq   = -(1.0f - m_kn);
    float gtphi_eq = -kerr.a * m_kn;
    float gphph_eq =  rh * rh + kerr.a2 + kerr.a2 * m_kn;
    float s_kn     = sqrt(max(0.5f * rh - kerr.Q2, 0.0f));
    float Omega    = s_kn / (rh * rh + kerr.a * s_kn);

    float U_denom  = -gtt_eq - 2.0f * gtphi_eq * Omega - gphph_eq * Omega * Omega;
    if (U_denom <= 1e-6f) { g_out = 1.0f; return float3(0.0f); }  // spacelike orbit: no emitter
    float Ut       = rsqrt(U_denom);
    float g        = 1.0f / max(Ut * (1.0f - Omega * kerr.L_arr), 1e-3f);
    g_out = g;

    float T_emit = sys.disk_temp * pow(max(Fn, 1e-6f), 0.25f);
    float T_obs  = T_emit;
    float boost  = 1.0f;
    if (sys.beaming_mode == BEAM_FULL) {
        T_obs = g * T_emit;
        boost = min(g * g * g * g, 15.0f);   // BPT g^4 cap: keeps the approaching
                                             // inner limb inside tonemap range
    } else if (sys.beaming_mode == BEAM_NO_BOOST) {
        T_obs = g * T_emit;
    }

    float3 chroma = blackbody_color(T_obs);
    // The 18.0 multiplier is the only artistic knob: calibrated so the disk
    // reads mid-gray after auto-exposure on the default scene. Shape and
    // ratios come from the Page-Thorne + Liouville physics above.
    return chroma * Fn * boost * outer_fade * sys.disk_density * 18.0f;
}

// =====================================================================
// Optically-thin volumetric plasma torus with covariant radiative transfer.
//
// Geometry and flow follow the EHT code-comparison parameterization
// (Gold et al. 2020, ApJ 897:148) — an analytic stand-in for a GRMHD
// snapshot with the same qualitative structure, and the setup that paper
// uses for its cross-code imaging tests:
//
//   n(r, th) = n0 exp(-[((r - r0)/w)^2 + (h cos th)^2] / 2),  w = 0.4 r0
//   l(R)     = (l0 / (1 + R)) R^(3/2),   R = r sin th
//
// (Gold et al. write the radial factor as exp(-(r/r0)^2/2), which peaks at
// the origin — fine for their test geometry, where the peak sits inside the
// horizon, but it renders as central fog rather than a torus. A Gaussian
// RING at r0 is the standard analytic stand-in and changes none of the
// transport physics: the Doppler-cancellation identity below is a property
// of the frequency dependence, not of the density profile.)
//
// l is the specific angular momentum -u_phi/u_t, which fixes the orbital
// frequency from the metric alone:
//
//   Omega = -(g_tphi + l g_tt) / (g_phph + l g_tphi)
//
// — a genuinely non-Keplerian rotating flow, unlike the thin disk.
//
// Transport is the covariant radiative-transfer equation in INVARIANT
// form (Mihalas & Mihalas; the formulation ipole/RAPTOR integrate):
//
//   d(I_nu / nu^3)/dlambda = (j_nu / nu^2) - (nu alpha_nu)(I_nu / nu^3)
//
// with fluid-frame quantities evaluated at nu' = nu_obs / g. Writing the
// power-law emissivity j_nu = C n nu^-alpha_s in invariant form gives
//
//   J_inv = C n g^(alpha_s + 2),   A_inv = A C n g^(alpha_s + 1.5)
//
// so the Doppler asymmetry EMERGES from the transport instead of being
// applied by hand — and at alpha_s = -2 it cancels exactly, the
// non-trivial symmetry of Gold et al. Test 2 that the suite checks.
//
// Color remains the documented visible-light remap: a blackbody at the
// g-shifted local temperature, as for the thin disk.
// =====================================================================

struct TorusSample {
    float n;        // density (normalized to 1 at the torus core)
    float g;        // redshift factor nu_obs / nu_emit
    float T;        // local (unshifted) color temperature
};

static inline bool torus_sample(float r_rs, float cos_th, float sin_th,
                                KerrConst k, constant SystemUniforms& sys,
                                thread TorusSample& out)
{
    // The literature parameterization is in M units; ours is rs = 2M.
    float r_M   = 2.0f * r_rs;
    float R_cyl = r_M * sin_th;

    float r0 = max(sys.torus_r0, 0.5f);
    float ar = (r_M - r0) / (0.4f * r0);
    float az = sys.torus_h * cos_th;
    float n  = exp(-0.5f * (ar * ar + az * az));
    if (n < 2.0e-3f) return false;                 // outside the emitting body

    // Specific angular momentum (M units) -> rs units (l has dimension length).
    float l_rs = 0.5f * (sys.torus_l0 / (1.0f + R_cyl)) * pow(max(R_cyl, 1e-4f), 1.5f);

    // Kerr-Newman metric components at this point (general theta, rs units).
    float Sigma  = kerr_Sigma(r_rs, cos_th, k.a2);
    float s2     = max(sin_th * sin_th, 1e-6f);
    float m_kn   = r_rs - k.Q2;
    float gtt    = -(1.0f - m_kn / Sigma);
    float gtphi  = -k.a * m_kn * s2 / Sigma;
    float gphph  = ((r_rs * r_rs + k.a2) * Sigma + k.a2 * m_kn * s2) * s2 / Sigma;

    // l = -u_phi/u_t  =>  Omega, then the four-velocity normalization.
    float denom = gphph + l_rs * gtphi;
    denom = (abs(denom) > 1e-9f) ? denom : copysign(1e-9f, denom);
    float Omega = -(gtphi + l_rs * gtt) / denom;

    float U_denom = -(gtt + 2.0f * Omega * gtphi + Omega * Omega * gphph);
    if (U_denom <= 1e-6f) return false;            // spacelike flow: no emitter
    float Ut = rsqrt(U_denom);

    // Same covariant redshift as the thin disk, with the torus Omega:
    //   g = 1 / [u^t (1 - Omega xi)],  xi = L/E the photon impact ratio.
    out.g = 1.0f / max(Ut * (1.0f - Omega * k.L_arr), 1e-3f);
    out.n = n;
    // Visible-light color remap: hotter toward the centre (virial scaling).
    out.T = sys.torus_temp * sqrt(max(sys.torus_r0, 0.5f) / max(r_M, 1.0f));
    return true;
}

kernel void raytrace(texture2d<float, access::write> out [[texture(0)]],
                     constant CameraData& cam [[buffer(0)]],
                     const device SimObject* objs [[buffer(1)]],
                     constant ObjectsUniform& u_obj [[buffer(2)]],
                     constant SystemUniforms& sys [[buffer(3)]],
                     uint2 pix [[thread_position_in_grid]]) {
    uint w = out.get_width(); uint h = out.get_height();
    if (pix.x >= w || pix.y >= h) return;

    // Subpixel jitter (Halton, from the host): with temporal accumulation this
    // converges to supersampled ground truth on a static camera.
    float px = float(pix.x) + 0.5f + sys.jitter_x;
    float py = float(pix.y) + 0.5f + sys.jitter_y;
    float ur = (2.0f * px / float(w) - 1.0f) * cam.aspect * cam.tanHalfFov;
    float vr = (1.0f - 2.0f * py / float(h)) * cam.tanHalfFov;
    float3 rd = normalize(ur * cam.camRight.xyz + vr * cam.camUp.xyz + cam.camForward.xyz);
    float3 ro = cam.camPos.xyz;

    if (u_obj.bh_index < 0 || u_obj.bh_index >= u_obj.count) return;
    const device SimObject* bh = &objs[u_obj.bh_index];
    float rs = bh->posRadius.w;
    float3 bhPos = bh->posRadius.xyz;
    float3 ro_bh = (ro - bhPos) / rs;

    KerrConst kerr;
    kerr.a  = 0.5f * sys.spin;
    kerr.a2 = kerr.a * kerr.a;
    float Q_ch = 0.5f * sys.charge;
    kerr.Q2 = Q_ch * Q_ch;

    float r_horizon = sys.r_horizon;
    const float r_in  = max(sys.r_isco, r_horizon * 1.2f);
    const float r_out = sys.disk_r_out;

    // ----- Camera Cartesian -> Boyer-Lindquist. -----
    float r_obs       = length(ro_bh);
    float cos_th_obs  = clamp(ro_bh.y / r_obs, -1.0f, 1.0f);
    float sin_th_obs  = sqrt(max(1.0f - cos_th_obs * cos_th_obs, 1e-9f));
    float phi_obs     = atan2(ro_bh.z, ro_bh.x);
    float sin_ph_obs  = sin(phi_obs);
    float cos_ph_obs  = cos(phi_obs);

    float3 r_hat, th_hat, ph_hat;
    bl_basis(sin_th_obs, cos_th_obs, sin_ph_obs, cos_ph_obs, r_hat, th_hat, ph_hat);
    float n_r  = dot(rd, r_hat);
    float n_th = dot(rd, th_hat);
    float n_ph = dot(rd, ph_hat);

    // ----- ZAMO tetrad initial conditions (Kerr-Newman: 2Mr -> r - Q^2). -----
    //   alpha^2 = -g_tt + g_tphi^2 / g_phph      (lapse)
    //   omega_z = -g_tphi / g_phph               (frame-drag rate)
    // With affine parameter chosen so E = 1:
    //   L    =  sqrt(g_phph) n_phi / (alpha + omega_z sqrt(g_phph) n_phi)
    //   p_r  =  sqrt(g_rr)   n_r   / (same denominator)
    //   p_th =  sqrt(g_thth) n_th  / (same denominator)
    float Sigma_obs = kerr_Sigma(r_obs, cos_th_obs, kerr.a2);
    float Delta_obs = kerr_Delta(r_obs, kerr.a2, kerr.Q2);
    float s2_obs    = sin_th_obs * sin_th_obs;
    float m_kn      = r_obs - kerr.Q2;               // 2Mr - Q^2 in rs units
    float gtt_neg   = 1.0f - m_kn / Sigma_obs;
    float gtphi     = -kerr.a * m_kn * s2_obs / Sigma_obs;
    float grr       = Sigma_obs / max(Delta_obs, 1e-9f);
    float gthth     = Sigma_obs;
    float gphph     = ((r_obs * r_obs + kerr.a2) * Sigma_obs
                       + kerr.a2 * m_kn * s2_obs) * s2_obs / Sigma_obs;

    float gphph_safe = max(gphph, 1e-9f);
    float omega_z    = -gtphi / gphph_safe;
    float alpha      = sqrt(max(gtt_neg + gtphi * gtphi / gphph_safe, 1e-9f));
    float sqrt_gphph = sqrt(max(gphph, 0.0f));

    // Past-directed backward ray: the arriving photon's momentum, continued
    // into the past, normalized so E = -p_t = -1. In the ZAMO frame this
    // gives the (alpha - omega_z ...) denominator — the future-directed
    // (alpha + ...) normalization would trace the time-reversed photon and
    // mirror every frame-dragging asymmetry.
    float E_setup     = alpha - omega_z * sqrt_gphph * n_ph;
    float E_setup_inv = 1.0f / max(E_setup, 1e-6f);
    kerr.E      = -1.0f;
    kerr.L      = sqrt_gphph  * n_ph * E_setup_inv;
    float p_r0  = sqrt(grr)   * n_r  * E_setup_inv;
    float p_th0 = sqrt(gthth) * n_th * E_setup_inv;
    // The photon's impact ratio xi = L/E = -L drives the emitter g-factor.
    kerr.L_arr = -kerr.L;

    {
        float c2 = cos_th_obs * cos_th_obs;
        kerr.QC = p_th0 * p_th0 + c2 * (kerr.L * kerr.L / max(s2_obs, 1e-9f) - kerr.a2);
    }

    GeoState s;
    s.r = r_obs; s.th = acos(cos_th_obs); s.ph = phi_obs;
    s.pr = p_r0; s.pth = p_th0;

    float prev_cos = cos_th_obs;
    float prev_r   = r_obs;
    float prev_ph  = phi_obs;
    // Polar-cap phi bypass: inside th < CAP of the axis, dphi/dlambda is
    // numerically meaningless (BL coordinate singularity, clamped 1/sin^2
    // integrand). Freeze phi at cap entry and apply the exact through-the-pole
    // jump of pi on exit — exact in spherical symmetry, O(a * CAP) for Kerr.
    const float POLE_CAP = 0.01f;
    bool  in_pole_cap = false;
    float ph_cap_entry = 0.0f;
    int   crossings = 0;         // equatorial crossings BEFORE the accepted hit
    int   hit_order = -1;        // image order n of the accepted disk hit
    float hit_g     = 1.0f;      // g-factor at the accepted disk hit
    float g_accum   = 0.0f;      // emission-weighted g (volumetric lens readout)
    float g_weight  = 0.0f;
    // Volumetric emitting body extent: the density profile is Gaussian, so
    // 4 scale radii is where n falls below the 2e-3 sampling threshold.
    const float vol_r_max = 0.5f * (sys.torus_r0 * 2.4f) + 1.0f;
    int   fate      = 0;         // 0 = budget exhausted, 1 = captured, 2 = escaped,
                                 // 3 = disk hit, 4 = absorbed in foreground media

    const float r_esc = max(500.0f, r_obs * 1.05f);

    float3 col_accum = float3(0.0f);
    float  trans     = 1.0f;

    for (int i = 0; i < 1200; i++) {
        // Capture: trans = 0 keeps the background out (the escape block is
        // gated on fate), but emission already accumulated BETWEEN the camera
        // and the horizon (jets, glow) is real foreground light and is kept —
        // zeroing it would notch the jet where it crosses the shadow.
        if (s.r < r_horizon * 1.02f) { fate = 1; trans = 0.0f; break; }
        if (s.r > r_esc)             { fate = 2; break; }
        if (s.r > 30.0f && s.pr > 0.0f) { fate = 2; break; }

        // ----- Adaptive affine step (RAPTOR-style inverse-sum controller). -----
        // Step shrinks where any coordinate moves fast (photon-shell whirls,
        // near-horizon approach) and grows to ~0.05 r in the far field.
        GeoState d = kerr_rhs(s, kerr);
        // Pole treatment (RAPTOR-style): the polar-rate term scales inversely
        // with the distance to the nearest pole, so steps shrink through the
        // coordinate singularity and the fast phi/p_theta swing is resolved;
        // the dphi term is weighted by sin(theta) since coordinate phi-motion
        // at the axis is degenerate and must not starve the step size.
        float sin_th_rate = sqrt(max(1.0f - prev_cos * prev_cos, 0.0f));
        float pole_dist = max(min(s.th, 3.14159265f - s.th), 1e-3f);
        // The p_theta term resolves the STIFF polar turning point: |dtheta|
        // vanishes exactly at the turn while the L^2/sin^3(theta) wall kicks
        // p_theta impulsively — without this term the controller steps through
        // the impulse and the theta-phase (hence every later equatorial
        // crossing radius) is corrupted for near-axis rays.
        float rate = fabs(d.r) / max(s.r, 0.5f) + fabs(d.th) / min(pole_dist, 1.0f)
                   + fabs(d.pth) / (fabs(s.pth) + 1.0f)
                   + fabs(d.ph) * sin_th_rate + 1e-9f;
        float dlam = 0.05f / rate;
        dlam = min(dlam, 0.35f * max(s.r - r_horizon, 1e-3f) / max(fabs(d.r), 1e-6f));
        // Volumetric mode integrates an extended emitting body rather than a
        // surface: cap the step so the density profile is sampled, not
        // stepped over. (The geometry-driven controller alone would take
        // dlam ~ 0.05 r, far too coarse across the torus.)
        if (sys.disk_model == DISK_VOLUMETRIC && s.r < vol_r_max)
            dlam = min(dlam, 0.35f);
        dlam = max(dlam, 1e-4f);

        s = kerr_rk4(s, dlam, kerr);

        // Axis reflection: a ray that runs through the pole continues on the
        // other side (the BL chart's double cover; the phi jump is handled by
        // the polar-cap bypass below).
        if (s.th < 0.0f)             { s.th = -s.th;             s.pth = -s.pth; }
        else if (s.th > 3.14159265f) { s.th = 6.2831853f - s.th; s.pth = -s.pth; }

        bool in_cap_now = (s.th < POLE_CAP) || (s.th > 3.14159265f - POLE_CAP);
        if (in_cap_now && !in_pole_cap) { in_pole_cap = true; ph_cap_entry = prev_ph; }
        else if (!in_cap_now && in_pole_cap) {
            in_pole_cap = false;
            s.ph = ph_cap_entry + 3.14159265f;
        }
        if (in_pole_cap) s.ph = ph_cap_entry;

        float cos_th = cos(s.th);

        // ----- Disk-surface intersection (optically thick). -----
        // The equatorial crossing is the disk's only geometric event; linear
        // interpolation to the sign change gives sub-step (r, phi) accuracy.
        // The interpolation factor is valid for BOTH crossing directions
        // (descending prev_cos > 0 > cos_th and ascending prev_cos < 0 < cos_th):
        // the denominator |prev_cos| + |cos_th| never vanishes inside this branch.
        bool just_crossed = (prev_cos * cos_th < 0.0f);
        if (just_crossed) crossings++;

        // ----- Optically-thin volumetric transport (GRMHD-style). -----
        if (sys.disk_model == DISK_VOLUMETRIC && s.r < vol_r_max && s.r > r_horizon * 1.05f) {
            float sin_th_v = sqrt(max(1.0f - cos_th * cos_th, 0.0f));
            TorusSample ts;
            if (torus_sample(s.r, cos_th, sin_th_v, kerr, sys, ts)) {
                // Emissivity scales with n^2 (two-body process, as for
                // free-free emission) — this is what concentrates the light
                // at the density peak instead of smearing it along the long
                // optically-thin sight lines through the envelope.
                float ne2 = ts.n * ts.n;
                float ga = sys.torus_alpha;
                float J_inv = ne2 * pow(ts.g, ga + 2.0f);
                float A_inv = sys.torus_absorb * ne2 * pow(ts.g, ga + 1.5f);

                float T_obs = (sys.beaming_mode == BEAM_NONE) ? ts.T : ts.g * ts.T;
                float3 chroma = blackbody_color(T_obs);
                if (sys.beaming_mode == BEAM_NO_BOOST) {
                    // Colour shift only: divide out the transport's own g
                    // dependence so intensity is Doppler-flat.
                    J_inv = ts.n;
                }
                float dI = J_inv * dlam * sys.torus_density * 1.2f;
                col_accum += trans * chroma * dI;

                // Emission-weighted redshift for the diagnostic lens.
                g_weight += dI * trans;
                g_accum  += ts.g * dI * trans;
                if (hit_order < 0) { hit_order = crossings; fate = 3; }

                if (A_inv > 0.0f) trans *= exp(-A_inv * dlam * sys.torus_density);
            }
        }

        // ----- Thin-disk surface intersection (optically thick). -----
        if (just_crossed && sys.disk_model == DISK_THIN) {
            float f     = clamp(prev_cos / (prev_cos - cos_th), 0.0f, 1.0f);
            float r_at  = mix(prev_r,  s.r,  f);
            float ph_at = mix(prev_ph, s.ph, f);
            if (r_at > r_in && r_at < r_out && sys.disk_density > 0.0f) {
                float g_here;
                float3 emis = disk_surface_emission(r_at, kerr, sys, r_in, r_out, g_here);
                // Higher-order images (n >= 1) are physically demagnified; the
                // boost slider amplifies them for visibility (0 = physical).
                int order = crossings - 1;
                if (order >= 1 && sys.photon_ring_boost > 0.0f)
                    emis *= (1.0f + sys.photon_ring_boost);
                col_accum += trans * emis;
                trans = 0.0f;
                hit_order = order;
                hit_g = g_here;
                fate = 3;
                (void)ph_at;
                break;   // optically thick — nothing shines through the disk
            }
        }

        // ----- Near-BH glow + ergosphere shimmer (decorative, faint). -----
        // Volumetric terms are weighted by the affine step so brightness is a
        // function of PATH LENGTH, not local step density — the adaptive
        // controller takes many tiny steps exactly where photons whirl, which
        // would otherwise overbrighten those rays and change the image
        // whenever the accuracy constant changes. The 0.06 max(r,1) divisor
        // reproduces the historical per-step calibration (old fixed policy
        // dlam = 0.06 r) so no visual retune is needed.
        float vol_w = dlam / (0.06f * max(s.r, 1.0f));
        if (s.r < 5.0f) {
            float glow = 0.0006f / (s.r * s.r);
            col_accum += trans * float3(0.12f, 0.07f, 0.04f) * glow * 30.0f * vol_w;
            trans *= (1.0f - min(glow * vol_w, 0.5f));

            float a_abs = abs(sys.spin);
            if (a_abs > 0.01f) {
                float spin2 = sys.spin * sys.spin;
                float r_E   = 0.5f + 0.5f * sqrt(max(1.0f - spin2 * cos_th * cos_th, 0.0f));
                if (s.r < r_E && s.r > r_horizon) {
                    float ergo_depth = (r_E - s.r) / max(r_E - r_horizon, 1e-6f);
                    float ergo_glow  = ergo_depth * ergo_depth * 0.015f * a_abs;
                    col_accum += trans * float3(0.2f, 0.05f, 0.35f) * ergo_glow * 5.0f * vol_w;
                }
            }
        }

        // ----- Polar relativistic jets (decorative). -----
        if (sys.jet_int > 0.0f && s.r > 1.2f && s.r < 25.0f) {
            float cos_axis = abs(cos_th);
            if (cos_axis > 0.92f) {
                float jet_core    = smoothstep(0.92f, 0.98f, cos_axis);
                float jet_falloff = exp(-s.r * 0.15f);
                float sin_th_j = sqrt(max(1.0f - cos_th * cos_th, 0.0f));
                float3 pj = bl_to_cart(s.r, sin_th_j, cos_th, sin(s.ph), cos(s.ph));
                float jet_turb = 0.7f + 0.3f * (float)noise_half(pj * 3.0f + sys.time * 0.5f);
                float jet_density = jet_core * jet_falloff * jet_turb * 0.08f * sys.jet_int;
                half3 jet_col  = mix(half3(0.4h, 0.6h, 1.0h), half3(0.8h, 0.9h, 1.0h), half(jet_core));
                col_accum += trans * float3(jet_col) * 20.0f * jet_density * vol_w;
                trans *= (1.0f - min(jet_density * vol_w, 0.5f));
            }
        }

        if (trans < 0.005f) { fate = 4; break; }   // opaque foreground (jet/glow)

        prev_cos = cos_th;
        prev_r   = s.r;
        prev_ph  = s.ph;
    }

    // ----- Background + stars at escape. -----
    // Budget-exhausted rays (fate 0) are rare pole-skimming or near-critical
    // orbits; their state is unreliable, so they fall back to the undeflected
    // camera direction instead of reconstructing a bogus exit direction.
    float3 exit_dir = rd;
    if (fate == 2) {
        float cos_th = cos(s.th);
        float sin_th = sqrt(max(1.0f - cos_th * cos_th, 1e-9f));
        float sin_ph_e = sin(s.ph), cos_ph_e = cos(s.ph);
        float3 r_hat_e, th_hat_e, ph_hat_e;
        bl_basis(sin_th, cos_th, sin_ph_e, cos_ph_e, r_hat_e, th_hat_e, ph_hat_e);

        // Local static-frame momentum components at the exit point:
        //   p^(r)  = sqrt(g_rr) dr/dl ~ Delta p_r / sqrt(Sigma Delta)
        //   p^(th) = p_th / sqrt(Sigma)
        //   p^(ph) = L / sqrt(g_phph) ~ L / (r sin th)
        float Sigma_e = kerr_Sigma(s.r, cos_th, kerr.a2);
        float Delta_e = max(kerr_Delta(s.r, kerr.a2, kerr.Q2), 1e-6f);
        float n_r_out  = Delta_e * s.pr / sqrt(max(Sigma_e * Delta_e, 1e-9f));
        float n_th_out = s.pth / sqrt(Sigma_e);
        float n_ph_out = kerr.L / max(s.r * sin_th, 1e-3f);
        float norm     = rsqrt(n_r_out * n_r_out + n_th_out * n_th_out + n_ph_out * n_ph_out + 1e-12f);
        exit_dir = (n_r_out * r_hat_e + n_th_out * th_hat_e + n_ph_out * ph_hat_e) * norm;
    }
    if (fate == 2 || (fate == 0 && trans > 0.005f)) {
        float cos_th = cos(s.th);
        float sin_th = sqrt(max(1.0f - cos_th * cos_th, 1e-9f));
        float sin_ph_e = sin(s.ph), cos_ph_e = cos(s.ph);
        float3 bg;
        if (sys.lens_mode == LENS_CHECKER) {
            bg = checker_sky(exit_dir, cam.camForward.xyz);
        } else {
            bg = (float3)sampleBackground_half(exit_dir, sys.time, sys.star_scint, sys.nebula_int);
        }

        // Companion stars: straight-ray sphere intersection from the escape
        // point in world space. Occlusion is inherent (only escaped rays get
        // here) and lensing comes from the geodesic bending before escape.
        // Residual deflection beyond the escape radius is O(2 rs / b) — negligible.
        if (sys.star_bodies != 0 && sys.lens_mode != LENS_CHECKER) {
            float3 p_esc = bl_to_cart(s.r, sin_th, cos_th, sin_ph_e, cos_ph_e) * rs + bhPos;
            float best_t = 1e30f;
            int   best_j = -1;
            for (int j = 0; j < u_obj.count; j++) {
                if (j == u_obj.bh_index) continue;
                float3 oc = objs[j].posRadius.xyz - p_esc;
                float  t  = dot(oc, exit_dir);
                if (t <= 0.0f) continue;
                float d2 = dot(oc, oc) - t * t;
                float R2 = objs[j].posRadius.w * objs[j].posRadius.w;
                if (d2 < R2 && t < best_t) { best_t = t; best_j = j; }
            }
            if (best_j >= 0) {
                float3 oc = objs[best_j].posRadius.xyz - p_esc;
                float d2 = dot(oc, oc) - best_t * best_t;
                float R2 = objs[best_j].posRadius.w * objs[best_j].posRadius.w;
                float limb = sqrt(max(1.0f - d2 / R2, 0.0f));        // limb darkening
                bg = objs[best_j].color.xyz * (0.6f + 1.4f * limb) * 2.0f;
            }
        }
        col_accum += trans * bg;
    }

    if (sys.disk_model == DISK_VOLUMETRIC && g_weight > 0.0f) hit_g = g_accum / g_weight;

    // ----- Lens output. -----
    float3 col = col_accum;
    if (sys.lens_mode == LENS_RING_ORDER) {
        // Categorical false color by ray fate / image order (display-linear;
        // post applies gamma only). Precedent: Gralla-Holz-Wald image
        // decomposition, EHT photon-ring papers, Bohn et al. fate maps.
        if      (fate == 1)      col = float3(0.06f, 0.06f, 0.07f);          // captured
        else if (fate == 4)      col = float3(0.35f, 0.42f, 0.55f);          // absorbed (jet/glow)
        else if (fate == 3) {
            float shade = 0.45f + 0.55f * clamp(length(col_accum) * 0.2f, 0.0f, 1.0f);
            if      (hit_order == 0) col = float3(0.90f, 0.62f, 0.08f) * shade; // direct
            else if (hit_order == 1) col = float3(0.10f, 0.75f, 0.85f) * shade; // 1st lensed
            else if (hit_order == 2) col = float3(0.80f, 0.20f, 0.85f) * shade; // 2nd
            else                     col = float3(0.95f, 0.10f, 0.15f) * shade; // >= 3rd
        }
        else                     col = float3(0.02f, 0.03f, 0.10f);          // sky
    } else if (sys.lens_mode == LENS_REDSHIFT) {
        if (fate == 3) {
            // Diverging map centered at g = 1, clamped to [0.4, 1.6].
            float t = (clamp(hit_g, 0.4f, 1.6f) - 0.4f) / 1.2f;
            col = diverging_map(t);
        } else if (fate == 1) col = float3(0.03f);
        else if (fate == 4)   col = float3(0.10f, 0.12f, 0.16f);   // absorbed foreground
        else                  col = float3(0.01f, 0.01f, 0.02f);
    }

    out.write(float4(col, 1.0f), pix);
}

// --- TEMPORAL ACCUMULATION ---
//
// One kernel serves three purposes, selected by sys.accum_alpha (host-set):
//   camera/params changed  -> alpha = 1        (replace: no ghosting)
//   camera static          -> alpha = 1/(N+1)  (progressive supersampling of
//                                               the jittered rays: converges
//                                               to ground truth in ~64 frames)
//   moving + motion blur   -> alpha = 1 - blur (exponential trail, in linear
//                                               HDR so exposure/bloom apply
//                                               uniformly afterwards)
kernel void temporal_accum(texture2d<float, access::read> curTex [[texture(0)]],
                           texture2d<float, access::read> prevTex [[texture(1)]],
                           texture2d<float, access::write> outTex [[texture(2)]],
                           constant SystemUniforms& sys [[buffer(0)]],
                           uint2 pix [[thread_position_in_grid]]) {
    uint w = outTex.get_width(); uint h = outTex.get_height();
    if (pix.x >= w || pix.y >= h) return;
    float3 cur  = curTex.read(pix).rgb;
    float3 prev = prevTex.read(pix).rgb;
    // Freshly created private textures have undefined contents; a NaN there
    // would persist through mix() forever. Sanitize before blending.
    if (any(isnan(prev)) || any(isinf(prev))) prev = cur;
    float3 acc = mix(prev, cur, clamp(sys.accum_alpha, 0.0f, 1.0f));
    outTex.write(float4(acc, 1.0f), pix);
}

// --- POST-PROCESSING ---

static inline float luma709(float3 c) { return dot(c, float3(0.2126f, 0.7152f, 0.0722f)); }

// Colors authored as display (sRGB) values -> linear, for the extended-linear
// drawable. The compositor applies the transfer function, so everything the
// shader writes must be scene-linear.
static inline float3 srgb_to_linear(float3 c) {
    return select(c / 12.92f, pow((c + 0.055f) / 1.055f, 2.4f), c > 0.04045f);
}

static inline float3 aces_fit(float3 x) {
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    return (x * (a * x + b)) / (x * (c * x + d) + e);
}

// EDR tonemap = the SDR look, plus real highlights.
//
// Naively retargeting the curve (H * ACES(x/H)) moves the shoulder out to H
// but drags the midtones down with it — middle gray falls from 0.267 to 0.071
// at H = 16, darkening the whole image. Instead: keep the ACES result
// bit-for-bit below the knee (where essentially the entire image lives), and
// re-expand only the band the SDR curve was crushing into the last few
// percent below reference white. The expansion is driven by the LINEAR
// signal so highlight detail survives instead of flat-clipping, and it grows
// logarithmically, so even a 500x input lands at ~4x white rather than H.
static inline float3 tonemap_edr(float3 x, float H) {
    float3 sdr = saturate(aces_fit(x));
    if (H <= 1.001f) return sdr;                  // SDR display: unchanged
    const float x_knee = 2.0f;                    // linear value where ACES saturates
    float3 over = max(x - x_knee, 0.0f);
    float3 ext  = log2(1.0f + over) * 0.35f;      // gentle above-white lift
    return min(sdr + ext, H);
}

kernel void post_process_suite(texture2d<float, access::read> inTex [[texture(0)]],
                               texture2d<float, access::write> outTex [[texture(1)]],
                               texture2d<float, access::sample> bloomTex [[texture(2)]],
                               constant SystemUniforms& sys [[buffer(0)]],
                               uint2 pix [[thread_position_in_grid]]) {
    uint w = outTex.get_width(); uint h = outTex.get_height();
    if (pix.x >= w || pix.y >= h) return;

    float3 col = inTex.read(pix).rgb;
    float2 uv = (float2(pix) + 0.5f) / float2(w, h);

    // False-color lenses are display-referred: bypass the photographic chain.
    if (sys.lens_mode == LENS_RING_ORDER || sys.lens_mode == LENS_REDSHIFT) {
        outTex.write(float4(srgb_to_linear(max(col, 0.0f)), 1.0f), pix);
        return;
    }

    // EHT observation lens: telescope-beam-blurred luminance through a radio
    // colormap (the bloom chain carries the beam-convolved image when active).
    if (sys.lens_mode == LENS_EHT) {
        float lum = luma709(bloomTex.sample(sampler(filter::linear), uv).rgb) * sys.exposure;
        float t = lum / (1.0f + lum);          // soft normalize into [0,1)
        outTex.write(float4(srgb_to_linear(hot_map(t * 1.6f)), 1.0f), pix);
        return;
    }

    // Exposure first: everything downstream operates on display-referred light.
    col *= sys.exposure;

    if (sys.enable_bloom != 0) {
        float3 bloom = bloomTex.sample(sampler(filter::linear), uv).rgb;
        col += bloom * 0.6f;                   // extract already applied exposure
    }

    if (sys.flare_int > 0.0f) {
        float flare = 0.0f;
        for (int i=-10; i<=10; i++) {
            uint2 p = uint2(clamp(int(pix.x) + i*4, 0, int(w)-1), pix.y);
            flare += max(0.0f, luma709(inTex.read(p).rgb) * sys.exposure - sys.bloom_threshold);
        }
        col += float3(0.1f, 0.3f, 1.0f) * (flare / 21.0f) * sys.flare_int;
    }

    if (sys.vignette_int > 0.0f) {
        float vignette = 1.0f - 0.4f * sys.vignette_int * dot(uv - 0.5f, uv - 0.5f) * 4.0f;
        col *= max(vignette, 0.0f);
    }

    // ACES filmic tonemap, extended to the display's EDR headroom. On an XDR
    // panel the Doppler-boosted inner limb genuinely emits above reference
    // white instead of being crushed to it.
    col = tonemap_edr(col, sys.edr_headroom);

    // Film grain rides on developed density: after the tonemap, scaled by a
    // photographic sqrt-luminance response (no black-floor lift on empty sky).
    if (sys.film_grain > 0.0f) {
        float g = (hash13(float3(uv * 100.0f, sys.time)) - 0.5f) * sys.film_grain;
        col = max(col + g * sqrt(max(luma709(col), 0.0f)), 0.0f);
    }

    // Scene-linear out: the drawable is extended-linear and the compositor
    // applies the transfer function.
    outTex.write(float4(col, 1.0f), pix);
}

// --- BLOOM / BEAM EXTRACTION ---
//
// Standard mode: threshold on the EXPOSED image (so the slider's meaning does
// not drift with auto-exposure) into a half-res target for the MPS blur.
// EHT lens: plain downsample (no threshold) — the MPS blur then acts as the
// telescope's restoring beam.
kernel void bloom_extract(texture2d<float, access::read> inTex [[texture(0)]],
                          texture2d<float, access::write> outTex [[texture(1)]],
                          constant SystemUniforms& sys [[buffer(0)]],
                          uint2 pix [[thread_position_in_grid]]) {
    uint w = outTex.get_width(); uint h = outTex.get_height();
    if (pix.x >= w || pix.y >= h) return;

    bool eht = (sys.lens_mode == LENS_EHT);
    if (!eht && sys.enable_bloom == 0) { outTex.write(float4(0.0f), pix); return; }

    uint2 src = pix * 2;
    uint sw = inTex.get_width(); uint sh = inTex.get_height();
    float3 c0 = inTex.read(min(src, uint2(sw-1, sh-1))).rgb;
    float3 c1 = inTex.read(min(src + uint2(1,0), uint2(sw-1, sh-1))).rgb;
    float3 c2 = inTex.read(min(src + uint2(0,1), uint2(sw-1, sh-1))).rgb;
    float3 c3 = inTex.read(min(src + uint2(1,1), uint2(sw-1, sh-1))).rgb;
    float3 avg = (c0 + c1 + c2 + c3) * 0.25f;

    if (eht) { outTex.write(float4(avg, 1.0f), pix); return; }

    avg *= sys.exposure;
    // Luminance-relative threshold preserves hue (no per-channel clipping).
    float lum = luma709(avg);
    float over = max(lum - sys.bloom_threshold, 0.0f);
    float3 bloom = (lum > 1e-6f) ? avg * (over / lum) : float3(0.0f);
    outTex.write(float4(bloom, 1.0f), pix);
}

// --- AUTO-EXPOSURE LUMINANCE ANALYSIS ---
//
// Mean of log2 luminance, one sample per 4x4 block, simd-summed so each
// simdgroup issues at most one atomic. Pixels below the floor (empty sky)
// are EXCLUDED from the average — flooring them at 0.001 used to drag the
// mean so low that exposure pegged at its clamp and clipped the disk white.
kernel void luminance_reduce(texture2d<float, access::read> inTex [[texture(0)]],
                             device atomic_uint* lumBuffer [[buffer(0)]],
                             uint2 pix [[thread_position_in_grid]],
                             uint simd_lane [[thread_index_in_simdgroup]]) {
    uint w = inTex.get_width(); uint h = inTex.get_height();
    uint2 src = pix * 4u;

    uint encoded = 0u;
    uint count = 0u;
    if (src.x < w && src.y < h) {
        float lum = luma709(inTex.read(src).rgb);
        if (lum >= 0.001f) {
            float log_lum = log2(lum);
            encoded = uint(clamp((log_lum + 12.0f) * 100.0f, 0.0f, 2400.0f));
            count = 1u;
        }
    }

    uint sum_simd = simd_sum(encoded);
    uint cnt_simd = simd_sum(count);
    if (simd_lane == 0u && cnt_simd > 0u) {
        atomic_fetch_add_explicit(lumBuffer, sum_simd, memory_order_relaxed);
        atomic_fetch_add_explicit(lumBuffer + 1, cnt_simd, memory_order_relaxed);
    }
}

// --- N-BODY PHYSICS ---
//
// Velocity-Verlet (kick-drift-kick) leapfrog with a FIXED physics timestep
// (sys.dt_phys, host-substepped): integration accuracy is independent of
// frame rate, and a stalled frame can no longer kick a star onto an escape
// orbit. Two dispatches per substep because all positions must commit before
// any body evaluates the new acceleration.

static inline float3 nbody_acceleration(uint id, const device SimObject* objects, int count) {
    constexpr float G   = 6.67430e-11f;
    constexpr float eps = 1e9f;
    float3 p = objects[id].posRadius.xyz;
    float3 a = float3(0.0f);
    for (int i = 0; i < count; i++) {
        if (i == (int)id) continue;
        float3 r_v = objects[i].posRadius.xyz - p;
        float r2 = dot(r_v, r_v);
        a += normalize(r_v) * (G * objects[i].mass / (r2 + eps * eps));
    }
    return a;
}

kernel void physics_init_accel(device SimObject* objects [[buffer(0)]],
                                constant ObjectsUniform& u [[buffer(1)]],
                                uint id [[thread_position_in_grid]]) {
    if (id >= (uint)u.count) return;
    if (objects[id].mass > 1e35f) {
        objects[id].acceleration.xyz = float3(0.0f);   // BH stays fixed
        return;
    }
    objects[id].acceleration.xyz = nbody_acceleration(id, objects, u.count);
}

kernel void physics_drift(device SimObject* objects [[buffer(0)]],
                          constant ObjectsUniform& u [[buffer(1)]],
                          constant SystemUniforms& sys [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= (uint)u.count || objects[id].mass > 1e35f) return;
    float dt = sys.dt_phys;
    float3 p = objects[id].posRadius.xyz;
    float3 v = objects[id].velocity.xyz;
    float3 a = objects[id].acceleration.xyz;
    objects[id].posRadius.xyz = p + v * dt + 0.5f * a * (dt * dt);
}

kernel void physics_force_kick(device SimObject* objects [[buffer(0)]],
                                constant ObjectsUniform& u [[buffer(1)]],
                                constant SystemUniforms& sys [[buffer(2)]],
                                uint id [[thread_position_in_grid]]) {
    if (id >= (uint)u.count || objects[id].mass > 1e35f) return;
    float dt = sys.dt_phys;
    float3 a_old = objects[id].acceleration.xyz;
    float3 a_new = nbody_acceleration(id, objects, u.count);
    objects[id].velocity.xyz += 0.5f * (a_old + a_new) * dt;
    objects[id].acceleration.xyz = a_new;
}

// --- SPACETIME GRID (embedding diagram) ---

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
            total_depression += BH_WELL_DEPTH * BH_SOFTENING / sqrt(d * d + BH_SOFTENING * BH_SOFTENING);
        } else if (o[i].mass > 1e28f) {
            float star_soft = max(o[i].posRadius.w * 3.0f, 5.0e12f);  // floor prevents tubes
            float mass_ratio = o[i].mass / 12.0e30f;
            total_depression += STAR_WELL_DEPTH * mass_ratio * star_soft / sqrt(d * d + star_soft * star_soft);
        }

        // Stylised quadrupole-pattern ripple: decorative sin(kr − ωt)·cos(2φ),
        // NOT a strain solution — amplitude/wavelength chosen for legibility.
        if (sys.gw_amp > 0.0f && o[i].mass > 1e28f) {
            float mass_scale = is_bh ? 1.0f : 0.08f;
            float phase = d * 1e-11f - sys.time * 2.0f;
            float ripple = sin(phase) * (sys.gw_amp * 2e11f * mass_scale) / (1.0f + d * 1e-12f);
            ripple *= cos(2.0f * atan2(delta.z, delta.x));
            p.y += ripple;
        }
    }

    p.y -= total_depression;
    float well_depth = clamp(total_depression / BH_WELL_DEPTH, 0.0f, 1.0f);
    p.y -= GRID_BASELINE;
    out.position = u.viewProj * float4(p, 1.0f);
    out.depth = well_depth;
    return out;
}

fragment float4 grid_fragment(VertexOut in [[stage_in]],
                              texture2d<float, access::sample> sceneTex [[texture(0)]],
                              constant SystemUniforms& sys [[buffer(0)]]) {
    float d = in.depth;  // 0 = flat surface, 1 = deep in gravity well

    // CRITICAL: flat grid lines (d ≈ 0) MUST be fully discarded — alpha
    // stacking of hundreds of near-flat lines otherwise forms an opaque band.
    if (d < 0.008f) discard_fragment();

    // Scene occlusion: hide the grid behind anything bright. The scene texture
    // is pre-exposure HDR, so scale by the live exposure to keep the visible
    // threshold stable across auto-exposure states.
    float2 screen_uv = in.position.xy / float2(sceneTex.get_width(), sceneTex.get_height());
    constexpr sampler s(filter::linear);
    float scene_lum = dot(sceneTex.sample(s, screen_uv).rgb, float3(0.2126f, 0.7152f, 0.0722f));
    if (scene_lum * sys.exposure > 0.05f) discard_fragment();

    float3 col = mix(float3(0.15f, 0.8f, 1.0f),
                     float3(0.5f, 0.2f, 1.0f),
                     smoothstep(0.0f, 0.3f, d));
    col = mix(col,
              float3(1.0f, 0.3f, 0.05f),
              smoothstep(0.3f, 0.75f, d));
    col = srgb_to_linear(col) * (1.5f + d * 1.5f);   // linear target
    float alpha = smoothstep(0.008f, 0.06f, d) * mix(0.5f, 0.9f, d);
    return float4(col, alpha);
}
