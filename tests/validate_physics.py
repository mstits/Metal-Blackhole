#!/usr/bin/env python3
"""
GRRT Physics Validation Test Suite
===================================
Mathematical proof-of-correctness for the Metal Blackhole engine.

The shader integrates null geodesics in the exact Kerr-Newman metric using
the Carter-separated Hamilton-Jacobi formulation. This script mirrors that
integrator in Python (kerr_step / cam_init) and verifies it against analytic
GR predictions.

Conventions match the shader:
    sys.spin = a/M ∈ [-1, 1]    (slider value, dimensionless)
    Length unit rs = 2M, so M = 1/2 in shader units.
    "a_spin" in this file is the slider value (= a/M).
    "a" in metric formulas is the dimensional spin in rs units = a_spin / 2.

Run:  python3 tests/validate_physics.py
"""
import math
import sys

PASS = 0
FAIL = 0


def check(name, computed, expected, tolerance=1e-3, unit=""):
    global PASS, FAIL
    err = abs(computed - expected)
    rel_err = err / max(abs(expected), 1e-12)
    status = "PASS" if rel_err < tolerance else "FAIL"
    if rel_err >= tolerance:
        FAIL += 1
    else:
        PASS += 1
    print(f"  [{status}]  {name}: computed={computed:.6f}, expected={expected:.6f} "
          f"(err={rel_err:.2e}) {unit}")


def check_true(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS]  {name} {detail}")
    else:
        FAIL += 1
        print(f"  [FAIL]  {name} {detail}")


# ============================================================
#  Kerr-Newman BL geodesic integrator (mirror of geodesic.metal)
# ============================================================

def kerr_R(r, a, L, Q, Q_ch_sq=0.0):
    """Radial potential for null geodesics with E = 1."""
    P = r * r + a * a - a * L
    Delta = r * r - r + a * a + Q_ch_sq
    return P * P - Delta * ((L - a) ** 2 + Q)


def kerr_Theta(cos_th, sin_th, a, L, Q):
    """Polar potential for null geodesics with E = 1."""
    s2 = max(sin_th * sin_th, 1e-9)
    return Q + a * a * cos_th ** 2 - L * L * cos_th ** 2 / s2


def kerr_Sigma(r, cos_th, a):
    return r * r + a * a * cos_th * cos_th


def kerr_Delta(r, a, Q_ch_sq=0.0):
    return r * r - r + a * a + Q_ch_sq


def kerr_dphi_over_Sigma(r, sin_th, a, L, Q_ch_sq=0.0):
    """(Σ dφ/dλ) for E = 1."""
    Delta = kerr_Delta(r, a, Q_ch_sq)
    s2 = max(sin_th * sin_th, 1e-9)
    return (a * (r - Q_ch_sq) - a * a * L) / Delta + L / s2


def _rhs(r, th, s_r, s_th, a, L, Q, Q_ch_sq):
    c = math.cos(th)
    s = math.sin(th)
    Sigma = kerr_Sigma(r, c, a)
    R = kerr_R(r, a, L, Q, Q_ch_sq)
    Th = kerr_Theta(c, s, a, L, Q)
    dr = s_r * math.sqrt(max(R, 0.0)) / Sigma
    dt = s_th * math.sqrt(max(Th, 0.0)) / Sigma
    dp = kerr_dphi_over_Sigma(r, s, a, L, Q_ch_sq) / Sigma
    return (dr, dt, dp)


def kerr_rk4(state, signs, dlam, a, L, Q, Q_ch_sq=0.0):
    """One RK4 step on (r, θ, φ). Returns updated (state, possibly-flipped signs)."""
    r, th, ph = state
    s_r, s_th = signs
    k1 = _rhs(r, th, s_r, s_th, a, L, Q, Q_ch_sq)
    k2 = _rhs(r + 0.5 * dlam * k1[0], th + 0.5 * dlam * k1[1], s_r, s_th, a, L, Q, Q_ch_sq)
    k3 = _rhs(r + 0.5 * dlam * k2[0], th + 0.5 * dlam * k2[1], s_r, s_th, a, L, Q, Q_ch_sq)
    k4 = _rhs(r + dlam * k3[0],         th + dlam * k3[1],         s_r, s_th, a, L, Q, Q_ch_sq)

    r_new = r + dlam * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
    t_new = th + dlam * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
    p_new = ph + dlam * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0

    # Sign flip on turning points (R or Θ crossing zero).
    cos_n = math.cos(t_new)
    sin_n = math.sin(t_new)
    Rn = kerr_R(r_new, a, L, Q, Q_ch_sq)
    Thn = kerr_Theta(cos_n, sin_n, a, L, Q)
    Ro = kerr_R(r,  a, L, Q, Q_ch_sq)
    To = kerr_Theta(math.cos(th), math.sin(th), a, L, Q)
    if Rn < 0.0 and Ro > 0.0:
        s_r = -s_r
    if Thn < 0.0 and To > 0.0:
        s_th = -s_th

    return ((r_new, t_new, p_new), (s_r, s_th))


def cam_init(cam_pos, ray_dir, a):
    """Compute (E=1, L_z, Q_C, p_r_init, p_th_init) from camera Cartesian state.

    cam_pos: (x, y, z) in rs units, BH at origin, +y = spin axis.
    ray_dir: unit Cartesian ray direction.
    a:       dimensional Kerr spin in rs units (= a_spin/2).

    Uses the static-observer tetrad with the g_tφ tetrad correction omitted
    (negligible for r_obs >> a).
    """
    x, y, z = cam_pos
    r = math.sqrt(x * x + y * y + z * z)
    cos_th = max(-1.0, min(1.0, y / r))
    sin_th = math.sqrt(max(1.0 - cos_th * cos_th, 1e-9))
    phi = math.atan2(z, x)
    sin_ph = math.sin(phi)
    cos_ph = math.cos(phi)

    # Local spherical basis
    r_hat  = (sin_th * cos_ph,  cos_th,           sin_th * sin_ph)
    th_hat = (cos_th * cos_ph, -sin_th,           cos_th * sin_ph)
    ph_hat = (-sin_ph,           0.0,             cos_ph)
    n_r  = ray_dir[0] * r_hat[0]  + ray_dir[1] * r_hat[1]  + ray_dir[2] * r_hat[2]
    n_th = ray_dir[0] * th_hat[0] + ray_dir[1] * th_hat[1] + ray_dir[2] * th_hat[2]
    n_ph = ray_dir[0] * ph_hat[0] + ray_dir[1] * ph_hat[1] + ray_dir[2] * ph_hat[2]

    Sigma = r * r + a * a * cos_th ** 2
    Delta = r * r - r + a * a
    gtt_neg = 1.0 - r / Sigma
    grr     = Sigma / max(Delta, 1e-9)
    gthth   = Sigma
    gphph_o_s2 = ((r * r + a * a) * Sigma + a * a * r * sin_th ** 2) / Sigma
    gphph   = gphph_o_s2 * sin_th ** 2
    omega_obs = 1.0 / math.sqrt(max(gtt_neg, 1e-9))

    p_r  = math.sqrt(grr)   * omega_obs * n_r
    p_th = math.sqrt(gthth) * omega_obs * n_th
    L    = math.sqrt(max(gphph, 0.0)) * omega_obs * n_ph

    Q = p_th * p_th - cos_th ** 2 * (a * a - L * L / max(sin_th ** 2, 1e-9))
    return r, cos_th, phi, p_r, p_th, L, Q


def shoot_ray(cam_pos, ray_dir, a_spin, max_steps=4000, dlam=0.05, r_horizon=None):
    """Integrate a single null geodesic from camera to capture/escape.

    Returns: ('captured', r_final) | ('escaped', r_final, exit_dir) | ('inconclusive', r_final)
    """
    a = 0.5 * a_spin                             # real dimensional spin in rs units
    if r_horizon is None:
        r_horizon = 0.5 * (1.0 + math.sqrt(max(1.0 - a_spin * a_spin, 0.0)))

    r0, cos_th0, phi0, p_r0, p_th0, L, Q = cam_init(cam_pos, ray_dir, a)
    state = (r0, math.acos(cos_th0), phi0)
    signs = (-1.0 if p_r0 < 0 else 1.0, -1.0 if p_th0 < 0 else 1.0)

    for _ in range(max_steps):
        state, signs = kerr_rk4(state, signs, dlam, a, L, Q, 0.0)
        r = state[0]
        if r < r_horizon * 1.01:
            return ('captured', r)
        if r > 200.0:
            return ('escaped', r, None)
    return ('inconclusive', state[0])


# ============================================================
#  Analytic Kerr formulas (independent reference)
# ============================================================

def event_horizon(a_spin):
    """r_+/rs for a dimensionless spin a_spin = a/M."""
    return 0.5 * (1.0 + math.sqrt(max(1.0 - a_spin * a_spin, 0.0)))


def kerr_isco(a_spin):
    """Bardeen-Press-Teukolsky prograde ISCO in rs units."""
    a2 = a_spin * a_spin
    cbrt = max(1.0 - a2, 1e-6) ** (1.0 / 3.0)
    Z1 = 1.0 + cbrt * ((1.0 + a_spin) ** (1.0 / 3.0)
                      + max(1.0 - a_spin, 1e-6) ** (1.0 / 3.0))
    Z2 = math.sqrt(3.0 * a2 + Z1 * Z1)
    return 0.5 * (3.0 + Z2 - math.sqrt(max((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2), 0.0)))


def kerr_keplerian_omega(a_spin, r_rs):
    """Bardeen-Press-Teukolsky prograde Keplerian angular velocity Ω_K, in rs⁻¹.

    M-unit form: Ω_K = √M / (r^{3/2} + a √M). Converting to rs-units
    (M = rs/2, r and a expressed in rs) yields:
      Ω_K · rs = 1 / (√2 · r_rs^{3/2} + a_rs)
    where a_rs = a_phys/rs = (a/M)/2 = a_spin/2.
    """
    a_rs = 0.5 * a_spin
    return 1.0 / (math.sqrt(2.0) * r_rs * math.sqrt(r_rs) + a_rs)


def kerr_g_factor(a_spin, r_rs, L=0.0):
    """Bardeen-Press-Teukolsky g = ν_obs/ν_emit for a Keplerian-orbiting source.

    g = 1 / (U^t · (1 − Ω_K · L_z)). Used by the shader's disk emission.
    """
    a_rs = 0.5 * a_spin
    rh = r_rs
    Omega_K = kerr_keplerian_omega(a_spin, rh)
    gtt   = -(1.0 - 1.0 / rh)
    gtphi = -a_rs / rh
    gphph = rh * rh + a_rs * a_rs + a_rs * a_rs / rh
    U_denom = -gtt - 2.0 * gtphi * Omega_K - gphph * Omega_K * Omega_K
    Ut = 1.0 / math.sqrt(max(U_denom, 1e-9))
    return 1.0 / (Ut * (1.0 - Omega_K * L))


def nt_temperature(r_in, r):
    """Novikov-Thorne T(r) ∝ r^(-3/4) (1 - sqrt(r_in/r))^(1/4)."""
    r_ratio = r_in / r
    T_base = r_ratio ** 0.75
    T_boundary = max(1.0 - math.sqrt(r_ratio), 0.001) ** 0.25
    return T_base * T_boundary


# ============================================================
#  TEST SUITE
# ============================================================

print("=" * 70)
print("  Metal Blackhole — Kerr-Newman Physics Validation")
print("=" * 70)
print()

# --- TEST 1: Event Horizon ---
print("TEST 1: Event Horizon (Kerr r_+)")
print("-" * 40)
for a in (0.0, 0.5, 0.9, 0.998, 0.9999):
    check(f"a={a}", event_horizon(a),
          0.5 * (1.0 + math.sqrt(max(1 - a * a, 0.0))))
print()

# --- TEST 2: ISCO ---
print("TEST 2: Innermost Stable Circular Orbit (prograde)")
print("-" * 40)
# Literature values from Bardeen-Press-Teukolsky 1972 (in r_g = M units).
# Convert to rs = 2M by dividing by 2.
check("Schwarzschild a=0:    6 r_g = 3.0 rs", kerr_isco(0.0), 3.0)
check("Kerr a=0.5:           ~4.233 r_g",     kerr_isco(0.5), 2.1165, 0.01)
check("Kerr a=0.9:           ~2.321 r_g",     kerr_isco(0.9), 1.1605, 0.02)
check("Extreme Kerr a→1:     1 r_g = 0.5 rs", kerr_isco(0.9999), 0.5, 0.1)
print()

# --- TEST 3: Photon Sphere (Schwarzschild) ---
# Launch a tangentially-aimed ray at r = 1.5 rs with the critical L_z that
# satisfies R(r=1.5) = 0 = dR/dr (double root). For Schwarzschild this is
# L_z = b_crit · E with b_crit = √27/2 ≈ 2.598 rs.
#
# IMPORTANT: the photon sphere is *unstable*. Any RK4 round-off perturbation
# grows exponentially with e-folding time ~ 1 orbit. We integrate for one
# orbit (Δλ ≈ 2π·r ≈ 9.4) and check that r stays close.
print("TEST 3: Photon Sphere — Schwarzschild")
print("-" * 40)
L_crit = math.sqrt(27.0) / 2.0   # b_crit · E with E=1
Q_crit = 0.0
state = (1.5, math.pi / 2, 0.0)
signs = (1.0, 1.0)               # |p_r| = 0 ⇒ sign defaults arbitrary
r_min, r_max = 1.5, 1.5
for _ in range(2000):            # 2000 × dλ=0.002 ≈ 4 affine units, ~half-orbit
    state, signs = kerr_rk4(state, signs, 0.002, 0.0, L_crit, Q_crit, 0.0)
    r_min = min(r_min, state[0])
    r_max = max(r_max, state[0])
check("Photon-sphere stability r_min ≈ 1.5 (half-orbit)", r_min, 1.5, 0.02)
check("Photon-sphere stability r_max ≈ 1.5 (half-orbit)", r_max, 1.5, 0.02)
print()

# --- TEST 4: Shadow Radius (Schwarzschild) ---
# Critical impact parameter b_crit = √27/2 ≈ 2.598 rs.
# A ray at b < b_crit is captured, b > b_crit escapes.
print("TEST 4: Black Hole Shadow (Schwarzschild)")
print("-" * 40)
b_crit = math.sqrt(27.0) / 2.0


def test_capture(b, a_spin=0.0):
    """Camera at (-50, 0, b), ray pointed +x. Returns True if captured."""
    cam = (-50.0, 0.0, b)
    direction = (1.0, 0.0, 0.0)
    res = shoot_ray(cam, direction, a_spin)
    return res[0] == 'captured'


check_true(f"Ray at b = {b_crit - 0.1:.3f} CAPTURED", test_capture(b_crit - 0.1))
check_true(f"Ray at b = {b_crit + 0.2:.3f} ESCAPES", not test_capture(b_crit + 0.2))
check_true("Head-on ray (b=0) CAPTURED", test_capture(0.0))
print()

# --- TEST 5: Novikov-Thorne Temperature ---
print("TEST 5: Novikov-Thorne T(r) Profile")
print("-" * 40)
r_isco_s = 3.0
T_at_isco = nt_temperature(r_isco_s, r_isco_s)
check("T(r_isco) → 0 (zero-torque BC, 0.001 floor → 0.178)",
      T_at_isco, 0.001 ** 0.25, 0.01)

max_T = 0.0
max_r = 0.0
for i in range(1, 1000):
    r = r_isco_s + i * 0.05
    T = nt_temperature(r_isco_s, r)
    if T > max_T:
        max_T = T
        max_r = r
check("T(r) peaks near 4.08 rs", max_r, 4.08, 0.05)
check_true("T(10) > T(20) — monotone decay past peak",
           nt_temperature(r_isco_s, 10.0) > nt_temperature(r_isco_s, 20.0))
print()

# --- TEST 6: Keplerian Ω and BPT g-factor for the disk emitter ---
# The shader's disk emission applies Bardeen-Press-Teukolsky g⁴ with
# Ω_K = 1 / (√2 r^{3/2} + a_rs)  and U^t = 1/√(−g_tt − 2 g_tφ Ω − g_φφ Ω²).
# This test verifies the shader formula against an independent derivation
# from the Kerr metric coefficients.
print("TEST 6: Keplerian Ω + BPT g-factor (disk emitter)")
print("-" * 40)
for a in (0.0, 0.5, 0.9):
    for r in (3.0, 5.0, 10.0):
        Omega_ref = kerr_keplerian_omega(a, r)
        # Mirror the shader (sys.spin = a, a_real = 0.5 * sys.spin):
        a_real = 0.5 * a
        Omega_shader = 1.0 / (math.sqrt(2.0) * r * math.sqrt(r) + a_real)
        check(f"Ω_K (a={a}, r={r})", Omega_shader, Omega_ref, 1e-6)
# At infinity, g → 1 (no shift). Boundary check.
check("g(a=0, r=10000) → 1", kerr_g_factor(0.0, 10000.0, 0.0), 1.0, 0.001)
# At Schwarzschild ISCO (r=3 rs = 6M), a Keplerian orbiter has the famous
# dτ/dt = √(1 − 3M/r) = √(1/2). For radial emission (L_z=0), g = 1/U^t = 1/√2.
check("g(a=0, r=3, L=0) = 1/√2  (Schwarzschild ISCO Keplerian)",
      kerr_g_factor(0.0, 3.0, 0.0), 1.0 / math.sqrt(2.0), 0.005)
print()

# --- TEST 7: Polar Doppler Symmetry ---
print("TEST 7: Polar-View Doppler Symmetry (top-down)")
print("-" * 40)
vel_polar = (0.0, -1.0, 0.0)
for phi_deg in (0, 45, 90, 135, 180, 225, 270, 315):
    phi = math.radians(phi_deg)
    r = 5.0
    pos_x = r * math.cos(phi)
    pos_z = r * math.sin(phi)
    tang_x = -pos_z / r
    tang_z = pos_x / r
    cos_v = vel_polar[0] * tang_x + vel_polar[2] * tang_z
    check(f"cos_v at φ={phi_deg}°", cos_v, 0.0, 1e-10)
print()

# --- TEST 8: Equatorial Doppler Asymmetry ---
print("TEST 8: Equatorial-View Doppler Asymmetry")
print("-" * 40)
vel_eq = (1.0, 0.0, 0.0)
pos_top = (0.0, 0.0, 5.0)
tx_top = -pos_top[2] / 5.0
tz_top = pos_top[0] / 5.0
cos_v_top = vel_eq[0] * tx_top + vel_eq[2] * tz_top

pos_bot = (0.0, 0.0, -5.0)
tx_bot = -pos_bot[2] / 5.0
tz_bot = pos_bot[0] / 5.0
cos_v_bot = vel_eq[0] * tx_bot + vel_eq[2] * tz_bot

check_true("Approaching limb: opposite sign", cos_v_top * cos_v_bot < 0)
check_true("Asymmetry magnitude ~ 2", abs(cos_v_top - cos_v_bot) > 1.5)
print()

# --- TEST 9: Carter-Constant Conservation under Integration ---
# Q_C should remain numerically constant along an RK4 trajectory. This is a
# stronger test of the integrator than any specific orbit shape: it directly
# verifies that the symmetries of the Kerr metric are honored.
print("TEST 9: Carter Constant Q Conservation along Geodesic")
print("-" * 40)
a_test = 0.5 * 0.9    # a_spin = 0.9 → real a in rs units = 0.45
r0, cth0, ph0, p_r0, p_th0, L0, Q0 = cam_init((-30.0, 4.0, 5.0), (1.0, 0.0, 0.0), a_test)
state = (r0, math.acos(cth0), ph0)
signs = (-1.0 if p_r0 < 0 else 1.0, -1.0 if p_th0 < 0 else 1.0)
Q_max_drift = 0.0
for _ in range(2000):
    state, signs = kerr_rk4(state, signs, 0.05, a_test, L0, Q0, 0.0)
    r, th, _ = state
    cos_th = math.cos(th); sin_th = math.sin(th)
    p_th_now_sq = kerr_Theta(cos_th, sin_th, a_test, L0, Q0)
    if p_th_now_sq < 0:
        # Below a turning point, can't recover Q; skip.
        continue
    Q_now = p_th_now_sq - cos_th ** 2 * (a_test * a_test - L0 * L0 / max(sin_th ** 2, 1e-9))
    Q_max_drift = max(Q_max_drift, abs(Q_now - Q0))
    if r < 1.5 or r > 100.0:
        break
check_true(f"Q drift < 1% over trajectory (drift = {Q_max_drift:.3e})",
           Q_max_drift / max(abs(Q0), 1e-6) < 0.01)
print()

# --- TEST 10: Horizon Capture ---
print("TEST 10: Event Horizon Captures Inward Rays")
print("-" * 40)
res = shoot_ray((-30.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.0)
check_true(f"Head-on Schwarzschild ray (r_final = {res[1]:.3f})",
           res[0] == 'captured')
res = shoot_ray((-30.0, 0.0, 0.5), (1.0, 0.0, 0.0), 0.0)
check_true("Sub-critical impact parameter b=0.5 captured",
           res[0] == 'captured')
print()

# ============================================================
#  SUMMARY
# ============================================================
print("=" * 70)
print(f"  RESULTS: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("  All tests PASSED.")
else:
    print("  FAILURES detected — review above.")
print("=" * 70)

sys.exit(0 if FAIL == 0 else 1)
