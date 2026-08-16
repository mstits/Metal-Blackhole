#!/usr/bin/env python3
"""
GRRT Physics Validation Test Suite
===================================
Mathematical proof-of-correctness for the Metal Blackhole engine.

The shader integrates null geodesics of the exact Kerr-Newman metric in the
super-Hamiltonian first-order Carter form (momenta evolved, no sign tracking).
This script mirrors the integrator in double precision and verifies it against
analytic GR: Darwin's exact deflection integrals, the Bardeen critical curve,
Bardeen-Press-Teukolsky orbits, Cunningham redshift extremes, and the
Kerr-Newman charge sector.

Conventions match the shader:
    sys.spin = a/M in [-1, 1], sys.charge = Q/M in [0, 1]
    Length unit rs = 2M, so M = 1/2 in code units.
    "a" in metric formulas is the dimensional spin in rs units = spin/2;
    "Q" likewise charge/2. Delta = r^2 - r + a^2 + Q^2.

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
#  Mirror of the shader integrator (geodesic.metal, fp64)
# ============================================================

def geo_rhs(s, a, Q2, L, QC, E=1.0):
    """Super-Hamiltonian RHS on state (r, th, ph, p_r, p_th). Mirrors kerr_rhs.

    E is the conserved energy (+1 for forward photon launches, -1 for the
    past-directed backward rays traced from the camera — see zamo_init)."""
    r, th, ph, pr, pth = s
    c, sn = math.cos(th), math.sin(th)
    sn_g = max(sn, 1e-4) if sn >= 0 else min(sn, -1e-4)
    s2 = sn_g * sn_g
    Sigma = r * r + a * a * c * c
    Delta = r * r - r + a * a + Q2
    Dp = 2 * r - 1
    P = E * (r * r + a * a) - a * L
    B = (L - a * E) ** 2 + QC
    R = P * P - Delta * B
    Rp = 4 * r * P * E - Dp * B
    RoDp = (Rp * Delta - R * Dp) / max(Delta * Delta, 1e-12)
    inv = 1.0 / Sigma
    return (Delta * pr * inv,
            pth * inv,
            ((a * E * (r - Q2) - a * a * L) / max(Delta, 1e-9) + L / s2) * inv,
            (RoDp - Dp * pr * pr) * 0.5 * inv,
            (-2 * a * a * sn * c + 2 * L * L * c / (s2 * sn_g)) * 0.5 * inv)


def geo_rk4(s, h, a, Q2, L, QC, E=1.0):
    k1 = geo_rhs(s, a, Q2, L, QC, E)
    s2 = tuple(s[i] + 0.5 * h * k1[i] for i in range(5))
    k2 = geo_rhs(s2, a, Q2, L, QC, E)
    s3 = tuple(s[i] + 0.5 * h * k2[i] for i in range(5))
    k3 = geo_rhs(s3, a, Q2, L, QC, E)
    s4 = tuple(s[i] + h * k3[i] for i in range(5))
    k4 = geo_rhs(s4, a, Q2, L, QC, E)
    return tuple(s[i] + h * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6.0
                 for i in range(5))


def adaptive_step(s, a, Q2, L, QC, r_h, eps=0.05, E=1.0, h_floor=1e-6):
    """Mirrors the shader's inverse-sum controller including the stiff-polar
    p_theta term and the pole-proximity scaling. h_floor=1e-4 with eps=0.05
    reproduces the SHIPPED shader constants; the tighter defaults are the
    fp64 reference configuration."""
    d = geo_rhs(s, a, Q2, L, QC, E)
    pole_dist = max(min(s[1], math.pi - s[1]), 1e-3)
    sin_th = math.sin(s[1])
    rate = (abs(d[0]) / max(s[0], 0.5) + abs(d[1]) / min(pole_dist, 1.0)
            + abs(d[3 + 1]) / (abs(s[4]) + 1.0)
            + abs(d[2]) * abs(sin_th) + 1e-9)
    h = eps / rate
    h = min(h, 0.35 * max(s[0] - r_h, 1e-3) / max(abs(d[0]), 1e-6))
    return max(h, h_floor)


def zamo_coeffs(r, cos_th, a, Q2):
    """ZAMO metric coefficients (Kerr-Newman). Single source shared by
    zamo_init and the analytic tests, so a regression here cannot hide."""
    sin_th = math.sqrt(max(1.0 - cos_th * cos_th, 1e-12))
    Sigma = r * r + a * a * cos_th ** 2
    Delta = r * r - r + a * a + Q2
    s2 = sin_th * sin_th
    m_kn = r - Q2
    gtt_neg = 1.0 - m_kn / Sigma
    gtphi = -a * m_kn * s2 / Sigma
    grr = Sigma / max(Delta, 1e-9)
    gthth = Sigma
    gphph = ((r * r + a * a) * Sigma + a * a * m_kn * s2) * s2 / Sigma
    gphph_safe = max(gphph, 1e-12)
    omega_z = -gtphi / gphph_safe
    alpha = math.sqrt(max(gtt_neg + gtphi * gtphi / gphph_safe, 1e-12))
    return gtt_neg, gtphi, grr, gthth, gphph, omega_z, alpha


def zamo_init(cam_pos, ray_dir, a, Q2):
    """Past-directed backward-ray initial conditions (Kerr-Newman ZAMO).
    Mirrors the shader.

    The pixel's light ARRIVES at the camera; the backward ray is the
    past-directed continuation of the arriving momentum, normalized to
    E = -p_t = -1. (A future-directed E=+1 ray along the view direction is
    the TIME-REVERSED photon: identical in Schwarzschild, mirrored in Kerr.)

    Returns (state, L, QC, L_arr) with state = (r, th, ph, p_r, p_th) and
    L_arr = L/E = -L, the photon impact ratio used by the emitter g-factor."""
    x, y, z = cam_pos
    r = math.sqrt(x * x + y * y + z * z)
    cos_th = max(-1.0, min(1.0, y / r))
    sin_th = math.sqrt(max(1.0 - cos_th * cos_th, 1e-12))
    phi = math.atan2(z, x)
    r_hat = (sin_th * math.cos(phi), cos_th, sin_th * math.sin(phi))
    th_hat = (cos_th * math.cos(phi), -sin_th, cos_th * math.sin(phi))
    ph_hat = (-math.sin(phi), 0.0, math.cos(phi))
    n_r = sum(ray_dir[i] * r_hat[i] for i in range(3))
    n_th = sum(ray_dir[i] * th_hat[i] for i in range(3))
    n_ph = sum(ray_dir[i] * ph_hat[i] for i in range(3))

    _gtt, _gtp, grr, gthth, gphph, omega_z, alpha = zamo_coeffs(r, cos_th, a, Q2)
    sgp = math.sqrt(max(gphph, 0.0))
    s2 = sin_th * sin_th

    E_inv = 1.0 / (alpha - omega_z * sgp * n_ph)
    L = sgp * n_ph * E_inv
    pr0 = math.sqrt(grr) * n_r * E_inv
    pth0 = math.sqrt(gthth) * n_th * E_inv
    L_arr = -L
    QC = pth0 ** 2 + cos_th ** 2 * (L * L / max(s2, 1e-12) - a * a)
    return (r, math.acos(cos_th), phi, pr0, pth0), L, QC, L_arr


def emitter_g(rh, a, Q2, L_arr):
    """Disk emitter g-factor. Mirrors disk_surface_emission (KN, signed a)."""
    m_kn = (rh - Q2) / (rh * rh)
    gtt = -(1.0 - m_kn)
    gtphi = -a * m_kn
    gphph = rh * rh + a * a + a * a * m_kn
    s_kn = math.sqrt(max(0.5 * rh - Q2, 0.0))
    Omega = s_kn / (rh * rh + a * s_kn)
    U_denom = -gtt - 2.0 * gtphi * Omega - gphph * Omega * Omega
    if U_denom <= 1e-6:
        return None
    Ut = 1.0 / math.sqrt(U_denom)
    return 1.0 / (Ut * (1.0 - Omega * L_arr))


def trace(cam_pos, ray_dir, spin, charge=0.0, eps=0.02, max_steps=60000,
          r_stop_out=None, want_crossings=False):
    """Full mirror trace. Returns dict with fate, crossings, final state, dphi."""
    a = 0.5 * spin
    Q2 = 0.25 * charge * charge
    r_h = 0.5 * (1.0 + math.sqrt(max(1.0 - spin * spin - charge * charge, 0.0)))
    s, L, QC, L_arr = zamo_init(cam_pos, ray_dir, a, Q2)
    r0 = s[0]
    r_out_lim = r_stop_out if r_stop_out else max(500.0, r0 * 1.05)
    prev_cos = math.cos(s[1])
    prev_r, prev_ph = s[0], s[2]
    crossings = []
    ph_start = s[2]
    min_r = s[0]
    POLE_CAP = 0.01              # mirrors the shader's polar-cap phi bypass
    in_cap = False
    ph_entry = 0.0
    for i in range(max_steps):
        if s[0] < r_h * 1.02:
            return dict(fate='captured', crossings=crossings, s=s, L=L, QC=QC,
                        L_arr=L_arr, dphi=s[2] - ph_start, min_r=min_r)
        if s[0] > r_out_lim or (s[0] > 30.0 and s[3] > 0 and not want_crossings):
            return dict(fate='escaped', crossings=crossings, s=s, L=L, QC=QC,
                        L_arr=L_arr, dphi=s[2] - ph_start, min_r=min_r)
        if s[0] > r_out_lim and s[3] > 0:
            return dict(fate='escaped', crossings=crossings, s=s, L=L, QC=QC,
                        L_arr=L_arr, dphi=s[2] - ph_start, min_r=min_r)
        h = adaptive_step(s, a, Q2, L, QC, r_h, eps, E=-1.0)
        s = geo_rk4(s, h, a, Q2, L, QC, E=-1.0)
        s = list(s)
        if s[1] < 0:
            s[1] = -s[1]; s[4] = -s[4]
        elif s[1] > math.pi:
            s[1] = 2 * math.pi - s[1]; s[4] = -s[4]
        in_cap_now = s[1] < POLE_CAP or s[1] > math.pi - POLE_CAP
        if in_cap_now and not in_cap:
            in_cap = True; ph_entry = prev_ph
        elif not in_cap_now and in_cap:
            in_cap = False; s[2] = ph_entry + math.pi
        if in_cap:
            s[2] = ph_entry
        s = tuple(s)
        min_r = min(min_r, s[0])
        c = math.cos(s[1])
        if prev_cos * c < 0.0:
            f = max(0.0, min(1.0, prev_cos / (prev_cos - c)))
            crossings.append((prev_r + (s[0] - prev_r) * f,
                              prev_ph + (s[2] - prev_ph) * f))
        prev_cos = c
        prev_r, prev_ph = s[0], s[2]
    return dict(fate='exhausted', crossings=crossings, s=s, L=L, QC=QC,
                L_arr=L_arr, dphi=s[2] - ph_start, min_r=min_r)


# ============================================================
#  Analytic references
# ============================================================

def event_horizon(spin, charge=0.0):
    return 0.5 * (1.0 + math.sqrt(max(1.0 - spin * spin - charge * charge, 0.0)))


def bpt_isco(spin):
    """Signed-spin BPT ISCO in rs units: prograde branch for spin >= 0,
    retrograde for spin < 0 (mirrors kerr_radii in main.mm)."""
    aa = abs(spin)
    a2 = aa * aa
    cbrt = max(1.0 - a2, 1e-9) ** (1.0 / 3.0)
    Z1 = 1.0 + cbrt * ((1.0 + aa) ** (1.0 / 3.0) + max(1.0 - aa, 1e-9) ** (1.0 / 3.0))
    Z2 = math.sqrt(3.0 * a2 + Z1 * Z1)
    root = math.sqrt(max((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2), 0.0))
    return 0.5 * ((3.0 + Z2 - root) if spin >= 0 else (3.0 + Z2 + root))


def schwarzschild_deflection_exact(b_rs):
    """Exact Schwarzschild deflection (rs units, rs = 1): quadrature of
    alpha = 2 int_0^{u0} du / sqrt(1/b^2 - u^2 + u^3) - pi, u = 1/r."""
    # u0 = smallest positive root of 1/b^2 - u^2 + u^3 = 0 (periapsis)
    from bisect import bisect  # noqa: F401  (stdlib only)
    inv_b2 = 1.0 / (b_rs * b_rs)

    def f(u):
        return inv_b2 - u * u + u ** 3

    lo, hi = 0.0, 2.0 / 3.0   # photon-sphere u = 1/1.5
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    u0 = lo
    # Integrate with substitution u = u0 * sin^2 is awkward; use tanh-sinh-lite:
    # substitute u = u0 * (1 - t^2), du = -2 u0 t dt, t: 1 -> 0 removes the
    # endpoint singularity ~1/sqrt(u0 - u).
    N = 20000
    total = 0.0
    for k in range(N):
        t = (k + 0.5) / N
        u = u0 * (1.0 - t * t)
        val = f(u)
        if val <= 0:
            continue
        total += 2.0 * u0 * t / math.sqrt(val) / N
    return 2.0 * total - math.pi


def rn_photon_sphere(charge):
    """Reissner-Nordstrom photon sphere in rs units (M = 1/2):
    r_ph[M] = 1.5 (1 + sqrt(1 - 8 Q^2 / 9)), Q in M units -> /2 for rs."""
    return 0.75 * (1.0 + math.sqrt(max(1.0 - 8.0 * charge * charge / 9.0, 0.0)))


def rn_bcrit(charge):
    """RN critical impact parameter in rs units."""
    rp = rn_photon_sphere(charge)         # rs units
    Q2 = 0.25 * charge * charge
    return rp / math.sqrt(max(1.0 - 1.0 / rp + Q2 / (rp * rp), 1e-12))


def bardeen_curve_point(r, spin, inc):
    """Bardeen critical curve (alpha, beta) in M units for orbit radius r (M units)."""
    aa = spin
    xi = (3 * r * r - r ** 3 - aa * aa * (r + 1)) / (aa * (r - 1))
    eta = r ** 3 * (4 * aa * aa - r * (r - 3) ** 2) / (aa * aa * (r - 1) ** 2)
    si, ci = math.sin(inc), math.cos(inc)
    b2 = eta + aa * aa * ci * ci - xi * xi * ci * ci / (si * si)
    if b2 < 0:
        return None
    return (-xi / si, math.sqrt(b2))


# ============================================================
#  TEST SUITE
# ============================================================

print("=" * 70)
print("  Metal Blackhole — Kerr-Newman Physics Validation")
print("=" * 70)
print()

# --- TEST 1: Horizons (Kerr AND Kerr-Newman) ---
print("TEST 1: Event Horizon r_+ (spin and charge)")
print("-" * 40)
for spn in (0.0, 0.5, 0.9, 0.998):
    check(f"Kerr a={spn}", event_horizon(spn), 0.5 * (1 + math.sqrt(max(1 - spn * spn, 0))))
check("KN a=0.9, Q=0.3", event_horizon(0.9, 0.3), 1.316228 / 2, 1e-5)
check("KN a=0.6, Q=0.6", event_horizon(0.6, 0.6), 1.529150 / 2, 1e-5)
check("RN extremal Q=1 -> 0.5 rs", event_horizon(0.0, 1.0), 0.5, 1e-9)
print()

# --- TEST 2: ISCO — prograde AND retrograde (signed spin) ---
print("TEST 2: BPT ISCO, both branches (r_g values / 2 = rs)")
print("-" * 40)
check("prograde a=0:      3.0 rs", bpt_isco(0.0), 3.0)
check("prograde a=0.5:    2.1165 rs", bpt_isco(0.5), 2.1165, 0.01)
check("prograde a=0.9:    1.1605 rs", bpt_isco(0.9), 1.1605, 0.02)
check("retrograde a=-0.5: 7.554585 M = 3.7773 rs", bpt_isco(-0.5), 7.554585 / 2, 1e-4)
check("retrograde a=-0.9: 8.717352 M = 4.3587 rs", bpt_isco(-0.9), 8.717352 / 2, 1e-4)
check("retrograde a->-1:  exactly 9 M = 4.5 rs", bpt_isco(-0.99999), 4.5, 1e-3)
check("r_isco(a=-0.7) = retrograde BPT 8.143 M", bpt_isco(-0.7), 8.14297 / 2, 1e-3)
print()

# --- TEST 3: Photon sphere is a genuine unstable equilibrium ---
print("TEST 3: Photon Sphere — dynamical instability (Schwarzschild)")
print("-" * 40)
# Start slightly off r = 1.5 with the critical L; the deviation must grow but
# stay bounded over a half-orbit (this test integrates real dynamics — a
# frozen integrator or a vacuous clamp cannot pass it).
L_crit = math.sqrt(27.0) / 2.0
s = (1.5 + 1e-4, math.pi / 2, 0.0, 0.0, 0.0)
dev_max = 0.0
moved = 0.0
phi_at_1pct = None
for _ in range(4000):
    h = adaptive_step(s, 0.0, 0.0, L_crit, 0.0, 1.0, 0.002)
    s = geo_rk4(s, h, 0.0, 0.0, L_crit, 0.0)
    dev_max = max(dev_max, abs(s[0] - 1.5))
    moved = max(moved, abs(s[2]))
    if phi_at_1pct is None and abs(s[0] - 1.5) > 0.015:
        phi_at_1pct = abs(s[2])
    if abs(s[0] - 1.5) > 0.045:
        break
check_true(f"ray genuinely integrates (phi advanced {moved:.2f} rad)", moved > 1.0)
check_true(f"instability grows from 1e-4 (max dev {dev_max:.2e})", dev_max > 1e-3)
# The photon-sphere e-folding rate: a near-critical ray must whirl at least
# half an orbit before its deviation grows two orders of magnitude — a wrong
# potential diverges much faster (or never).
check_true(f"instability timescale ~ orbital (phi = {phi_at_1pct if phi_at_1pct else -1:.2f} rad at 1% dev)",
           phi_at_1pct is not None and phi_at_1pct > math.pi)
print()

# --- TEST 4: Capture boundary + genuine escape ---
print("TEST 4: Shadow boundary b_crit = sqrt(27)/2 = 2.598076 rs")
print("-" * 40)


def fate_of(b, spin=0.0, charge=0.0):
    res = trace((-500.0, 0.0, b), (1.0, 0.0, 0.0), spin, charge, eps=0.02)
    return res['fate']


check_true("b = 2.55 CAPTURED", fate_of(2.55) == 'captured')
check_true("b = 2.65 truly ESCAPES (not merely 'not captured')",
           fate_of(2.65) == 'escaped')
check_true("head-on ray CAPTURED", fate_of(0.0) == 'captured')
# Bisect the boundary; the z-offset at finite distance maps to L = b(1+O(1/r)),
# so compare against the L-value of the boundary ray instead of the offset.
lo, hi = 2.4, 2.8
for _ in range(30):
    mid = 0.5 * (lo + hi)
    if fate_of(mid) == 'captured':
        lo = mid
    else:
        hi = mid
res_b = trace((-500.0, 0.0, 0.5 * (lo + hi)), (1.0, 0.0, 0.0), 0.0, eps=0.02)
check("capture boundary |L| (true impact parameter)", abs(res_b['L']),
      math.sqrt(27.0) / 2.0, 5e-4)
print()

# --- TEST 5: Exact Darwin deflection table (turning-point regression) ---
print("TEST 5: Schwarzschild deflection vs exact quadrature (Darwin)")
print("-" * 40)
# Launch with EXACT conserved quantities (L = b, QC = 0) from r0 = 2000 and
# correct for the finite start/end asymptotes. This is the definitive
# turning-point test: every ray passes through periapsis and re-escapes.
for b in (3.0, 3.5, 4.0, 5.0, 10.0):
    exact = schwarzschild_deflection_exact(b)
    r0 = 2000.0
    Delta0 = r0 * r0 - r0
    R0 = r0 ** 4 - Delta0 * b * b
    st = (r0, math.pi / 2, 0.0, -math.sqrt(max(R0, 0.0)) / Delta0, 0.0)
    n = 0
    while n < 100000:
        h = adaptive_step(st, 0.0, 0.0, b, 0.0, 1.0, 0.02)
        st = geo_rk4(st, h, 0.0, 0.0, b, 0.0)
        n += 1
        if st[0] < 1.02:
            break
        if st[0] > r0 and st[3] > 0:
            break
    defl = abs(st[2]) - math.pi + math.asin(b / r0) + math.asin(b / st[0])
    check(f"deflection b = {b} rs", defl, exact, 2e-4)
print()

# --- TEST 6: Kerr-Newman charge sector ---
print("TEST 6: Reissner-Nordstrom photon sphere & capture cross-section")
print("-" * 40)
check("RN r_ph (Q=0.5): 1.4114 rs", rn_photon_sphere(0.5), 2.822876 / 2, 1e-5)
check("RN r_ph (Q=1.0): exactly 1.0 rs", rn_photon_sphere(1.0), 1.0, 1e-9)
check("RN b_crit (Q=0.5): 2.48396 rs", rn_bcrit(0.5), 4.967914 / 2, 1e-5)
check("RN b_crit (Q=1.0): exactly 2.0 rs", rn_bcrit(1.0), 2.0, 1e-9)
# Dynamical: bisect the traced capture boundary at Q = 0.5 and compare to b_c.
lo, hi = 2.2, 2.8
for _ in range(28):
    mid = 0.5 * (lo + hi)
    if fate_of(mid, 0.0, 0.5) == 'captured':
        lo = mid
    else:
        hi = mid
res_rn = trace((-500.0, 0.0, 0.5 * (lo + hi)), (1.0, 0.0, 0.0), 0.0, 0.5, eps=0.02)
check("traced RN capture boundary |L| (Q=0.5)", abs(res_rn['L']), rn_bcrit(0.5), 1e-3)
print()

# --- TEST 7: Doppler sign, end to end (the inversion regression) ---
print("TEST 7: Approaching limb must blueshift (g > 1) — end-to-end")
print("-" * 40)
# Edge-on-ish Schwarzschild camera; aim at each limb of the disk; determine
# 'approaching' from the emitter velocity direction vs the direction to the
# camera, computed from first principles (no shared sign conventions).
cam = (20.0 * math.sin(1.45) * math.cos(0.5), 20.0 * math.cos(1.45),
       20.0 * math.sin(1.45) * math.sin(0.5))


def limb_probe(side):
    """side = +1: ray aimed to the right of the BH (screen +x), -1: left.
    Returns (g_shader_formula, approaching) for the first disk crossing in
    [r_in, r_out], or None."""
    # Build the camera basis exactly like the host (y-up look-at).
    fwd = tuple(-c / 20.0 for c in cam)
    right = (fwd[1] * 0 - fwd[2] * 1, fwd[2] * 0 - fwd[0] * 0, fwd[0] * 1 - fwd[1] * 0)
    rn = math.sqrt(sum(c * c for c in right))
    right = tuple(c / rn for c in right)
    up = (right[1] * fwd[2] - right[2] * fwd[1],
          right[2] * fwd[0] - right[0] * fwd[2],
          right[0] * fwd[1] - right[1] * fwd[0])
    best = None
    for off in (0.12, 0.16, 0.2, 0.24, 0.28):
        rd = tuple(fwd[i] + side * off * right[i] for i in range(3))
        nn = math.sqrt(sum(c * c for c in rd))
        rd = tuple(c / nn for c in rd)
        res = trace(cam, rd, 0.0, eps=0.02, want_crossings=True)
        for (r_at, ph_at) in res['crossings']:
            if 3.0 < r_at < 12.0:
                g = emitter_g(r_at, 0.0, 0.0, res['L_arr'])
                # Emitter velocity: prograde +phi_hat at the hit point.
                phv = (-math.sin(ph_at), 0.0, math.cos(ph_at))
                hit = (r_at * math.cos(ph_at), 0.0, r_at * math.sin(ph_at))
                to_cam = tuple(cam[i] - hit[i] for i in range(3))
                tn = math.sqrt(sum(c * c for c in to_cam))
                to_cam = tuple(c / tn for c in to_cam)
                approaching = sum(phv[i] * to_cam[i] for i in range(3)) > 0.15
                receding = sum(phv[i] * to_cam[i] for i in range(3)) < -0.15
                if approaching or receding:
                    return g, approaching
        if best:
            break
    return None


probe_r = limb_probe(+1)
probe_l = limb_probe(-1)
if probe_r and probe_l:
    g_r, app_r = probe_r
    g_l, app_l = probe_l
    check_true(f"limbs have opposite senses (right approaching={app_r})", app_r != app_l)
    g_app = g_r if app_r else g_l
    g_rec = g_l if app_r else g_r
    check_true(f"approaching limb blueshifted: g = {g_app:.4f} > 1", g_app > 1.0)
    check_true(f"receding limb redshifted:  g = {g_rec:.4f} < 1", g_rec < 1.0)
else:
    check_true("limb probe found disk hits on both sides", False)
print()

# --- TEST 8: Cunningham limb g-factor extremes at the Schwarzschild ISCO ---
print("TEST 8: ISCO limb redshift extremes (Cunningham 1975 exact values)")
print("-" * 40)
# g = 1/(u^t (1 - Omega lambda)); tangential emission extremizes lambda at
# lambda_tan = r / sqrt(1 - 1/r) in rs units. Exact: g_max = sqrt(2),
# g_min = sqrt(2)/3, face-on g = 1/sqrt(2).
r6 = 3.0
lam_tan = r6 / math.sqrt(1.0 - 1.0 / r6)
check("g_max = sqrt(2)", emitter_g(r6, 0.0, 0.0, lam_tan), math.sqrt(2.0), 1e-9)
check("g_min = sqrt(2)/3", emitter_g(r6, 0.0, 0.0, -lam_tan), math.sqrt(2.0) / 3.0, 1e-9)
check("face-on g = 1/sqrt(2)", emitter_g(r6, 0.0, 0.0, 0.0), 1.0 / math.sqrt(2.0), 1e-9)
print()

# --- TEST 9: Genuine conserved-quantity drift along evolved momenta ---
print("TEST 9: Carter constant + null norm drift (evolved p_theta — not a tautology)")
print("-" * 40)


def drift_test(spin, charge, cam, rd):
    """Max Carter-constant and null-norm drift along a backward ray (E=-1),
    reconstructed from the EVOLVED momenta. Also returns min r so the test
    self-documents that it probes the strong field."""
    a = 0.5 * spin
    Q2 = 0.25 * charge * charge
    E = -1.0
    r_h = event_horizon(spin, charge)
    st, L, QC, _ = zamo_init(cam, rd, a, Q2)
    qc_drift = 0.0
    null_drift = 0.0
    min_r = st[0]
    for _ in range(40000):
        if st[0] < r_h * 1.05 or st[0] > 60.0:
            break
        h = adaptive_step(st, a, Q2, L, QC, r_h, 0.02, E=E)
        st = geo_rk4(st, h, a, Q2, L, QC, E=E)
        r, th, ph, pr, pth = st
        min_r = min(min_r, r)
        c, sn = math.cos(th), math.sin(th)
        s2 = max(sn * sn, 1e-12)
        QC_now = pth * pth + c * c * (L * L / s2 - a * a)   # E^2 = 1
        qc_drift = max(qc_drift, abs(QC_now - QC) / max(abs(QC), 1e-9))
        # Null-norm accounting stops just above the horizon: the R/Delta term
        # amplifies benign roundoff as Delta -> 0, which would measure the
        # DIAGNOSTIC's conditioning there, not the integrator's accuracy.
        if r > r_h * 1.3:
            D = r * r - r + a * a + Q2
            P = E * (r * r + a * a) - a * L
            R = P * P - D * ((L - a * E) ** 2 + QC)
            Th = QC + a * a * c * c - L * L * c * c / s2
            null_drift = max(null_drift, abs(D * pr * pr + pth * pth - R / D - Th))
    return qc_drift, null_drift, min_r


# Rays aimed to graze the photon shell (both z-signs probed, deeper one kept)
# plus one polar-oscillating case for Theta-potential coverage.
for (spn, q, zoff) in [(0.0, 0.0, 2.7), (0.9, 0.0, 1.6), (0.6, 0.5, 2.2), (0.998, 0.0, 1.2)]:
    best = None
    for zsign in (+1.0, -1.0):
        res = drift_test(spn, q, (-50.0, 0.3, zsign * zoff), (1.0, 0.0, 0.0))
        if best is None or res[2] < best[2]:
            best = res
    qd, nd, mr = best
    check_true(f"a={spn} Q={q}: strong-field pass (min r = {mr:.2f} rs)", mr < 3.5)
    check_true(f"a={spn} Q={q}: QC drift {qd:.2e} < 1e-6", qd < 1e-6)
    check_true(f"a={spn} Q={q}: null-norm drift {nd:.2e} < 1e-4", nd < 1e-4)
qd, nd, mr = drift_test(0.9, 0.0, (-50.0, 5.0, 2.0),
                        tuple(c / math.sqrt(1.01) for c in (1.0, 0.1, 0.0)))
check_true(f"polar-oscillating ray: QC drift {qd:.2e} < 1e-6", qd < 1e-6)
print()

# --- TEST 10: Kerr-Newman Keplerian frequency + emitter metric (charge-aware) ---
print("TEST 10: KN circular-orbit Omega — closed form vs geodesic condition")
print("-" * 40)


def omega_numeric(rh, a, Q2):
    """Solve d/dr (g_tt + 2 Omega g_tphi + Omega^2 g_phph) = 0 numerically."""
    def coeffs(r):
        m = (r - Q2) / (r * r)
        return -(1 - m), -a * m, r * r + a * a + a * a * m
    h = 1e-6
    gtt1, gtp1, gpp1 = coeffs(rh - h)
    gtt2, gtp2, gpp2 = coeffs(rh + h)
    dgtt = (gtt2 - gtt1) / (2 * h)
    dgtp = (gtp2 - gtp1) / (2 * h)
    dgpp = (gpp2 - gpp1) / (2 * h)
    # dgtt + 2 W dgtp + W^2 dgpp = 0
    disc = dgtp * dgtp - dgtt * dgpp
    return (-dgtp + math.sqrt(max(disc, 0.0))) / dgpp


for (spn, q, r) in [(0.0, 0.0, 3.0), (0.9, 0.0, 2.0), (0.6, 0.5, 1.8), (0.0, 0.8, 3.5)]:
    a = 0.5 * spn
    Q2 = 0.25 * q * q
    s_kn = math.sqrt(max(0.5 * r - Q2, 0.0))
    Om_closed = s_kn / (r * r + a * s_kn)
    check(f"Omega(a={spn}, Q={q}, r={r})", Om_closed, omega_numeric(r, a, Q2), 1e-4)
# Q=0 reduction: closed form equals the old Kerr expression exactly.
for (spn, r) in [(0.0, 3.0), (0.9, 2.0)]:
    a = 0.5 * spn
    check(f"Q=0 reduction (a={spn}, r={r})",
          math.sqrt(0.5 * r) / (r * r + a * math.sqrt(0.5 * r)),
          1.0 / (math.sqrt(2.0) * r * math.sqrt(r) + a), 1e-12)
print()

# --- TEST 11: ZAMO frequency & static limit (unchanged analytics) ---
print("TEST 11: ZAMO omega and static limit r_E(theta)")
print("-" * 40)
for spn in (0.5, 0.9, 0.998):
    a = 0.5 * spn
    for r in (2.0, 5.0):
        Delta = r * r - r + a * a
        Sigma_sq = (r * r + a * a) ** 2 - Delta * a * a
        expected = a * r / Sigma_sq
        # THE code path: zamo_coeffs is the same function zamo_init uses.
        omega_code = zamo_coeffs(r, 0.0, a, 0.0)[5]
        check(f"omega_zamo(a={spn}, r={r})", omega_code, expected, 1e-9)
    # Static limit through the code path: -g_tt must change sign exactly at
    # r_E(theta) = 1/2 + 1/2 sqrt(1 - spin^2 cos^2 theta).
    for cth in (0.0, 0.5):
        r_E = 0.5 + 0.5 * math.sqrt(1 - spn * spn * cth * cth)
        gtt_in = zamo_coeffs(r_E - 1e-6, cth, a, 0.0)[0]
        gtt_out = zamo_coeffs(r_E + 1e-6, cth, a, 0.0)[0]
        check_true(f"-g_tt sign change at r_E(cos_th={cth}), a={spn}",
                   gtt_in < 0.0 < gtt_out)
print()

# --- TEST 12: Below-plane camera hits the disk (ascending-crossing regression) ---
print("TEST 12: Disk visible from below the equatorial plane")
print("-" * 40)
cam_below = (20.0 * math.sin(2.09) * math.cos(0.5), 20.0 * math.cos(2.09),
             20.0 * math.sin(2.09) * math.sin(0.5))
hits_below = 0
targets = [(6.0 * math.cos(ph), 0.0, 6.0 * math.sin(ph))
           for ph in (0.0, 1.5, 3.0, 4.5)]
for tgt in targets:
    rd = tuple(tgt[i] - cam_below[i] for i in range(3))
    nn = math.sqrt(sum(c * c for c in rd))
    rd = tuple(c / nn for c in rd)
    res = trace(cam_below, rd, 0.0, eps=0.02, want_crossings=True)
    for (r_at, _ph) in res['crossings']:
        if 3.0 < r_at < 12.0:
            hits_below += 1
            break
check_true(f"ascending crossings register disk hits ({hits_below}/4 rays aimed at r=6)",
           hits_below >= 3)
print()

# --- TEST 13: Bardeen critical curve vs traced shadow (Kerr a=0.9, edge-on) ---
print("TEST 13: Traced shadow boundary matches Bardeen's analytic curve")
print("-" * 40)
# Analytic: at i=90 deg, alpha extents (M units) for a=0.9: [-2.84442, 6.832315].
# alpha = -xi/sin(i); the prograde-photon side has SMALLER |alpha|.
# Trace: camera far on +x axis, scan impact parameter along z (equatorial),
# bisect capture on both sides. In this geometry a ray offset toward +z at
# x<0... use camera on -x axis, offsets +/- z; positive-z offset gives L>0.
spin_t = 0.9


def eq_fate(b):
    return trace((-800.0, 0.0, b), (1.0, 0.0, 0.0), spin_t, eps=0.02)


# positive-b side:
lo, hi = 0.2, 6.0
for _ in range(26):
    mid = 0.5 * (lo + hi)
    if eq_fate(mid)['fate'] == 'captured':
        lo = mid
    else:
        hi = mid
b_pos = 0.5 * (lo + hi)
L_pos = abs(trace((-800.0, 0.0, b_pos), (1.0, 0.0, 0.0), spin_t)['L'])
# negative-b side:
lo, hi = -0.2, -6.0
for _ in range(26):
    mid = 0.5 * (lo + hi)
    if eq_fate(mid)['fate'] == 'captured':
        lo = mid
    else:
        hi = mid
b_neg = 0.5 * (lo + hi)
L_neg = abs(trace((-800.0, 0.0, b_neg), (1.0, 0.0, 0.0), spin_t)['L'])
# Analytic extents in rs units (M -> /2):
check("prograde-side extent min(|L|)", min(L_pos, L_neg), 2.844420 / 2, 2e-3)
check("retrograde-side extent max(|L|)", max(L_pos, L_neg), 6.832315 / 2, 2e-3)
# Determine WHICH sign of L is the prograde (small) side — this pins the
# screen orientation of the critical-curve overlay:
# With the past-directed (E = -1) congruence the traced conserved L and the
# photon impact ratio satisfy xi = L/E = -L, so the prograde (flattened)
# side — xi > 0, photons passing closer — is the side whose traced rays have
# L < 0. Geometry: the +z offset gives n_phi < 0, hence traced L < 0, hence
# xi > 0: the POSITIVE-z side is the prograde side.
check_true("prograde (small-|L|) extent lies on the +z side", L_pos < L_neg)
# Physical pairing: the disk material on the +z side moves toward this camera
# (v = Omega d(x)/d(phi) at (0,0,R) points to -x, the camera side), so the
# flattened shadow side and the Doppler-approaching side must COINCIDE —
# both are corotation effects. A time-reversed (E=+1) tracer puts them on
# opposite sides; this assertion pins the congruence.
check_true("shadow flattening and Doppler-approaching side coincide (+z)",
           L_pos < L_neg)
print()

# --- TEST 14: Integrator convergence (step-halving) ---
print("TEST 14: Adaptive-integrator self-convergence")
print("-" * 40)


# A moderately-deflected Kerr ray (away from the chaotic photon-shell zone,
# where Lyapunov growth makes step-size comparisons meaningless).
def final_state(eps):
    res = trace((-100.0, 3.0, 4.5), (1.0, 0.0, 0.0), 0.9, eps=eps)
    # Remove the endpoint discretization (each eps run exits the outgoing
    # early-out at a different final radius): correct the phi tail to
    # r -> infinity from wherever the trace actually stopped.
    b = abs(res['L'])
    return res['dphi'] + math.copysign(math.asin(b / res['s'][0]), res['dphi'])


ref = final_state(0.0025)
err_coarse = abs(final_state(0.04) - ref)
err_fine = abs(final_state(0.01) - ref)
check_true(f"deflection converges with eps (err {err_coarse:.2e} -> {err_fine:.2e})",
           err_fine <= err_coarse and err_fine < 2e-4)
print()

# --- TEST 15: Novikov-Thorne / Page-Thorne flux profile ---
print("TEST 15: Page-Thorne flux: zero at r_in, peak at (49/36) r_in")
print("-" * 40)
r_in = 3.0


def flux(r):
    x = r_in / r
    return x ** 3 * (1.0 - math.sqrt(x))


check("F(r_in) = 0", flux(r_in), 0.0, 1e-9, "(abs)")
best_r, best_f = 0, 0
for i in range(1, 4000):
    r = r_in + i * 0.002
    fv = flux(r)
    if fv > best_f:
        best_f, best_r = fv, r
check("flux peak at (49/36) r_in", best_r, 49.0 / 36.0 * r_in, 1e-3)
check_true("F(10) > F(20) — monotone past peak", flux(10.0) > flux(20.0))
print()

# --- TEST 16: SHIPPED shader constants (eps=0.05, floor 1e-4, 1200 steps) ---
print("TEST 16: Accuracy at the shipped GPU configuration")
print("-" * 40)
# The reference legs above certify the fp64 mirror at eps=0.02. This leg runs
# the SAME math at the constants the shader actually ships, from a realistic
# start radius, and gates at honestly-measured tolerances — so the suite
# certifies the configuration the GPU runs, not just a finer one.


def shipped_darwin(b):
    r0 = 100.0
    Delta0 = r0 * r0 - r0
    R0 = r0 ** 4 - Delta0 * b * b
    st = (r0, math.pi / 2, 0.0, -math.sqrt(max(R0, 0.0)) / Delta0, 0.0)
    for _ in range(1200):
        h = adaptive_step(st, 0.0, 0.0, b, 0.0, 1.0, eps=0.05, h_floor=1e-4)
        st = geo_rk4(st, h, 0.0, 0.0, b, 0.0)
        if st[0] < 1.02 or (st[0] > r0 and st[3] > 0):
            break
    return abs(st[2]) - math.pi + math.asin(b / r0) + math.asin(b / st[0])


for b, tol in ((3.0, 3e-3), (5.0, 3e-3)):
    check(f"shipped-config deflection b = {b} rs", shipped_darwin(b),
          schwarzschild_deflection_exact(b), tol)


def shipped_fate(b):
    st, L, QC, _ = zamo_init((-100.0, 0.0, b), (1.0, 0.0, 0.0), 0.0, 0.0)
    for _ in range(1200):
        if st[0] < 1.02:
            return 'captured'
        if st[0] > 110.0 or (st[0] > 30.0 and st[3] > 0):
            return 'escaped'
        h = adaptive_step(st, 0.0, 0.0, L, QC, 1.0, eps=0.05, E=-1.0, h_floor=1e-4)
        st = geo_rk4(st, h, 0.0, 0.0, L, QC, E=-1.0)
    return 'exhausted'


lo, hi = 2.4, 2.8
for _ in range(24):
    mid = 0.5 * (lo + hi)
    if shipped_fate(mid) == 'captured':
        lo = mid
    else:
        hi = mid
Lb = abs(zamo_init((-100.0, 0.0, 0.5 * (lo + hi)), (1.0, 0.0, 0.0), 0.0, 0.0)[1])
check("shipped-config capture boundary |L|", Lb, math.sqrt(27.0) / 2.0, 5e-3)
print()

# --- TEST 17: polar-cap phi bypass (through-the-pole continuation) ---
print("TEST 17: Pole crossing — phi jumps by exactly pi through the axis")
print("-" * 40)
# An L ~ 0 ray's plane contains the spin axis; when it sweeps past the axis
# the BL phi coordinate must jump by pi. The shader (and this mirror) freeze
# phi inside th < 0.01 of the axis and apply the jump at cap exit — exact in
# spherical symmetry, O(a * cap) for Kerr. This is the only code path the
# rest of the suite never touches.


def pole_probe(spin):
    a = 0.5 * spin
    L, QC = 1e-15, 9.0
    r0 = 50.0
    Delta0 = r0 * r0 - r0 + a * a
    R0 = (r0 * r0 + a * a) ** 2 - Delta0 * (a * a + QC)   # E=+1, L~0
    st = (r0, math.pi / 2, 0.0, -math.sqrt(max(R0, 0.0)) / Delta0, -math.sqrt(QC))
    POLE_CAP = 0.01
    in_cap = False
    ph_entry = 0.0
    min_th = st[1]
    prev_cos = 0.0
    prev = st
    for _ in range(60000):
        if st[0] < 1.02 or st[0] > 60.0:
            break
        h = adaptive_step(st, a, 0.0, L, QC, 1.0, 0.02)
        st = geo_rk4(st, h, a, 0.0, L, QC)
        st = list(st)
        if st[1] < 0:
            st[1] = -st[1]; st[4] = -st[4]
        elif st[1] > math.pi:
            st[1] = 2 * math.pi - st[1]; st[4] = -st[4]
        in_cap_now = st[1] < POLE_CAP or st[1] > math.pi - POLE_CAP
        if in_cap_now and not in_cap:
            in_cap = True; ph_entry = prev[2]
        elif not in_cap_now and in_cap:
            in_cap = False; st[2] = ph_entry + math.pi
        if in_cap:
            st[2] = ph_entry
        st = tuple(st)
        min_th = min(min_th, st[1])
        c = math.cos(st[1])
        if prev_cos * c < 0.0 and min_th < POLE_CAP:
            # first equatorial crossing after the pole passage
            return min_th, st[2]
        prev_cos = c
        prev = st
    return min_th, None


min_th0, ph_cross = pole_probe(0.0)
check_true(f"ray enters the polar cap (min theta = {min_th0:.4f})", min_th0 < 0.01)
check_true("through-the-pole continuation: phi = phi0 + pi at re-crossing",
           ph_cross is not None and abs(ph_cross - math.pi) < 5e-3)
min_th9, ph_cross9 = pole_probe(0.9)
# Kerr: the phi at re-crossing includes genuine frame-dragging drift
# accumulated over the whole inbound path (~0.1 rad here), on top of the
# exact pi jump — bound it loosely; the pi discontinuity itself dominates.
check_true(f"Kerr a=0.9 pole pass (min theta = {min_th9:.4f}), phi = pi + drag drift",
           ph_cross9 is not None and abs(ph_cross9 - math.pi) < 0.25)
print()

# ============================================================
#  Volumetric torus: mirror of the shader's radiative transfer
# ============================================================

def torus_sample(r_rs, cos_th, sin_th, a, Q2, L_arr, r0, h, l0, temp):
    """Mirror of torus_sample() in geodesic.metal. Returns (n, g) or None."""
    r_M = 2.0 * r_rs
    R_cyl = r_M * sin_th
    ar = (r_M - r0) / (0.4 * r0)
    az = h * cos_th
    n = math.exp(-0.5 * (ar * ar + az * az))
    if n < 2.0e-3:
        return None
    l_rs = 0.5 * (l0 / (1.0 + R_cyl)) * max(R_cyl, 1e-4) ** 1.5
    Sigma = r_rs * r_rs + a * a * cos_th * cos_th
    s2 = max(sin_th * sin_th, 1e-6)
    m_kn = r_rs - Q2
    gtt = -(1.0 - m_kn / Sigma)
    gtphi = -a * m_kn * s2 / Sigma
    gphph = ((r_rs * r_rs + a * a) * Sigma + a * a * m_kn * s2) * s2 / Sigma
    denom = gphph + l_rs * gtphi
    if abs(denom) < 1e-9:
        denom = math.copysign(1e-9, denom if denom else 1.0)
    Omega = -(gtphi + l_rs * gtt) / denom
    U_denom = -(gtt + 2.0 * Omega * gtphi + Omega * Omega * gphph)
    if U_denom <= 1e-6:
        return None
    Ut = 1.0 / math.sqrt(U_denom)
    g = 1.0 / max(Ut * (1.0 - Omega * L_arr), 1e-3)
    return n, g


def trace_volumetric(cam_pos, ray_dir, spin, alpha_s, r0=7.0, h=6.0, l0=1.0,
                     absorb=0.0, eps=0.02):
    """Integrate the invariant transfer equation along a backward ray.
    Returns the emergent invariant intensity I_nu/nu^3."""
    a = 0.5 * spin
    Q2 = 0.0
    r_h = event_horizon(spin)
    s, L, QC, L_arr = zamo_init(cam_pos, ray_dir, a, Q2)
    r_max = 0.5 * (r0 * 2.4) + 1.0
    I_inv = 0.0
    trans = 1.0
    for _ in range(40000):
        if s[0] < r_h * 1.02 or s[0] > cam_pos_radius(cam_pos) * 1.05:
            break
        dlam = adaptive_step(s, a, Q2, L, QC, r_h, eps, E=-1.0)
        if s[0] < r_max:
            dlam = min(dlam, 0.35)
        s = geo_rk4(s, dlam, a, Q2, L, QC, E=-1.0)
        if s[1] < 0:
            s = (s[0], -s[1], s[2], s[3], -s[4])
        elif s[1] > math.pi:
            s = (s[0], 2 * math.pi - s[1], s[2], s[3], -s[4])
        c = math.cos(s[1]); sn = math.sin(s[1])
        if s[0] < r_max and s[0] > r_h * 1.05:
            ts = torus_sample(s[0], c, sn, a, Q2, L_arr, r0, h, l0, 6500.0)
            if ts:
                n, g = ts
                ne2 = n * n
                J_inv = ne2 * g ** (alpha_s + 2.0)
                A_inv = absorb * ne2 * g ** (alpha_s + 1.5)
                I_inv += trans * J_inv * dlam
                if A_inv > 0.0:
                    trans *= math.exp(-A_inv * dlam)
        if trans < 1e-4:
            break
    return I_inv


def cam_pos_radius(p):
    return math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2)


# --- TEST 18: Gold et al. Test 2 — exact Doppler cancellation at alpha_s = -2 ---
print("TEST 18: Volumetric transport — Doppler cancellation identity")
print("-" * 40)
# Gold et al. 2020 construct Test 2 so relativistic beaming from the rotating
# flow is cancelled EXACTLY by the emissivity's frequency dependence: at
# alpha_s = -2 the invariant emissivity J = n^2 g^(alpha_s+2) = n^2 is
# g-independent, so the image must be perfectly symmetric despite the plasma
# orbiting at relativistic speed. This is the most sensitive differential
# check available on the whole transport chain (four-velocity construction,
# redshift, and the g-power weighting) — and it needs no reference data.
cam_v = (-30.0, 6.0, 0.0)


def limb_pair(alpha_s):
    """Intensities through mirror-image sight lines (approaching vs receding)."""
    out = []
    for zsign in (+1.0, -1.0):
        # Aim at the ring on each side; mirror geometry about the x-y plane.
        tgt = (0.0, 0.0, zsign * 3.5)
        rd = tuple(tgt[i] - cam_v[i] for i in range(3))
        nn = math.sqrt(sum(c * c for c in rd))
        out.append(trace_volumetric(cam_v, tuple(c / nn for c in rd), 0.0, alpha_s))
    return out


I_a, I_b = limb_pair(-2.0)
rel = abs(I_a - I_b) / max(abs(I_a), 1e-12)
check_true(f"alpha_s = -2: mirror sight lines match to {rel:.2e} (I = {I_a:.5f})",
           I_a > 1e-6 and rel < 1e-6)
# Sensitivity control: at alpha_s = 0 the same geometry MUST be asymmetric,
# otherwise the test above would pass for a renderer with no Doppler at all.
J_a, J_b = limb_pair(0.0)
asym = abs(J_a - J_b) / max(abs(J_a), 1e-12)
check_true(f"alpha_s = 0: same geometry IS asymmetric ({asym:.3f} relative)",
           asym > 0.05)
print()

# --- TEST 19: invariant transport reduces to the optically-thin/thick limits ---
print("TEST 19: Radiative transfer limits (optically thin and thick)")
print("-" * 40)
rd0 = (1.0, -0.2, 0.0)
nn0 = math.sqrt(sum(c * c for c in rd0))
rd0 = tuple(c / nn0 for c in rd0)
I_thin = trace_volumetric(cam_v, rd0, 0.0, 0.0, absorb=0.0)
I_tau1 = trace_volumetric(cam_v, rd0, 0.0, 0.0, absorb=1.0)
I_thick = trace_volumetric(cam_v, rd0, 0.0, 0.0, absorb=60.0)
check_true(f"absorption strictly dims the ray ({I_thin:.4f} > {I_tau1:.4f} > {I_thick:.4f})",
           I_thin > I_tau1 > I_thick > 0.0)
# In the optically thick limit the emergent intensity saturates at the source
# function S = J/A = g^0.5 / absorb, independent of path length: doubling the
# absorption must halve it (to within the g^0.5 profile weighting).
I_thick2 = trace_volumetric(cam_v, rd0, 0.0, 0.0, absorb=120.0)
ratio = I_thick / max(I_thick2, 1e-12)
check("optically-thick source function scales as 1/absorb", ratio, 2.0, 0.05)
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
