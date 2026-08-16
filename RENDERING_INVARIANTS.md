# Rendering Invariants & Lessons Learned

Hard-won rules from debugging the geodesic integrator, the temporal pipeline,
and the coordinate singularities. **Violating any of these will reintroduce
visual or physics bugs.** Every invariant below is enforced or cross-checked by
`tests/validate_physics.py` (87 tests) where a test can express it.

---

## 1. Geodesic Integrator

### INVARIANT: Trace the PAST-directed ray (E = −1), never the time-reverse

The pixel's light ARRIVES at the camera; the backward ray is the past-directed
continuation of the arriving momentum, normalized to `E = −p_t = −1` with the
ZAMO denominator `α − ω_z √g_φφ n_φ`. A future-directed `E = +1` ray launched
along the view direction is the TIME-REVERSED photon: identical in
Schwarzschild (t → −t is an isometry there), but MIRRORED in Kerr — it flips
the sign of every `aL` coupling, so the shadow's flattened (prograde) side
lands on the side opposite the Doppler-approaching limb, which is physically
impossible (both are corotation effects and must coincide).

Regression: TEST 13 asserts the flattening/approaching pairing; the measured
rendered shadow extends 2.1× farther on the receding side (Bardeen: 2.4×,
partially disk-covered). The photon impact ratio for shading is `ξ = L/E = −L`.

### INVARIANT: Evolve momenta, never track √R / √Θ signs

The integrator uses the super-Hamiltonian first-order form on
`(r, θ, φ, p_r, p_θ)`. The classic `Σ dr/dλ = ±√R` root-tracking form has a
fatal failure mode: when an RK4 step lands past a turning point, `√(max(R,0))`
is identically zero in the forbidden region, so the coordinate freezes there
**permanently** — every ray with a radial or polar turning point stalls at
periapsis instead of turning around. The momentum form passes through turning
points smoothly (p_r crosses zero like any ODE variable).

Regression: validate_physics TEST 5 — rays at b = 3…10 rs must pass through
periapsis, re-escape, and reproduce Darwin's exact deflection to ~1e-5.

### INVARIANT: The step controller must include the p_θ stiffness term

```metal
float rate = fabs(d.r) / max(s.r, 0.5f) + fabs(d.th) / min(pole_dist, 1.0f)
           + fabs(d.pth) / (fabs(s.pth) + 1.0f)      // DO NOT REMOVE
           + fabs(d.ph) * sin_th_rate + 1e-9f;
```

Near-axis rays hit a *stiff* polar turning point: `|dθ/dλ|` vanishes exactly at
the turn while the `L²/sin³θ` wall kicks `p_θ` impulsively. Without the
`|dp_θ|` term the controller sails through the impulse, the θ-phase corrupts,
and every later equatorial-crossing radius is wrong — visible as a dotted
column of false disk hits along the projected spin axis.

### INVARIANT: Polar-cap φ bypass

Inside `θ < 0.01` of the axis, `dφ/dλ` is numerically meaningless (BL
coordinate singularity with a clamped `1/sin²θ` integrand). The kernel freezes
φ at cap entry and applies the exact through-the-pole jump of π at exit. This
is exact in spherical symmetry and O(a·0.01) for Kerr. Do not "simplify" by
integrating φ through the cap.

### INVARIANT: Crossing interpolation must be valid for BOTH directions

```metal
// CORRECT — denominator |prev_cos| + |cos_th| never vanishes in this branch
float f = clamp(prev_cos / (prev_cos - cos_th), 0.0f, 1.0f);

// WRONG — max() destroys the sign for ascending crossings and silently
// discards every disk hit from below the equator
float f = prev_cos / max(prev_cos - cos_th, 1e-9f);
```

Regression: TEST 12 — a camera below the disk plane must see the disk.

### INVARIANT: Kerr-Newman charge enters EVERY metric coefficient

`2Mr → r − Q²` (rs units) in `g_tt`, `g_tφ`, `g_φφ` of the camera tetrad AND
the disk emitter — not only in `Δ`. A tetrad that is charge-aware in `g_rr`
(via Δ) but Kerr-only elsewhere is not orthonormal in the metric the ray is
propagated in, and biases `L_z`/`Q_C` per pixel.

Regression: TESTs 1, 6, 10 (KN horizons, RN photon sphere/capture, KN Ω).

---

## 2. Redshift & Beaming

### INVARIANT: The emitter g-factor uses the photon impact ratio ξ = L/E = −L

With the past-directed congruence the traced conserved `L` (for `E = −1`)
relates to the photon's impact ratio as `ξ = −L`:

```metal
kerr.L_arr = -kerr.L;                                   // xi = L/E
g = 1.0f / max(Ut * (1.0f - Omega * kerr.L_arr), 1e-3f);
```

Using the raw traced `L` here inverts the Doppler asymmetry — the approaching
limb dims and the receding limb brightens, which is wrong in the most
recognizable feature of every published black hole image.

Regression: TEST 7 — end-to-end first-principles check that the approaching
limb has g > 1.

### INVARIANT: Doppler color is a temperature shift, not an RGB tint

A shifted blackbody is exactly another blackbody at `T_obs = g·T_emit`. Color
must come from the Planck-locus LUT at `T_obs`; intensity from `Fn·g⁴`
(bolometric Liouville), capped at 15 to keep the approaching inner limb inside
tonemap range. An RGB lerp toward "hot"/"cool" colors cannot move along the
Planckian locus and reads as tinting, not physics.

### INVARIANT: No emission from spacelike orbits

If `U_denom = −g_tt − 2g_tφΩ − g_φφΩ² ≤ 0` the circular orbit is spacelike
(inside the sense-appropriate photon orbit) — return zero emission. Clamping
`U_denom` and emitting produces clamp noise, not physics. With the signed-spin
ISCO branches feeding `r_in` this region is normally never sampled; the guard
is the backstop.

---

## 3. Retrograde Spin & Volumetric Accumulation

### INVARIANT: The ISCO branch must match the disk's rotation sense

The disk always rotates in +φ. For `spin < 0` it is retrograde relative to the
hole: `kerr_radii()` must select the **retrograde** BPT branch (`+root`), which
is up to 7× larger than the prograde one (9M vs 1M at extremal spin). Using
`|a|` with the prograde branch anchors the disk deep inside the region where
counter-rotating circular orbits do not exist.

Regression: TEST 2 (retrograde values, exact 9M limit).

### INVARIANT: Volumetric emission is weighted by the affine step

Glow, ergosphere shimmer, and jet contributions accumulate as
`emissivity · vol_w` with `vol_w = dlam / (0.06 · max(r, 1))`, and attenuate
`trans` likewise. Per-STEP accumulation under the adaptive controller makes
brightness proportional to local step density — photon-shell whirl rays take
hundreds of tiny steps and glow ~2× brighter every time the accuracy constant
is halved. The `0.06·r` divisor reproduces the historical calibration of the
old fixed-step policy, so brightness is step-invariant without a retune.

### INVARIANT: Capture keeps foreground emission

On horizon capture: `trans = 0` (the fate gate keeps the background out) but
`col_accum` is KEPT — jet and glow light accumulated between the camera and
the horizon is real foreground emission. Zeroing it notches the jet where it
crosses the shadow. (Safe only together with the step-weighted accumulation
above; per-step glow would flood the shadow.)

### Ray fates

`0` budget-exhausted (falls back to the undeflected direction), `1` captured,
`2` escaped, `3` disk hit, `4` absorbed in foreground media (jet/glow at
`trans < 0.005`). The lenses color 4 distinctly — conflating it with the sky
would falsely label the optically-thick jet cone in the image-order map.

---

## 4. Temporal Pipeline

### INVARIANT: Auto-exposure reads the slot the semaphore guarantees complete

```objc
// CORRECT — this slot was last written by GPU frame N-3 (semaphore depth 3)
uint32_t* lumData = (uint32_t*)lumBuffer[currentFrame].contents;  // read BEFORE memset
// WRONG — slot (currentFrame+1)%3 belongs to frame N-2, which may not have
// executed yet when the GPU runs 2+ frames behind: the CPU reads only its own
// memset zeros and exposure never converges
```

### INVARIANT: Metering excludes sub-floor pixels

Pixels with luminance < 0.001 (empty sky) are **excluded** from the log-average
— not floored. Flooring them drags the mean so low that exposure pegs at its
clamp and the disk clips to white.

### INVARIANT: Accumulation must sanitize its history

Freshly created private textures have undefined contents; a NaN there persists
through `mix()` forever. `temporal_accum` replaces non-finite history with the
current frame before blending.

### INVARIANT: Bounded history while ANY raytraced content animates

Companion stars sweep pixels per frame; jet turbulence, star twinkle, and
nebula drift all consume `sys.time` inside the accumulated raytrace pass. A
deep accumulator freezes them into a temporal mean, so `accum_alpha` is
floored at 0.1 whenever any of them is animating.

### INVARIANT: Never blend into invalid history; no jitter without history

After `createResources()` the accumulation textures hold undefined memory:
the motion-blur alpha override is gated on `accumHistoryValid`, which resize
clears. And when history is being REPLACED (`alpha ≈ 1`, camera moving), the
subpixel jitter is zeroed — uncompensated jitter just makes high-contrast
edges crawl during interaction.

---

## 4b. Volumetric Transport & EDR

### INVARIANT: Volumetric emission uses the INVARIANT transfer equation

`d(I_ν/ν³)/dλ = j_ν/ν² − (ν α_ν)(I_ν/ν³)`, with fluid-frame quantities at
`ν' = 1/g`. For `j_ν ∝ n² ν^-α_s` this gives `J_inv = n² g^(α_s+2)` and
`A_inv = A n² g^(α_s+1.5)`. Do NOT apply a Doppler boost on top — the g
dependence is already carried by the invariant emissivity, and double-counting
destroys the cancellation identity below.

Regression: TEST 18 — at `α_s = −2` the exponent vanishes, so mirror-image
sight lines through a relativistically rotating flow must give IDENTICAL
intensity (verified to 1e-15), while `α_s = 0` on the same geometry must stay
asymmetric (63%). This single test covers the four-velocity construction from
`l(R)`, the redshift factor, and the g-power weighting at once.

### INVARIANT: EDR tonemapping expands highlights, never retargets the curve

`H · ACES(x/H)` looks like the natural way to move the shoulder to the display
headroom, but it drags the midtones down with it — middle gray falls from
0.267 to 0.071 at H = 16. Keep the ACES result bit-for-bit below the knee and
add a logarithmic lift driven by the LINEAR signal above it. At `H = 1` the
operator must reduce exactly to the SDR curve.

### INVARIANT: everything written to the drawable must be scene-linear

The layer is `RGBA16Float` in extended-linear space, so the compositor applies
the transfer function. Display-authored colors — the false-color lenses, the
grid, ImGui's palette and every overlay `IM_COL32` — must be converted with
`srgb_to_linear` / `linCol()` first, or the UI washes out.

---

## 5. Post-Processing

- **Exposure first.** Bloom extraction, flare thresholding, and the grid's
  scene-occlusion test all operate on *exposed* values so their thresholds mean
  the same thing at any auto-exposure state.
- **Bloom threshold along luminance**, scaling the color, never per-channel
  `max()` — per-channel clipping shifts bloom hue.
- **Film grain after the tonemap**, scaled by √luminance. Grain added in linear
  HDR before ACES is half-wave rectified on black sky (lifts blacks).
- **False-color lenses bypass the photographic chain** (exposure, bloom, ACES);
  they are display-referred data visualizations and get gamma only.
- **The CAMetalLayer colorspace is set to sRGB** to match the manual gamma
  encode; without it, wide-gamut displays oversaturate everything.

---

## 6. Grid Rendering

### INVARIANT: Flat grid lines (depth < 0.008) use discard_fragment()

Alpha-blending hundreds of near-flat lines forms an opaque band
(`(1-0.03)^400 ≈ 0` transmission).

### INVARIANT: Scene occlusion tests EXPOSED luminance

```metal
if (scene_lum * sys.exposure > 0.05f) discard_fragment();
```

The scene texture is pre-exposure HDR; a fixed threshold on raw values makes
grid occlusion drift with auto-exposure.

---

## 7. Compiler & Precision

### INVARIANT: One precision policy on BOTH shader compile paths — relaxed, never fast

- Runtime path: `MTLMathModeRelaxed` (macOS 15+); pre-15 falls back to
  `fastMathEnabled = NO`, which is strictly SAFER (the deprecated API has no
  relaxed setting).
- Offline path: `scripts/build_metallib.sh` passes `-fmetal-math-mode=relaxed`
  (or `-fno-fast-math`, strictly safer, on older toolchains).

Be precise about what relaxed means: it PERMITS non-IEEE sqrt/division while
preserving Inf/NaN — it is not "IEEE-compliant fast math" (no Metal mode is).
It is a deliberate 3× performance trade-off: the fp64 mirror suite certifies
the algorithm, and A/B captures against `MTLMathModeSafe` show sub-1% mean
pixel differences. Full fast math (`MTLMathModeFast`) remains FORBIDDEN — it
additionally flushes NaN and reassociates aggressively, and historically made
the disk vanish at edge-on angles. If the metallib toolchain disappears, the
build script deletes any stale metallib so the app cannot silently prefer an
outdated binary over freshly edited source.

### INVARIANT: ARC on the Objective-C++ sources

`src/main.mm` and `imgui_impl_metal.mm` are written ARC-style. Compiled without
`-fobjc-arc` every implied release is a no-op: ~100 MB leaked per window
resize, plus per-frame ImGui transients. The flag is per-source in
CMakeLists.txt.

### INVARIANT: Shader edits must be dependency-tracked

The metallib + fallback-copy commands use `add_custom_command(OUTPUT … DEPENDS
shaders/geodesic.metal include/ShaderCommon.h …)`. POST_BUILD commands only run
on relink, which silently serves stale shaders after `make`.

---

## 8. Diagnostic Tools

- **P key**: capture the framebuffer (PPM, `O_CREAT|O_EXCL|O_NOFOLLOW`).
- **QA harness**: `BH_QA=1` renders 100 frames with input ignored, captures
  frame 90, exits at 100. `BH_ELEV/BH_AZIM/BH_SPIN/BH_CHARGE/BH_LENS/
  BH_BEAMING/BH_OVERLAYS/BH_SHOT_DIR` select the scenario — the basis for
  reproducible visual regression captures.
- **PPM → PNG**: `sips -s format png file.ppm --out file.png`
- **Physics suite**: `python3 tests/validate_physics.py` (83 tests, fp64 mirror
  of the shader integrator — including a leg at the exact shipped GPU step
  constants and a through-the-pole continuation test).
- **Critical-curve overlay**: doubles as a live validation — the rendered
  shadow edge must land on Bardeen's analytic curve at any spin/charge/
  inclination.

---

## Quick Reference: What NOT to Change

| Guard | Why |
|-------|-----|
| `E = −1` past-directed congruence (`α − ω√g_φφ n_φ` denominator) | Kerr frame-dragging handedness |
| `vol_w = dlam/(0.06·max(r,1))` on glow/ergo/jet | step-density-invariant volumetrics |
| `fabs(d.pth)/(fabs(s.pth)+1)` in the step controller | resolves stiff polar turning points |
| Polar-cap φ freeze + π jump | BL axis singularity |
| `clamp(prev_cos/(prev_cos - cos_th), 0, 1)` | below-plane disk visibility |
| `kerr.L_arr` in the g-factor (never `kerr.L`) | Doppler asymmetry sign |
| `U_denom <= 1e-6 → no emission` | spacelike-orbit guard |
| Signed-spin ISCO/photon branches in `kerr_radii()` | retrograde disks |
| `lumBuffer[currentFrame]` read before memset | exposure convergence |
| `min(g⁴, 15)` | HDR cap for tonemapping |
| Fast-math-off on both compile paths | geodesic precision |
| `-fobjc-arc` on the .mm sources | memory leaks |
