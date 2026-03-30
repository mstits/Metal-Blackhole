# Metal Blackhole

A high-fidelity, real-time black hole simulation optimized for Apple Silicon via the Metal API. This project implements general relativity geodesics, volumetric plasma physics, and cinematic post-processing to create a physically authentic and visually stunning representation of a Kerr-Newman black hole.
<img width="1312" height="940" alt="blackhole_screenshot" src="https://github.com/user-attachments/assets/98ee9e2e-913c-41ba-a067-f5cb44b1712f" />

## Technical Highlights

### Core Physics & Metrics
- **Kerr-Newman Metric:** Support for mass, angular momentum (Spin 'a'), and electric charge (Charge 'Q').
- **RK4 Geodesic Solver:** 4th-Order Runge-Kutta integration for high-precision light-ray curvature.
- **Dimensionless Units:** Refactored mathematics ($r/rs$) to maintain numerical stability at any scale.
- **N-Body Gravity:** GPU-accelerated Newtonian solver for orbiting companion stars.

### Rendering & Optics
- **Volumetric Accretion Torus:** 5-octave spatiotemporal fBM noise simulating turbulent plasma.
- **Relativistic Effects:** Accurate Doppler beaming ($1/D^4$) and gravitational redshift.
- **Volumetric Self-Shadowing:** Secondary ray-marching for realistic internal occlusion.
- **Star Lensing:** Dynamic lensing and smearing of companion stars through curved spacetime.
- **Gravitational Waves:** Quadrupole ripple visualization on the spacetime manifold grid.

### Cinematic Suite
- **ACES Filmic Tonemapping:** Hollywood-standard color science.
- **Anamorphic Lens Flare:** Procedural flares generated from the high-intensity core.
- **Motion Blur:** Temporal feedback loops for shutter-accurate trails.
- **Film Science:** 70mm film grain and optical vignetting.

### Performance (Apple Silicon Optimized)
- **Triple Buffering:** Zero CPU-GPU synchronization stalls.
- **MTLMathModeRelaxed:** FMA-enabled compilation with IEEE-compliant sqrt/division for geodesic accuracy.
- **SIMD-Aligned Threadgroups:** 32×8 threadgroups aligned with Apple Silicon's 32-wide execution width.
- **Non-Uniform Dispatch:** `dispatchThreads` for hardware-managed boundary handling.
- **Adaptive Integration:** Euler for weak-field (r > 8), RK4 for strong-field — 1 vs 4 force evaluations per step.
- **Half-Precision (FP16):** Disk color computation at 2× ALU throughput.
- **Double-Buffered Intermediates:** Eliminates per-frame blit copy for motion blur accumulation.
- **SIMD Group Voting:** Low-level culling of unnecessary calculations.

## Controls
- **Left Click + Drag:** Rotate Camera
- **Shift + Left Click + Drag:** Pan Target
- **Scroll:** Zoom (In/Out)
- **UI Sliders:** Real-time control of physics, disk density, shadows, and optics.

## Building
Requires `cmake`, `glfw`, and `glm`.
```bash
mkdir build && cd build
cmake ..
make
./MetalBlackhole
```

## Security Considerations

### Runtime Shader Compilation (GPU Code Injection Risk)

Shader source files (`geodesic.metal`, `ShaderCommon.h`) are loaded from the filesystem at runtime and compiled into GPU code via `newLibraryWithSource:`. If an attacker can write to the build/executable directory, they could inject arbitrary GPU compute code.

**Recommended Mitigation:** With full Xcode installed, precompile shaders to a signed `.metallib` binary:

```bash
xcrun -sdk macosx metal -c -I include/ -std=metal3.0 shaders/geodesic.metal -o build/geodesic.air
xcrun -sdk macosx metallib build/geodesic.air -o build/geodesic.metallib
```

Then load via `[device newLibraryWithURL:...]` instead of runtime source compilation. **Do NOT use `-ffast-math`** — it breaks geodesic accuracy at edge-on camera angles.

### Input Validation

All GPU uniform values are clamped to valid ranges at the CPU→GPU write site to prevent NaN propagation, division-by-zero, or excessive loop iteration from out-of-range parameters.

### Compiler Hardening

The build enables `-Wall -Wextra -Wformat-security -fstack-protector-strong`.

## Credits
Built by [mstits](https://github.com/mstits).
