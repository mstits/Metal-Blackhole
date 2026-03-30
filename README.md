# Metal Blackhole

A high-fidelity, real-time black hole simulation optimized for Apple Silicon via the Metal API. This project implements general relativity geodesics, volumetric plasma physics, and cinematic post-processing to create a physically authentic and visually stunning representation of a Kerr-Newman black hole.
<img width="1770" height="1263" alt="Metal-BlackHole_Screen_Shot" src="https://github.com/user-attachments/assets/98787b99-b1b8-4cc2-af2b-103536b6ee38" />

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

### Performance
- **Triple Buffering:** Zero CPU-GPU synchronization stalls.
- **Half-Precision (FP16):** Optimized kernels for doubled arithmetic throughput.
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

## Credits
Built by [mstits](https://github.com/mstits).
