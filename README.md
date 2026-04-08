# Metal Blackhole

A high-fidelity, real-time black hole simulation optimized for Apple Silicon via the Metal API. This project implements general relativity geodesics, volumetric plasma physics, and cinematic post-processing to create a physically authentic and visually stunning representation of a Kerr-Newman black hole.
<img width="1312" height="940" alt="blackhole_screenshot" src="https://github.com/user-attachments/assets/98ee9e2e-913c-41ba-a067-f5cb44b1712f" />

---

## Table of Contents

- [Technical Highlights](#technical-highlights)
- [Architecture](#architecture)
- [GPU Rendering Pipeline](#gpu-rendering-pipeline)
- [Physics Model](#physics-model)
- [Project Structure](#project-structure)
- [Controls](#controls)
- [Building](#building)
- [Presets](#presets)
- [Validation](#validation)
- [Security Considerations](#security-considerations)
- [Credits](#credits)

---

## Technical Highlights

### Core Physics & Metrics
- **Kerr-Newman Metric:** Support for mass, angular momentum (Spin `a`), and electric charge (Charge `Q`).
- **RK4 Geodesic Solver:** 4th-Order Runge-Kutta integration of the effective gravitational potential with gravitomagnetic (Lense-Thirring) frame dragging — the same formulation used by DNEG for *Interstellar*.
- **Dimensionless Units:** Refactored mathematics (`r/rs`) to maintain numerical stability at any scale.
- **N-Body Gravity:** GPU-accelerated Newtonian solver for orbiting companion stars.
- **Spectral Doppler Shift:** Color temperature shifts from orbital velocity — approaching limb blue-shifts, receding limb red-shifts.

### Rendering & Optics
- **Volumetric Accretion Torus:** 5-octave spatiotemporal fBM noise simulating turbulent plasma.
- **Relativistic Effects:** Accurate Doppler beaming (`D⁴`) and gravitational redshift.
- **Novikov-Thorne Disk:** Temperature profile with zero-torque ISCO boundary condition.
- **Volumetric Self-Shadowing:** Secondary ray-marching for realistic internal occlusion.
- **Star Lensing:** Dynamic lensing and smearing of companion stars through curved spacetime.
- **Photon Rings:** Primary, secondary (half-orbit), and higher-order gravitationally focused images.
- **Gravitational Waves:** Quadrupole ripple visualization on the spacetime manifold grid.
- **Polar Relativistic Jets:** Collimated blue-white emission along the spin axis with turbulent noise.
- **Ergosphere Glow:** Faint violet emission inside the static limit surface for spinning black holes.

### Cinematic Suite
- **ACES Filmic Tonemapping:** Hollywood-standard color science.
- **MPS Bloom:** MetalPerformanceShaders Gaussian blur for cinematic glow around bright regions.
- **Anamorphic Lens Flare:** Procedural horizontal streaks from high-intensity sources.
- **Auto-Exposure:** GPU-computed log-luminance histogram with temporal smoothing.
- **Optical Vignette:** Edge darkening for natural lens falloff.
- **Motion Blur:** Temporal feedback loops for shutter-accurate trails.
- **Film Grain:** 70mm grain simulation.

### Performance (Apple Silicon Optimized)
- **Triple Buffering:** Zero CPU-GPU synchronization stalls via `dispatch_semaphore`.
- **Precompiled `.metallib`:** Offline shader compilation for instant startup (graceful fallback to runtime compilation).
- **MTLMathModeRelaxed:** FMA-enabled compilation with IEEE-compliant sqrt/division for geodesic accuracy.
- **SIMD-Aligned Threadgroups:** 32×8 threadgroups aligned with Apple Silicon's 32-wide execution width.
- **Non-Uniform Dispatch:** `dispatchThreads` for hardware-managed boundary handling.
- **Adaptive Integration:** Euler for weak-field (`r > 8`), RK4 for strong-field — 1 vs 4 force evaluations per step.
- **Half-Precision (FP16):** Disk color computation at 2× ALU throughput.
- **Double-Buffered Intermediates:** Eliminates per-frame blit copy for motion blur accumulation.

---

## Architecture

```mermaid
graph TD
    subgraph Host ["CPU Host (main.mm)"]
        GLFW["GLFW Window + Input"]
        CAM["Camera (Orbital)"]
        IMGUI["ImGui Control Panel"]
        UNI["Uniform Upload<br/>(Triple-Buffered)"]
    end

    subgraph GPU ["Metal GPU Pipeline"]
        FLUID["Fluid Simulation<br/>(Advection Kernel)"]
        PHYS["N-Body Physics<br/>(Newtonian Gravity)"]
        RAY["Geodesic Raytracer<br/>(RK4 + Euler)"]
        BLOOM_EX["Bloom Extraction<br/>(Half-Res)"]
        MPS["MPS Gaussian Blur"]
        LUM["Luminance Reduction<br/>(Auto-Exposure)"]
        POST["Post-Processing Suite<br/>(ACES + Bloom + Flare)"]
        GRID["Grid Renderer<br/>(Spacetime Manifold)"]
    end

    subgraph Output ["Display"]
        DRAW["CAMetalLayer Drawable"]
    end

    GLFW --> CAM
    GLFW --> IMGUI
    CAM --> UNI
    IMGUI --> UNI
    UNI --> FLUID
    UNI --> PHYS
    UNI --> RAY
    FLUID --> RAY
    PHYS --> RAY
    RAY --> BLOOM_EX
    RAY --> LUM
    BLOOM_EX --> MPS
    MPS --> POST
    LUM --> POST
    RAY --> POST
    POST --> DRAW
    GRID --> DRAW
    IMGUI --> DRAW

    style Host fill:#1a1a2e,stroke:#e94560,color:#eee
    style GPU fill:#0f3460,stroke:#00d2ff,color:#eee
    style Output fill:#16213e,stroke:#0f3460,color:#eee
```

---

## GPU Rendering Pipeline

Each frame dispatches the following compute and render passes in order:

```mermaid
flowchart LR
    A["1. Fluid Sim<br/>1024×1024<br/>Disk turbulence"] --> B["2. N-Body Physics<br/>per-object<br/>Gravity solver"]
    B --> C["3. Geodesic Raytrace<br/>Full resolution<br/>1200 steps/ray"]
    C --> D["4. Bloom Extract<br/>Half resolution<br/>Threshold filter"]
    D --> E["5. MPS Blur<br/>σ = 8.0<br/>Gaussian bloom"]
    C --> F["6. Luminance<br/>Every 4th pixel<br/>Log₂ accumulate"]
    E --> G["7. Post-Process<br/>Bloom + Flare<br/>ACES Tonemap"]
    F --> G
    C --> G
    G --> H["8. Grid Render<br/>120×120 mesh<br/>Gravity well"]
    H --> I["9. ImGui<br/>Control panel<br/>Physics readouts"]
    I --> J(("Present<br/>Drawable"))

    style A fill:#2d1b69,stroke:#8b5cf6,color:#fff
    style B fill:#1e3a5f,stroke:#3b82f6,color:#fff
    style C fill:#7c2d12,stroke:#f97316,color:#fff
    style D fill:#4a1942,stroke:#c084fc,color:#fff
    style E fill:#4a1942,stroke:#c084fc,color:#fff
    style F fill:#1a4731,stroke:#22c55e,color:#fff
    style G fill:#78350f,stroke:#f59e0b,color:#fff
    style H fill:#1e3a5f,stroke:#3b82f6,color:#fff
    style I fill:#374151,stroke:#9ca3af,color:#fff
    style J fill:#064e3b,stroke:#10b981,color:#fff
```

### Raytracer Detail (Step 3)

The core raytracer integrates photon trajectories backward from the camera through curved spacetime:

```mermaid
flowchart TD
    START["Camera Ray Origin<br/>ro = camPos, rd = pixel direction"] --> INIT["Initialize in BH coordinates<br/>pos = (ro - bhPos) / rs<br/>vel = rd"]
    INIT --> LOOP{"Integration Loop<br/>i < 1200"}

    LOOP -->|"r < r_horizon"| HORIZON["Event Horizon Hit<br/>trans = 0, col = black<br/>BREAK"]
    LOOP -->|"r > 500 or<br/>r > 30 & outbound"| EXIT["Exit: Sample Background<br/>Stars + Nebula via<br/>deflected exit vel"]
    LOOP -->|Continue| DT["Adaptive Step Size<br/>dt = max(r × 0.06, 0.001)<br/>+ disk proximity clamp"]

    DT --> INTEGRATE{"r > 8?"}
    INTEGRATE -->|"Yes (weak field)"| EULER["Euler Step<br/>1 force eval"]
    INTEGRATE -->|"No (strong field)"| RK4["RK4 Step<br/>4 force evals"]

    EULER --> DISK{"In Disk Slab?<br/>|y| < disk_h<br/>r_in < rh < r_out"}
    RK4 --> DISK

    DISK -->|Yes| EMIT["Accumulate Emission<br/>• NT Temperature T(r)<br/>• Doppler beaming D⁴<br/>• Gravitational redshift<br/>• Foreshortening<br/>• Self-shadowing<br/>• Photon ring boost"]
    DISK -->|No| GLOW{"r < 5?"}

    EMIT --> LOOP
    GLOW -->|Yes| ERGO["BH Glow + Ergosphere"] --> LOOP
    GLOW -->|No| JET{"In Jet Cone?<br/>cos θ > 0.92"} -->|Yes| JETEMIT["Jet Emission"] --> LOOP
    JET -->|No| LOOP

    HORIZON --> OUT["Output float4(col, 1)"]
    EXIT --> OUT

    style HORIZON fill:#450a0a,stroke:#ef4444,color:#fff
    style EXIT fill:#052e16,stroke:#22c55e,color:#fff
    style RK4 fill:#7c2d12,stroke:#f97316,color:#fff
    style EULER fill:#1e3a5f,stroke:#3b82f6,color:#fff
    style EMIT fill:#78350f,stroke:#f59e0b,color:#fff
```

---

## Physics Model

### Geodesic Equation

The engine uses an effective gravitational potential with a gravitomagnetic perturbation to simulate photon propagation through Kerr spacetime:

```mermaid
graph LR
    subgraph Schwarzschild ["Schwarzschild Term"]
        S["a⃗ = −1.5 |h⃗|² p⃗ / r⁵"]
    end
    subgraph Charge ["Reissner-Nordström Term"]
        Q["a⃗ += Q² p⃗ / r⁴"]
    end
    subgraph FrameDrag ["Frame Dragging (Lense-Thirring)"]
        FD["J⃗ = (0, a/2, 0)<br/>B⃗ = (3p⃗(p⃗·J⃗)/r² − J⃗) / r³<br/>a⃗ += 4(v⃗ × B⃗)"]
    end
    S --> ACC["Total Acceleration"]
    Q --> ACC
    FD --> ACC
    ACC --> INT["RK4 / Euler<br/>Integration"]

    style Schwarzschild fill:#1a1a2e,stroke:#e94560,color:#eee
    style Charge fill:#16213e,stroke:#00d2ff,color:#eee
    style FrameDrag fill:#0f3460,stroke:#8b5cf6,color:#eee
    style ACC fill:#78350f,stroke:#f59e0b,color:#fff
    style INT fill:#064e3b,stroke:#10b981,color:#fff
```

### Accretion Disk Model

| Property | Formula | Source |
|----------|---------|--------|
| Inner Edge | `r_in = max(r_isco, r_horizon × 1.2)` | Bardeen (1972) |
| Emissivity | `F ∝ (r_in/r)³` | Shakura-Sunyaev |
| Temperature | `T ∝ r^(-3/4) × (1 − √(r_in/r))^(1/4)` | Novikov-Thorne |
| Orbital Velocity | `v = √(1/r) + ω_fd × r` | Keplerian + ZAMO |
| ZAMO Frequency | `ω = a / (r³ + a²r + a²)` | Exact Kerr |
| Doppler Beaming | `D⁴ = 1 / (1 − v⃗·r̂)⁴`, capped at 15× | Relativistic invariant |
| Gravitational Redshift | `g = √(1 − 3/(2r) + a/r^(3/2))` | Kerr circular orbit |

### Spin-Dependent GR Parameters

| Feature | Schwarzschild (a=0) | Kerr (a=0.9) | Extreme Kerr (a→1) |
|---------|---------------------|--------------|---------------------|
| Event Horizon (r₊) | 1.000 rs | 0.718 rs | 0.500 rs |
| ISCO (prograde) | 3.000 rs | 1.160 rs | 0.500 rs |
| Photon Sphere | 1.500 rs | — | — |
| Shadow Radius | 2.598 rs | — | — |
| Ergosphere | None | r < 1.0 rs | r < 1.0 rs |

---

## Project Structure

```
metal_blackhole/
├── src/
│   └── main.mm                  # Application entry, Metal engine, ImGui panel
├── shaders/
│   └── geodesic.metal           # All GPU kernels (raytrace, fluid, post, physics, grid)
├── include/
│   ├── ShaderCommon.h           # Shared CPU/GPU struct definitions
│   └── Camera.h                 # Orbital camera controller
├── scripts/
│   └── build_metallib.sh        # Offline shader precompilation
├── tests/
│   └── validate_physics.py      # 47-test physics validation suite
├── libs/
│   └── imgui/                   # Dear ImGui (vendored)
├── RENDERING_INVARIANTS.md      # Critical shader invariants & lessons learned
├── CMakeLists.txt               # Build configuration
└── README.md                    # This file
```

### Source File Map

```mermaid
graph TD
    subgraph CPU ["Host (C++ / Obj-C++)"]
        MAIN["main.mm<br/>━━━━━━━━━━━━━━━━━<br/>MetalEngine class<br/>GridRenderer class<br/>Camera input handling<br/>ImGui control panel<br/>N-body scene setup<br/>Triple-buffer management"]
        CAM_H["Camera.h<br/>━━━━━━━━━━━━━━━━━<br/>Orbital camera<br/>View/proj matrices"]
    end

    subgraph Shared ["Shared (CPU ↔ GPU)"]
        COMMON["ShaderCommon.h<br/>━━━━━━━━━━━━━━━━━<br/>SimObject struct<br/>CameraData struct<br/>SystemUniforms struct<br/>ObjectsUniform struct<br/>GridUniforms struct"]
    end

    subgraph GPU_K ["GPU Kernels (Metal)"]
        GEO["geodesic.metal<br/>━━━━━━━━━━━━━━━━━<br/>raytrace — geodesic ray march<br/>simulate_disk_fluid — advection<br/>update_physics — N-body gravity<br/>bloom_extract — threshold filter<br/>luminance_reduce — auto-exposure<br/>post_process_suite — ACES + bloom<br/>grid_vertex / grid_fragment"]
    end

    MAIN --> COMMON
    CAM_H --> MAIN
    COMMON --> GEO

    style CPU fill:#1a1a2e,stroke:#e94560,color:#eee
    style Shared fill:#16213e,stroke:#f59e0b,color:#eee
    style GPU_K fill:#0f3460,stroke:#00d2ff,color:#eee
```

---

## Controls

| Input | Action |
|-------|--------|
| **Left Click + Drag** | Rotate camera orbit |
| **Shift + Left Click + Drag** | Pan camera target |
| **Scroll Wheel** | Zoom in / out |
| **P** | Capture screenshot (PPM to `/tmp/`) |
| **Escape** | Quit |
| **ImGui Sliders** | Real-time control of physics, disk, shadows, and optics |

---

## Building

### Requirements
- macOS with Apple Silicon (M1/M2/M3/M4)
- `cmake` ≥ 3.10
- `glfw` and `glm` (via Homebrew or vcpkg)
- Xcode (for precompiled `.metallib`; optional — falls back to runtime compilation)

### Build & Run
```bash
# Install dependencies (if using Homebrew)
brew install cmake glfw glm

# Build
mkdir build && cd build
cmake ..
make

# Run
./MetalBlackhole
```

### Shader Compilation

Shaders are automatically precompiled to a `.metallib` binary at build time via `scripts/build_metallib.sh`. This requires a full Xcode installation.

If the Metal compiler is unavailable, the build gracefully falls back to copying source files for runtime compilation with a console warning on startup.

---

## Presets

The ImGui panel includes one-click presets for common spacetime geometries:

| Preset | Spin (a) | Charge (Q) | Features |
|--------|----------|------------|----------|
| **Schwarzschild** | 0.0 | 0.0 | Pure GR — no spin, no charge |
| **Kerr** | 0.7 | 0.0 | Spinning BH with frame dragging |
| **Extreme Kerr** | 0.998 | 0.0 | Near-maximal spin + relativistic jets |
| **Charged (RN)** | 0.0 | 0.5 | Reissner-Nordström geometry |
| **Kerr-Newman** | 0.6 | 0.3 | Full KN metric + jets |
| **Cinematic** | 0.85 | 0.0 | All visual effects enabled |

---

## Validation

The project includes a physics validation test suite that mathematically proves correctness against known analytical GR solutions:

```bash
python3 tests/validate_physics.py
```

### Test Coverage (47 tests)

| Test | What It Validates |
|------|-------------------|
| Event Horizon | Kerr `r₊ = (1 + √(1−a²))/2` for a ∈ {0, 0.5, 0.9, 0.998, 0.9999} |
| ISCO | Bardeen-Press-Teukolsky formula across spin range |
| Photon Sphere | Circular null orbit stability at r = 1.5 rs (Schwarzschild) |
| Shadow Radius | Critical impact parameter b = √27/2 ≈ 2.598 rs |
| NT Temperature | Zero-torque boundary condition T → 0 at ISCO |
| Gravitational Redshift | `g = √(1 − 3/(2r))` for circular orbits |
| ZAMO Frequency | Exact Kerr `ω = a/(r³ + a²r + a²)` across 12 (a, r) pairs |
| Polar Doppler | Zero beaming from top-down view (8 azimuthal samples) |
| Equatorial Doppler | Left-right asymmetry with opposite signs |
| Horizon Capture | Head-on and sub-critical rays absorbed, zero light bleed |

---

## Security Considerations

### Precompiled Shader Library

Shaders are precompiled to a signed `.metallib` binary at build time. This eliminates the GPU code injection risk of runtime `newLibraryWithSource:` compilation.

### Input Validation

All GPU uniform values are clamped to valid ranges at the CPU→GPU write site to prevent NaN propagation, division-by-zero, or excessive loop iteration from out-of-range parameters.

### Compiler Hardening

The build enables `-Wall -Wextra -Wformat-security -fstack-protector-strong`.

---

## Credits

Built by [mstits](https://github.com/mstits).
