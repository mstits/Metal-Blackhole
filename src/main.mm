#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <QuartzCore/QuartzCore.h>
#import <ImageIO/ImageIO.h>
#import <CoreGraphics/CoreGraphics.h>
#define EXR_PARALLEL
#include "exr_out.h"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_metal.h"

// Metal clips z to [0, w]; GLM defaults to OpenGL's [-w, w].
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <unistd.h>
#include <vector>

using namespace glm;
using namespace std;

// --- SHARED STRUCTS (single source of truth in ShaderCommon.h) ---
#include "ShaderCommon.h"

const int WINDOW_WIDTH = 1200;
const int WINDOW_HEIGHT = 800;

const double G_const = 6.67430e-11;
const double c_const = 299792458.0;
// Single source of truth for the BH mass: the rendered Schwarzschild radius,
// the N-body gravity, and the initial circular velocities all derive from it.
const double SagA_mass = 8.54e36;             // Sgr A* (kg)
const double SagA_rs = 2.0 * G_const * SagA_mass / (c_const * c_const);

const float PI_F = 3.14159265358979323846f;

// --- USER-FACING SETTINGS ---
// Defaults boot the app in "physics-first" mode: Schwarzschild geometry,
// cinematic effects off, learning overlays off.
struct SimSettings {
  bool  Gravity            = true;
  float blackHoleSpin      = 0.0f;
  float blackHoleCharge    = 0.0f;
  float timeScale          = 1.0f;
  float starScintillation  = 0.5f;
  float nebulaIntensity    = 1.0f;
  float bloomThreshold     = 0.85f;
  float flareIntensity     = 0.5f;
  float motionBlur         = 0.85f;
  float filmGrain          = 0.02f;
  float diskDensity        = 0.65f;
  float diskOuterRadius    = 12.0f;     // rs units
  float diskTempK          = 5500.0f;   // Novikov-Thorne peak temperature
  float gwAmplitude        = 0.5f;
  float jetIntensity       = 0.7f;
  float photonRingBoost    = 0.0f;      // 0 = physical; >0 amplifies n>=1 images
  float vignetteIntensity  = 0.6f;
  float ehtBeamFwhm        = 2.6f;      // telescope beam FWHM in rs units

  int  bhMode          = BH_KERR_NEWMAN;
  int  projection      = PROJ_PERSPECTIVE;
  float mpM1           = 0.25f;   // code units (rs = 2 M_total)
  float mpM2           = 0.25f;
  float mpSep          = 6.0f;
  float mpGlow         = 0.5f;
  int  diskModel       = DISK_THIN;
  float torusR0        = 7.0f;     // ring radius (M units)
  float torusH         = 6.0f;
  float torusL0        = 1.0f;
  float torusAlpha     = 0.0f;     // j_nu ~ nu^0; -2 gives exact Doppler cancellation
  float torusAbsorb    = 1.0f;     // moderate self-absorption: the core
                                   // becomes optically thick (tau ~ 2), which
                                   // is what gives a real flow its photosphere
                                   // and keeps the ring from saturating
  float torusTemp      = 6500.0f;
  float torusDensity   = 1.0f;
  bool enEDR           = true;
  int  lensMode        = LENS_STANDARD;
  int  beamingMode     = BEAM_FULL;
  bool enStarBodies    = true;
  bool enCharge        = false;
  bool enDisk          = true;
  bool enGW            = true;
  bool enJets          = false;
  bool enNebula        = false;
  bool enScintillation = false;
  bool enBloom         = false;
  bool enFlare         = false;
  bool enMotionBlur    = false;
  bool enFilmGrain     = false;
  bool enVignette      = false;
  bool enGrid          = false;
  // Learning overlays (stackable, compose with any lens)
  bool ovMarkers       = false;         // horizon/ergosphere/photon/ISCO circles
  bool ovCriticalCurve = false;         // Bardeen analytic shadow boundary
  bool ovGeodesicFan   = false;         // 2D equatorial trajectory inset
};
static SimSettings g_sim;

// --- RUNTIME STATE ---
float simulationTime = 0.0f;
double lastFrameTime = 0.0;
float deltaTime = 0.016f;
int frameCount = 0;
bool screenshotPending = false;
bool exrPending = false;
bool panoPending = false;    // 360 equirectangular panorama
bool stillPending = false;   // high-resolution reference still
// QA harness: BH_QA=1 captures a screenshot at frame 90 and exits at frame 100.
// BH_ELEV / BH_AZIM / BH_SPIN / BH_CHARGE / BH_LENS override initial state.
bool qaMode = false;

#include "Camera.h"
Camera camera((float)(SagA_rs * 20.0));

// Spin- and charge-dependent Kerr-Newman radii in rs units.
//
// Conventions: sliders spin = a/M in [-1,1] and charge = Q/M in [0,1]. The
// disk always rotates in +phi, so for spin < 0 it is RETROGRADE relative to
// the hole and the retrograde ISCO / photon-orbit branches apply — the
// counter-rotating flow's innermost stable orbit is far larger than the
// prograde one (9M vs 1M at |a| -> 1).
struct KerrRadii { float r_horizon, r_isco, r_photon; };

static KerrRadii kerr_radii(float spin, float charge) {
  float a  = std::abs(spin);
  float a2 = a * a;
  float q2 = charge * charge;

  // Outer horizon r_+/rs = ½(1 + √(1 − a² − q²)) — depends on spin AND charge.
  float r_h = (1.0f + std::sqrt(std::max(1.0f - a2 - q2, 0.0f))) * 0.5f;

  // Bardeen-Press-Teukolsky ISCO, branch chosen by the disk's rotation sense
  // relative to the hole (Kerr-only; floored by 1.2 r_+ as a charge guard).
  float cbrt_1ma2 = std::pow(std::max(1.0f - a2, 1e-6f), 1.0f / 3.0f);
  float Z1 = 1.0f + cbrt_1ma2 * (std::pow(1.0f + a, 1.0f / 3.0f)
                               + std::pow(std::max(1.0f - a, 1e-6f), 1.0f / 3.0f));
  float Z2 = std::sqrt(3.0f * a2 + Z1 * Z1);
  float root = std::sqrt(std::max((3.0f - Z1) * (3.0f + Z1 + 2.0f * Z2), 0.0f));
  float r_isco_kerr = (spin >= 0.0f) ? (3.0f + Z2 - root) * 0.5f
                                     : (3.0f + Z2 + root) * 0.5f;
  float r_isco = std::max(r_isco_kerr, r_h * 1.2f);

  // Equatorial photon orbit, same sense convention. For the non-rotating
  // charged case use the closed-form Reissner-Nordström photon sphere
  // r_ph = (3M/2)(1 + sqrt(1 - 8Q²/9M²)) instead of the Kerr formula.
  float r_photon;
  if (a < 0.01f && charge > 0.001f) {
    r_photon = 0.75f * (1.0f + std::sqrt(std::max(1.0f - 8.0f * q2 / 9.0f, 0.0f)));
  } else {
    r_photon = (spin >= 0.0f)
        ? 1.0f + std::cos(2.0f / 3.0f * std::acos(-a))
        : 1.0f + std::cos(2.0f / 3.0f * std::acos(+a));
  }
  return { r_h, r_isco, r_photon };
}

// --- CPU MIRROR OF THE GEODESIC INTEGRATOR (equatorial, for the fan inset) ---
// Same super-Hamiltonian form as the shader, restricted to theta = pi/2.
struct FanRay { std::vector<ImVec2> pts; int fate; };  // fate: 0 escaped, 1 captured

static std::vector<FanRay> compute_geodesic_fan(float spin, float charge) {
  std::vector<FanRay> rays;
  const double a = 0.5 * spin;
  const double Q2 = 0.25 * (double)charge * charge;
  const double r_h = 0.5 * (1.0 + std::sqrt(std::max(1.0 - (double)spin * spin
                                                          - (double)charge * charge, 0.0)));
  const int NR = 33;
  for (int k = 0; k < NR; k++) {
    double b = -8.0 + 16.0 * k / (NR - 1);
    double x0 = -16.0, z0 = b;
    double r = std::sqrt(x0 * x0 + z0 * z0);
    double ph = std::atan2(z0, x0);
    // Direction +x decomposed into the local (r_hat, ph_hat) basis.
    double n_r = std::cos(ph), n_ph = -std::sin(ph);
    double Sigma = r * r, Delta = r * r - r + a * a + Q2;
    double m_kn = r - Q2;
    double gtt_neg = 1.0 - m_kn / Sigma;
    double gtphi = -a * m_kn / Sigma;
    double gphph = ((r * r + a * a) * Sigma + a * a * m_kn) / Sigma;
    double omega_z = -gtphi / gphph;
    double alpha = std::sqrt(std::max(gtt_neg + gtphi * gtphi / gphph, 1e-12));
    double sgp = std::sqrt(gphph);
    double E_inv = 1.0 / (alpha + omega_z * sgp * n_ph);
    double L = sgp * n_ph * E_inv;
    double pr = std::sqrt(Sigma / std::max(Delta, 1e-9)) * n_r * E_inv;

    FanRay ray; ray.fate = 0;
    for (int i = 0; i < 4000; i++) {
      if (r < r_h * 1.02) { ray.fate = 1; break; }
      if (r > 24.0 && pr > 0.0) break;
      // RK4 on (r, ph, pr); dth = 0 in the equatorial plane.
      auto rhs = [&](double rr, double pp, double prr, double* d) {
        double D = rr * rr - rr + a * a + Q2;
        double Dp = 2.0 * rr - 1.0;
        double P = rr * rr + a * a - a * L;
        double B = (L - a) * (L - a);
        double R = P * P - D * B;
        double Rp = 4.0 * rr * P - Dp * B;
        double RoDp = (Rp * D - R * Dp) / std::max(D * D, 1e-12);
        double inv = 1.0 / (rr * rr);
        d[0] = D * prr * inv;
        d[1] = ((a * (rr - Q2) - a * a * L) / std::max(D, 1e-9) + L) * inv;
        d[2] = (RoDp - Dp * prr * prr) * 0.5 * inv;
        (void)pp;
      };
      double d1[3], d2[3], d3[3], d4[3];
      double rate_probe[3]; rhs(r, ph, pr, rate_probe);
      double rate = std::abs(rate_probe[0]) / std::max(r, 0.5) + std::abs(rate_probe[1]) + 1e-9;
      double h = std::min(0.04 / rate, 0.35 * std::max(r - r_h, 1e-3) / std::max(std::abs(rate_probe[0]), 1e-6));
      h = std::max(h, 1e-4);
      rhs(r, ph, pr, d1);
      rhs(r + 0.5 * h * d1[0], ph + 0.5 * h * d1[1], pr + 0.5 * h * d1[2], d2);
      rhs(r + 0.5 * h * d2[0], ph + 0.5 * h * d2[1], pr + 0.5 * h * d2[2], d3);
      rhs(r + h * d3[0], ph + h * d3[1], pr + h * d3[2], d4);
      r  += h * (d1[0] + 2 * d2[0] + 2 * d3[0] + d4[0]) / 6.0;
      ph += h * (d1[1] + 2 * d2[1] + 2 * d3[1] + d4[1]) / 6.0;
      pr += h * (d1[2] + 2 * d2[2] + 2 * d3[2] + d4[2]) / 6.0;
      if ((i & 3) == 0)
        ray.pts.push_back(ImVec2((float)(r * std::cos(ph)), (float)(r * std::sin(ph))));
    }
    rays.push_back(std::move(ray));
  }
  return rays;
}

// Bardeen critical curve (analytic shadow boundary) with the Kerr-Newman
// charge generalization, in M = 1 units; returns (alpha, beta) pairs.
//   xi  = (3r² − r³ − a²(r+1) − 2Q²r) / (a(r−1))
//   eta = r²(−r⁴ + 6r³ − 9r² + 4a²r + 12Q²r − 4Q²r² − 4a²Q² − 4Q⁴) / (a²(r−1)²)
static std::vector<ImVec2> compute_critical_curve(float spin, float charge, float inc) {
  std::vector<ImVec2> upper;
  double a = std::abs(spin), Q = charge;
  inc = std::max(inc, 0.05f);
  double si = std::sin(inc), ci = std::cos(inc);
  if (a < 0.01) {
    // Spherical: radius from the (possibly charged) photon sphere.
    double rp = (Q > 0.001)
        ? 1.5 * (1.0 + std::sqrt(std::max(1.0 - 8.0 * Q * Q / 9.0, 0.0)))
        : 3.0;
    double bc = rp / std::sqrt(std::max(1.0 - 2.0 / rp + Q * Q / (rp * rp), 1e-9));
    for (int k = 0; k <= 128; k++) {
      double t = 2.0 * M_PI * k / 128.0;
      upper.push_back(ImVec2((float)(bc * std::cos(t)), (float)(bc * std::sin(t))));
    }
    return upper;
  }
  double r_h = 1.0 + std::sqrt(std::max(1.0 - a * a - Q * Q, 0.0));
  // The beta^2 >= 0 window in r collapses like O(a) around r = 3 for small
  // spin; a uniform sweep of [r_h, 4.5] then catches only a handful of
  // samples and draws a jagged polygon. Locate the window boundaries first,
  // then distribute all samples inside it.
  auto b2_of = [&](double r) {
    double xi = (3 * r * r - r * r * r - a * a * (r + 1) - 2 * Q * Q * r) / (a * (r - 1));
    double eta = r * r * (-std::pow(r, 4) + 6 * r * r * r - 9 * r * r + 4 * a * a * r
                 + 12 * Q * Q * r - 4 * Q * Q * r * r - 4 * a * a * Q * Q - 4 * std::pow(Q, 4))
                 / (a * a * (r - 1) * (r - 1));
    double si = std::sin(inc), ci = std::cos(inc);
    return eta + a * a * ci * ci - xi * xi * ci * ci / (si * si);
  };
  double r_lo = r_h + 1e-6, r_hi = 4.6;
  // Coarse scan for any valid point, then bisect both window edges.
  double r_seed = -1;
  for (int k = 0; k <= 2000; k++) {
    double r = r_lo + (r_hi - r_lo) * k / 2000.0;
    if (b2_of(r) >= 0) { r_seed = r; break; }
  }
  if (r_seed < 0) return {};
  double lo = r_lo, hi = r_seed;
  for (int it = 0; it < 60; it++) { double m = 0.5 * (lo + hi); (b2_of(m) >= 0 ? hi : lo) = m; }
  double r_min = hi;
  lo = r_seed; hi = r_hi;
  for (int it = 0; it < 60; it++) { double m = 0.5 * (lo + hi); (b2_of(m) >= 0 ? lo : hi) = m; }
  double r_max = lo;
  std::vector<ImVec2> pts;
  for (int k = 0; k <= 800; k++) {
    double r = r_min + (r_max - r_min) * k / 800.0;
    double xi = (3 * r * r - r * r * r - a * a * (r + 1) - 2 * Q * Q * r) / (a * (r - 1));
    double eta = r * r * (-std::pow(r, 4) + 6 * r * r * r - 9 * r * r + 4 * a * a * r
                 + 12 * Q * Q * r - 4 * Q * Q * r * r - 4 * a * a * Q * Q - 4 * std::pow(Q, 4))
                 / (a * a * (r - 1) * (r - 1));
    double b2 = eta + a * a * ci * ci - xi * xi * ci * ci / (si * si);
    if (b2 < 0) continue;
    double al = -xi / si;
    if (spin < 0) al = -al;   // mirrored for retrograde spin
    pts.push_back(ImVec2((float)al, (float)std::sqrt(b2)));
  }
  // Close the curve: upper branch then mirrored lower branch.
  std::vector<ImVec2> curve = pts;
  for (auto it = pts.rbegin(); it != pts.rend(); ++it)
    curve.push_back(ImVec2(it->x, -it->y));
  return curve;
}

class GridRenderer {
public:
  id<MTLRenderPipelineState> pipelineState;
  id<MTLDepthStencilState> depthState;
  id<MTLBuffer> vertexBuffer;
  id<MTLBuffer> indexBuffer;
  int indexCount = 0;

  GridRenderer(id<MTLDevice> device, id<MTLLibrary> library) {
    NSError *err = nil;
    MTLRenderPipelineDescriptor *pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = [library newFunctionWithName:@"grid_vertex"];
    pd.fragmentFunction = [library newFunctionWithName:@"grid_fragment"];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
    pd.colorAttachments[0].blendingEnabled = YES;
    pd.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    pd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    pd.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;

    MTLVertexDescriptor *vd = [[MTLVertexDescriptor alloc] init];
    vd.attributes[0].format = MTLVertexFormatFloat3;
    vd.attributes[0].offset = 0;
    vd.attributes[0].bufferIndex = 0;
    vd.layouts[0].stride = sizeof(vec3);
    vd.layouts[0].stepRate = 1;
    vd.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
    pd.vertexDescriptor = vd;

    pipelineState = [device newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!pipelineState) {
      std::cerr << "Grid PSO Error: " << err.localizedDescription.UTF8String << std::endl;
      exit(1);   // consistent with every compute PSO: fail loudly at startup
    }

    MTLDepthStencilDescriptor *dd = [[MTLDepthStencilDescriptor alloc] init];
    dd.depthCompareFunction = MTLCompareFunctionLessEqual;
    dd.depthWriteEnabled = YES;
    depthState = [device newDepthStencilStateWithDescriptor:dd];

    const int N = 400;
    const float spacing = 1.0e12f;  // 400 × 1e12 = ±200e12 extent (covers all stars)
    vector<vec3> verts;
    vector<uint32_t> inds;
    for (int z = 0; z <= N; ++z) {
      for (int x = 0; x <= N; ++x) {
        verts.emplace_back((x - N / 2.0f) * spacing, 0.0f, (z - N / 2.0f) * spacing);
      }
    }
    for (int z = 0; z < N; ++z) {
      for (int x = 0; x < N; ++x) {
        int i = z * (N + 1) + x;
        inds.push_back(i); inds.push_back(i + 1);
        inds.push_back(i); inds.push_back(i + N + 1);
      }
    }
    indexCount = (int)inds.size();
    vertexBuffer = [device newBufferWithBytes:verts.data() length:verts.size() * sizeof(vec3) options:MTLResourceStorageModeShared];
    indexBuffer = [device newBufferWithBytes:inds.data() length:inds.size() * sizeof(uint32_t) options:MTLResourceStorageModeShared];
  }
};

// The drawable is extended-linear, so display-authored UI/overlay colors must
// be linearized before they are blended into it.
static inline float srgb_to_lin(float c) {
  return (c <= 0.04045f) ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}
static inline ImU32 linCol(int r, int g, int b, int a = 255) {
  return IM_COL32((int)(srgb_to_lin(r / 255.0f) * 255.0f + 0.5f),
                  (int)(srgb_to_lin(g / 255.0f) * 255.0f + 0.5f),
                  (int)(srgb_to_lin(b / 255.0f) * 255.0f + 0.5f), a);
}
static void linearizeImGuiStyle() {
  ImGuiStyle &st = ImGui::GetStyle();
  for (int i = 0; i < ImGuiCol_COUNT; i++) {
    ImVec4 &c = st.Colors[i];
    c.x = srgb_to_lin(c.x); c.y = srgb_to_lin(c.y); c.z = srgb_to_lin(c.z);
  }
}

// Write scene-linear half-float RGBA as an OpenEXR via ImageIO. The buffer is
// the accumulated HDR render target, so the file holds true radiometric values
// (highlights well above 1.0) with no tonemapping or UI baked in.
static void writeEXR(id<MTLBuffer> buf, int w, int h, NSUInteger bpr, int samples,
                     int projection) {
  const char *dir = getenv("BH_SHOT_DIR");
  char path[512];
  snprintf(path, sizeof(path), "%s/bh_ref_%d_%d.exr",
           dir ? dir : "/tmp", (int)getpid(), frameCount);

  // The readback is tightly packed RGBA16Float (bytesPerRow = w*8), which is
  // exactly what the writer's memcpy path expects.
  exr_image im = {};
  im.width = w; im.height = h;
  im.pixels = buf.contents;
  im.src_comps = 4;
  im.src_is_half = 1;
  im.write_alpha = 0;              // alpha is a constant 1 in this pipeline
  im.type = EXR_HALF;
  im.compression = EXR_ZIP;
  bool ok = (exr_write(path, &im) == 0);
  printf(ok ? "Reference EXR: %s (%dx%d, %d samples, bit-exact)\n"
            : "Reference EXR FAILED: %s (%dx%d, %d samples)\n",
         path, w, h, samples + 1);
  (void)bpr;

  // A 360 panorama is only usable if a player can recognise it, and ImageIO's
  // OpenEXR writer silently discards metadata (verified: GPano survives in
  // JPEG/PNG/HEIC/TIFF, vanishes in EXR). So panoramas also get a tonemapped
  // 16-bit PNG carrying the GPano XMP.
  if (projection != PROJ_EQUIRECT) return;
  char pngPath[512];
  snprintf(pngPath, sizeof(pngPath), "%s/bh_pano_%d_%d.png",
           dir ? dir : "/tmp", (int)getpid(), frameCount);
  std::vector<uint16_t> rgb16((size_t)w * h * 3);
  const __fp16 *src = (const __fp16 *)buf.contents;
  for (size_t i = 0; i < (size_t)w * h; i++) {
    for (int k = 0; k < 3; k++) {
      float v = (float)src[i * 4 + k];
      // SDR view of the reference data: ACES then sRGB encode.
      float t = (v * (2.51f * v + 0.03f)) / (v * (2.43f * v + 0.59f) + 0.14f);
      t = std::clamp(t, 0.0f, 1.0f);
      t = (t <= 0.0031308f) ? t * 12.92f : 1.055f * std::pow(t, 1.0f / 2.4f) - 0.055f;
      uint16_t q = (uint16_t)(std::clamp(t, 0.0f, 1.0f) * 65535.0f + 0.5f);
      rgb16[i * 3 + k] = (uint16_t)((q >> 8) | (q << 8));   // PNG is big-endian
    }
  }
  CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
  CFDataRef data = CFDataCreate(NULL, (const UInt8 *)rgb16.data(),
                                (CFIndex)(rgb16.size() * 2));
  CGDataProviderRef prov = CGDataProviderCreateWithCFData(data);
  CGImageRef img = CGImageCreate(w, h, 16, 48, (size_t)w * 6, cs,
                                 (CGBitmapInfo)kCGImageAlphaNone, prov, NULL, false,
                                 kCGRenderingIntentDefault);
  if (img) {
    CFStringRef ps = CFStringCreateWithCString(NULL, pngPath, kCFStringEncodingUTF8);
    CFURLRef url = CFURLCreateWithFileSystemPath(NULL, ps, kCFURLPOSIXPathStyle, false);
    CGImageDestinationRef dst =
        CGImageDestinationCreateWithURL(url, CFSTR("public.png"), 1, NULL);
    if (dst) {
      CGMutableImageMetadataRef md = CGImageMetadataCreateMutable();
      CGImageMetadataRegisterNamespaceForPrefix(
          md, CFSTR("http://ns.google.com/photos/1.0/panorama/"), CFSTR("GPano"), NULL);
      auto setNum = [&](const char *key, int val) {
        CFStringRef k = CFStringCreateWithCString(NULL, key, kCFStringEncodingUTF8);
        CFStringRef v = CFStringCreateWithFormat(NULL, NULL, CFSTR("%d"), val);
        CGImageMetadataSetValueWithPath(md, NULL, k, v);
        CFRelease(k); CFRelease(v);
      };
      CGImageMetadataSetValueWithPath(md, NULL, CFSTR("GPano:ProjectionType"),
                                      CFSTR("equirectangular"));
      setNum("GPano:FullPanoWidthPixels", w);
      setNum("GPano:FullPanoHeightPixels", h);
      setNum("GPano:CroppedAreaImageWidthPixels", w);
      setNum("GPano:CroppedAreaImageHeightPixels", h);
      setNum("GPano:CroppedAreaLeftPixels", 0);
      setNum("GPano:CroppedAreaTopPixels", 0);
      CGImageDestinationAddImageAndMetadata(dst, img, md, NULL);
      bool pok = CGImageDestinationFinalize(dst);
      printf("  360 companion: %s (%s)\n", pngPath, pok ? "GPano tagged" : "FAILED");
      CFRelease(md); CFRelease(dst);
    }
    CFRelease(url); CFRelease(ps);
    CGImageRelease(img);
  }
  CGDataProviderRelease(prov); CFRelease(data); CGColorSpaceRelease(cs);
}

// Halton low-discrepancy sequence for subpixel jitter.
static float halton(int index, int base) {
  float f = 1.0f, r = 0.0f;
  while (index > 0) {
    f /= base;
    r += f * (index % base);
    index /= base;
  }
  return r;
}

class MetalEngine {
public:
  id<MTLDevice> device;
  id<MTLCommandQueue> commandQueue;
  id<MTLComputePipelineState> raytracePSO;
  id<MTLComputePipelineState> temporalPSO;
  id<MTLComputePipelineState> physicsInitPSO;
  id<MTLComputePipelineState> physicsDriftPSO;
  id<MTLComputePipelineState> physicsKickPSO;
  id<MTLComputePipelineState> postPSO;
  id<MTLComputePipelineState> bloomExtractPSO;
  id<MTLComputePipelineState> lumReducePSO;

  id<MTLBuffer> camBuffer[3];
  id<MTLBuffer> objBuffer;          // Single buffer — GPU-only writes
  id<MTLBuffer> objUniformBuffer[3];
  id<MTLBuffer> sysUniformBuffer[3];
  id<MTLBuffer> gridUniformBuffer[3];
  id<MTLBuffer> lumBuffer[3];
  int currentFrame = 0;
  float currentExposure = 1.0f;

  id<MTLTexture> rtTex;               // raw raytrace output (this frame)
  id<MTLTexture> accumTex[2];         // temporal accumulation ping-pong
  id<MTLTexture> bloomTex;
  id<MTLTexture> bloomBlurTex;
  id<MTLTexture> depthTexture;
  int currentAccumIdx = 0;

  MPSImageGaussianBlur *mpsBloom;
  MPSImageGaussianBlur *mpsEht;
  float ehtSigmaPx = -1.0f;

  CAMetalLayer *metalLayer;
  std::unique_ptr<GridRenderer> gridRenderer;
  int bhIndex = -1;

  int drawableW, drawableH;
  GLFWwindow* winRef;
  dispatch_semaphore_t frameSemaphore;

  // Temporal accumulation state: N frames accumulated since the scene state
  // last changed. Static camera + jittered rays -> progressive supersampling.
  int accumN = 0;
  float edrHeadroom = 1.0f;     // display-reported; 1.0 on SDR panels
  bool accumHistoryValid = false;   // false after createResources(): the fresh
                                    // private textures hold undefined memory
  uint64_t lastStateHash = 0;

  // Geodesic-fan cache (recomputed when spin/charge change).
  std::vector<FanRay> fanRays;
  float fanSpin = -99.0f, fanCharge = -99.0f;

  MetalEngine(GLFWwindow *window, const vector<SimObject> &initialObjects) {
    winRef = window;
    frameSemaphore = dispatch_semaphore_create(3);
    device = MTLCreateSystemDefaultDevice();
    commandQueue = [device newCommandQueue];

    NSWindow *nswin = glfwGetCocoaWindow(window);
    glfwGetFramebufferSize(window, &drawableW, &drawableH);

    NSView *view = [nswin contentView];
    metalLayer = [CAMetalLayer layer];
    metalLayer.device = device;
    // Extended dynamic range: a half-float drawable in an extended-LINEAR
    // colorspace. Values above 1.0 drive the display brighter than reference
    // white, so the Doppler-boosted inner limb can actually glare on an XDR
    // panel instead of being crushed into SDR white by the tonemapper.
    metalLayer.pixelFormat = MTLPixelFormatRGBA16Float;
    metalLayer.framebufferOnly = NO;
    metalLayer.wantsExtendedDynamicRangeContent = YES;
    metalLayer.drawableSize = CGSizeMake(drawableW, drawableH);
    CGColorSpaceRef xlin = CGColorSpaceCreateWithName(kCGColorSpaceExtendedLinearSRGB);
    metalLayer.colorspace = xlin;
    CGColorSpaceRelease(xlin);
    [view setLayer:metalLayer];
    [view setWantsLayer:YES];

    NSError *err = nil;
    NSString *execDir = [[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent];
    id<MTLLibrary> lib = nil;

    NSString *metallibPath = [execDir stringByAppendingPathComponent:@"geodesic.metallib"];
    if ([[NSFileManager defaultManager] fileExistsAtPath:metallibPath]) {
      lib = [device newLibraryWithURL:[NSURL fileURLWithPath:metallibPath] error:&err];
      if (lib) std::cout << "Loaded precompiled geodesic.metallib" << std::endl;
    }

    if (!lib) {
      NSString *headerPath = [execDir stringByAppendingPathComponent:@"ShaderCommon.h"];
      NSString *shaderPath = [execDir stringByAppendingPathComponent:@"geodesic.metal"];
      NSString *headerSrc = [NSString stringWithContentsOfFile:headerPath encoding:NSUTF8StringEncoding error:&err];
      if (!headerSrc) { std::cerr << "Cannot load ShaderCommon.h: " << err.localizedDescription.UTF8String << std::endl; exit(1); }
      NSString *shaderSrc = [NSString stringWithContentsOfFile:shaderPath encoding:NSUTF8StringEncoding error:&err];
      if (!shaderSrc) { std::cerr << "Cannot load geodesic.metal: " << err.localizedDescription.UTF8String << std::endl; exit(1); }
      NSString *fullSrc = [NSString stringWithFormat:@"%@\n%@", headerSrc, shaderSrc];
      MTLCompileOptions *compileOpts = [[MTLCompileOptions alloc] init];
      // PRECISION POLICY: MTLMathModeRelaxed — permits non-IEEE sqrt/division
      // but preserves Inf/NaN. This is a deliberate 3x-performance trade-off:
      // the fp64 mirror suite (tests/validate_physics.py) certifies the
      // algorithm, A/B captures against MTLMathModeSafe show sub-1% mean
      // pixel differences, and full Fast math (which additionally flushes
      // NaN and reassociates aggressively) remains forbidden — it corrupts
      // the geodesics visibly. Pre-macOS-15 systems fall back to
      // fastMathEnabled = NO, which is STRICTLY SAFER (equal to Safe) since
      // the deprecated API has no relaxed setting. The offline metallib path
      // (scripts/build_metallib.sh) uses the same relaxed policy.
      if (@available(macOS 15.0, *)) {
          compileOpts.mathMode = MTLMathModeRelaxed;
      } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
          // Only executes on macOS < 15, where the symbol is not deprecated.
          compileOpts.fastMathEnabled = NO;
#pragma clang diagnostic pop
      }
      compileOpts.languageVersion = MTLLanguageVersion3_0;
      lib = [device newLibraryWithSource:fullSrc options:compileOpts error:&err];
      if (!lib) { std::cerr << "Metal Library Error: " << err.localizedDescription.UTF8String << std::endl; exit(1); }
      std::cout << "Using runtime shader compilation (development mode)" << std::endl;
    }

    auto createPSO = [&](const char* name) {
      id<MTLFunction> func = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
      if (!func) { std::cerr << "Metal Function Not Found: " << name << std::endl; exit(1); }
      MTLComputePipelineDescriptor *desc = [[MTLComputePipelineDescriptor alloc] init];
      desc.computeFunction = func;
      desc.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
      id<MTLComputePipelineState> pso = [device newComputePipelineStateWithDescriptor:desc options:0 reflection:nil error:&err];
      if (!pso) { std::cerr << "PSO Creation Error for " << name << ": " << err.localizedDescription.UTF8String << std::endl; exit(1); }
      return pso;
    };

    raytracePSO       = createPSO("raytrace");
    temporalPSO       = createPSO("temporal_accum");
    physicsInitPSO    = createPSO("physics_init_accel");
    physicsDriftPSO   = createPSO("physics_drift");
    physicsKickPSO    = createPSO("physics_force_kick");
    postPSO           = createPSO("post_process_suite");
    bloomExtractPSO   = createPSO("bloom_extract");
    lumReducePSO      = createPSO("luminance_reduce");

    gridRenderer = std::make_unique<GridRenderer>(device, lib);
    mpsBloom = [[MPSImageGaussianBlur alloc] initWithDevice:device sigma:8.0f];
    mpsEht = nil;

    objBuffer = [device newBufferWithBytes:initialObjects.data() length:sizeof(SimObject) * initialObjects.size() options:MTLResourceStorageModeShared];

    for (size_t i = 0; i < initialObjects.size(); i++) {
      if (initialObjects[i].mass > 1e35f) { bhIndex = (int)i; break; }
    }
    if (bhIndex < 0) {
      std::cerr << "Fatal: no black hole (mass > 1e35) in initial object set" << std::endl;
      exit(1);
    }

    for (int i = 0; i < 3; i++) {
      camBuffer[i] = [device newBufferWithLength:sizeof(CameraData) options:MTLResourceStorageModeShared];
      objUniformBuffer[i] = [device newBufferWithLength:sizeof(ObjectsUniform) options:MTLResourceStorageModeShared];
      sysUniformBuffer[i] = [device newBufferWithLength:sizeof(SystemUniforms) options:MTLResourceStorageModeShared];
      gridUniformBuffer[i] = [device newBufferWithLength:sizeof(GridUniforms) options:MTLResourceStorageModeShared];
      lumBuffer[i] = [device newBufferWithLength:sizeof(uint32_t) * 2 options:MTLResourceStorageModeShared];
      memset(lumBuffer[i].contents, 0, sizeof(uint32_t) * 2);
    }

    // Bootstrap velocity-Verlet accelerations (a_0 seed).
    {
      ObjectsUniform *ouPtr = (ObjectsUniform *)objUniformBuffer[0].contents;
      ouPtr->count = (int)initialObjects.size();
      ouPtr->bh_index = bhIndex;
      id<MTLCommandBuffer> bootCmd = [commandQueue commandBuffer];
      id<MTLComputeCommandEncoder> boot = [bootCmd computeCommandEncoder];
      [boot setComputePipelineState:physicsInitPSO];
      [boot setBuffer:objBuffer offset:0 atIndex:0];
      [boot setBuffer:objUniformBuffer[0] offset:0 atIndex:1];
      [boot dispatchThreads:MTLSizeMake(initialObjects.size(), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      [boot endEncoding];
      [bootCmd commit];
      [bootCmd waitUntilCompleted];
    }

    createResources();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOther(window, true);
    ImGui_ImplMetal_Init(device);
    linearizeImGuiStyle();
  }

  void createResources() {
    MTLTextureDescriptor *itd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:drawableW height:drawableH mipmapped:NO];
    itd.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    itd.storageMode = MTLStorageModePrivate;
    rtTex = [device newTextureWithDescriptor:itd];
    accumTex[0] = [device newTextureWithDescriptor:itd];
    accumTex[1] = [device newTextureWithDescriptor:itd];

    int bloomW = std::max(1, drawableW / 2);
    int bloomH = std::max(1, drawableH / 2);
    MTLTextureDescriptor *btd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:bloomW height:bloomH mipmapped:NO];
    btd.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    btd.storageMode = MTLStorageModePrivate;
    bloomTex = [device newTextureWithDescriptor:btd];
    bloomBlurTex = [device newTextureWithDescriptor:btd];

    MTLTextureDescriptor *dd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float width:drawableW height:drawableH mipmapped:NO];
    dd.usage = MTLTextureUsageRenderTarget;
    dd.storageMode = MTLStorageModePrivate;
    depthTexture = [device newTextureWithDescriptor:dd];

    accumN = 0;   // fresh (undefined) accumulation targets: force replace
    accumHistoryValid = false;
  }

  // Hash of everything that changes the rendered image; a change resets the
  // temporal accumulation so stale frames never ghost into the new state.
  uint64_t stateHash() {
    struct Snap {
      vec3 pos, target;
      float spin, charge, density, r_out, temp, jets, nebula, scint, boost;
      float t_r0, t_h, t_l0, t_alpha, t_absorb, t_temp, t_dens, edr;
      float mp1, mp2, msep, mglow;
      int lens, beaming, stars, disk, gravity, dmodel, bmode, proj;
      int w, h;
    } s;
    memset(&s, 0, sizeof(s));
    s.pos = camera.position(); s.target = camera.target;
    s.spin = g_sim.blackHoleSpin; s.charge = g_sim.enCharge ? g_sim.blackHoleCharge : 0.0f;
    s.density = g_sim.enDisk ? g_sim.diskDensity : 0.0f;
    s.r_out = g_sim.diskOuterRadius; s.temp = g_sim.diskTempK;
    s.jets = g_sim.enJets ? g_sim.jetIntensity : 0.0f;
    s.nebula = g_sim.enNebula ? g_sim.nebulaIntensity : 0.0f;
    s.scint = g_sim.enScintillation ? g_sim.starScintillation : 0.0f;
    s.boost = g_sim.photonRingBoost;
    s.t_r0 = g_sim.torusR0; s.t_h = g_sim.torusH; s.t_l0 = g_sim.torusL0;
    s.t_alpha = g_sim.torusAlpha; s.t_absorb = g_sim.torusAbsorb;
    s.t_temp = g_sim.torusTemp; s.t_dens = g_sim.torusDensity;
    s.edr = g_sim.enEDR ? edrHeadroom : 1.0f;
    s.dmodel = g_sim.diskModel;
    s.bmode = g_sim.bhMode; s.proj = g_sim.projection;
    s.mp1 = g_sim.mpM1; s.mp2 = g_sim.mpM2;
    s.msep = g_sim.mpSep; s.mglow = g_sim.mpGlow;
    s.lens = g_sim.lensMode; s.beaming = g_sim.beamingMode;
    s.stars = g_sim.enStarBodies ? 1 : 0; s.disk = g_sim.enDisk ? 1 : 0;
    // N-body star motion is deliberately NOT hashed: the stars crawl a few
    // pixels per minute, and resetting the accumulator every frame for them
    // would disable progressive supersampling of the (static) disk + shadow.
    s.gravity = 0;
    s.w = drawableW; s.h = drawableH;
    // FNV-1a over the snapshot bytes.
    const uint8_t* p = (const uint8_t*)&s;
    uint64_t hsh = 1469598103934665603ull;
    for (size_t i = 0; i < sizeof(s); i++) { hsh ^= p[i]; hsh *= 1099511628211ull; }
    return hsh;
  }

  void render(int activeObjectCount) {
    dispatch_semaphore_wait(frameSemaphore, DISPATCH_TIME_FOREVER);

    int curW, curH;
    glfwGetFramebufferSize(winRef, &curW, &curH);
    if (curW != drawableW || curH != drawableH) {
      resize(curW, curH);
    }

    double now = glfwGetTime();
    deltaTime = std::clamp(float(now - lastFrameTime), 0.0001f, 0.1f);
    lastFrameTime = now;
    simulationTime += deltaTime * g_sim.timeScale;
    currentFrame = (currentFrame + 1) % 3;
    frameCount++;

    CameraData *cPtr = (CameraData *)camBuffer[currentFrame].contents;
    vec3 camPos = camera.position();
    vec3 fwd = normalize(camera.target - camPos);
    vec3 right = normalize(cross(fwd, vec3(0, 1, 0)));
    vec3 up = cross(right, fwd);
    cPtr->camPos = glm::vec4(camPos, 0.0f); cPtr->camRight = glm::vec4(right, 0.0f); cPtr->camUp = glm::vec4(up, 0.0f);
    cPtr->camForward = glm::vec4(fwd, 0.0f); cPtr->tanHalfFov = tan(radians(30.0f));
    cPtr->aspect = float(drawableW) / float(drawableH);

    ObjectsUniform *ouPtr = (ObjectsUniform *)objUniformBuffer[currentFrame].contents;
    ouPtr->count = activeObjectCount;
    ouPtr->bh_index = bhIndex;

    // Joint extremality clamp: a² + Q² <= 1 keeps the parameters inside the
    // Kerr-Newman black-hole family (no silent naked-singularity renders).
    float spin_c = std::clamp(g_sim.blackHoleSpin, -1.0f, 1.0f);
    float charge_raw = g_sim.enCharge ? std::clamp(g_sim.blackHoleCharge, 0.0f, 1.0f) : 0.0f;
    float q_max = std::sqrt(std::max(1.0f - spin_c * spin_c, 0.0f));
    float charge_c = std::min(charge_raw, q_max);
    KerrRadii kr = kerr_radii(spin_c, charge_c);

    // Temporal accumulation control.
    uint64_t hsh = stateHash();
    bool stateChanged = (hsh != lastStateHash);
    lastStateHash = hsh;
    if (stateChanged) accumN = 0; else accumN = std::min(accumN + 1, 255);
    float accumAlpha;
    if (accumN == 0)                      accumAlpha = 1.0f;
    else                                  accumAlpha = 1.0f / float(accumN + 1);
    // Time-animated raytrace content (orbiting stars, jet turbulence, star
    // twinkle) must not freeze into a deep temporal mean: cap the history at
    // ~10 frames whenever any of it is animating (still a visible AA win on
    // the static disk/shadow).
    bool animated = g_sim.timeScale > 0.0f &&
                    ((g_sim.Gravity && g_sim.enStarBodies) || g_sim.enJets ||
                     g_sim.enScintillation || g_sim.enNebula);
    if (animated) accumAlpha = std::max(accumAlpha, 0.1f);
    // Motion-blur trail — but never blend into UNDEFINED history: right after
    // createResources() the accumulation textures hold garbage, and a resize
    // is not motion.
    if (stateChanged && g_sim.enMotionBlur && accumHistoryValid)
      accumAlpha = std::max(1.0f - std::clamp(g_sim.motionBlur, 0.0f, 0.99f), 0.01f);
    accumHistoryValid = true;   // valid from the moment this frame's accum pass runs

    SystemUniforms *sysPtr = (SystemUniforms *)sysUniformBuffer[currentFrame].contents;
    sysPtr->time = simulationTime;
    sysPtr->spin = spin_c;
    sysPtr->star_scint   = g_sim.enScintillation ? std::clamp(g_sim.starScintillation, 0.0f, 1.0f) : 0.0f;
    sysPtr->nebula_int   = g_sim.enNebula        ? std::clamp(g_sim.nebulaIntensity,   0.0f, 2.0f) : 0.0f;
    sysPtr->charge       = charge_c;
    sysPtr->bloom_threshold = std::clamp(g_sim.bloomThreshold, 0.0f, 1.0f);
    sysPtr->enable_bloom = g_sim.enBloom ? 1 : 0;
    sysPtr->flare_int    = g_sim.enFlare      ? std::clamp(g_sim.flareIntensity, 0.0f, 2.0f)  : 0.0f;
    sysPtr->motion_blur  = g_sim.enMotionBlur ? std::clamp(g_sim.motionBlur,     0.0f, 0.99f) : 0.0f;
    sysPtr->film_grain   = g_sim.enFilmGrain  ? std::clamp(g_sim.filmGrain,      0.0f, 0.1f)  : 0.0f;
    sysPtr->disk_density = g_sim.enDisk       ? std::clamp(g_sim.diskDensity,    0.0f, 1.0f)  : 0.0f;
    sysPtr->disk_r_out   = std::clamp(g_sim.diskOuterRadius, 4.0f, 24.0f);
    sysPtr->disk_temp    = std::clamp(g_sim.diskTempK, 1000.0f, 20000.0f);
    sysPtr->vignette_int = g_sim.enVignette ? std::clamp(g_sim.vignetteIntensity, 0.0f, 1.0f) : 0.0f;
    sysPtr->gw_amp       = g_sim.enGW     ? std::clamp(g_sim.gwAmplitude,     0.0f, 2.0f) : 0.0f;
    sysPtr->jet_int      = g_sim.enJets   ? std::clamp(g_sim.jetIntensity,    0.0f, 2.0f) : 0.0f;
    sysPtr->r_horizon    = kr.r_horizon;
    sysPtr->r_isco       = kr.r_isco;
    sysPtr->photon_ring_boost = std::clamp(g_sim.photonRingBoost, 0.0f, 4.0f);
    sysPtr->accum_alpha  = accumAlpha;
    // Subpixel jitter converges under accumulation; when history is being
    // REPLACED (camera moving), an uncompensated jitter just makes edges crawl
    // — render the unjittered center sample instead.
    bool jitterOn = accumAlpha < 0.999f;
    sysPtr->jitter_x     = jitterOn ? halton((frameCount % 64) + 1, 2) - 0.5f : 0.0f;
    sysPtr->jitter_y     = jitterOn ? halton((frameCount % 64) + 1, 3) - 0.5f : 0.0f;
    // EDR headroom tracks display brightness/ambient, so re-query it live.
    if (frameCount % 30 == 1) {
      NSScreen *scr = [glfwGetCocoaWindow(winRef) screen] ?: [NSScreen mainScreen];
      float hr = (float)scr.maximumExtendedDynamicRangeColorComponentValue;
      edrHeadroom = std::clamp(hr, 1.0f, 16.0f);
    }
    sysPtr->edr_headroom = g_sim.enEDR ? edrHeadroom : 1.0f;
    sysPtr->disk_model   = g_sim.diskModel;
    sysPtr->bh_mode      = g_sim.bhMode;
    sysPtr->projection   = g_sim.projection;
    sysPtr->mp_m1        = std::clamp(g_sim.mpM1, 0.0f, 2.0f);
    sysPtr->mp_m2        = std::clamp(g_sim.mpM2, 0.0f, 2.0f);
    sysPtr->mp_sep       = std::clamp(g_sim.mpSep, 0.0f, 40.0f);
    sysPtr->mp_glow      = std::clamp(g_sim.mpGlow, 0.0f, 2.0f);
    sysPtr->torus_r0     = std::clamp(g_sim.torusR0, 2.0f, 30.0f);
    sysPtr->torus_h      = std::clamp(g_sim.torusH, 0.0f, 12.0f);
    sysPtr->torus_l0     = std::clamp(g_sim.torusL0, 0.0f, 6.0f);
    sysPtr->torus_alpha  = std::clamp(g_sim.torusAlpha, -3.0f, 3.0f);
    sysPtr->torus_absorb = std::clamp(g_sim.torusAbsorb, 0.0f, 8.0f);
    sysPtr->torus_temp   = std::clamp(g_sim.torusTemp, 1000.0f, 20000.0f);
    sysPtr->torus_density = std::clamp(g_sim.torusDensity, 0.0f, 4.0f);
    sysPtr->lens_mode    = g_sim.lensMode;
    sysPtr->beaming_mode = g_sim.beamingMode;
    sysPtr->star_bodies  = g_sim.enStarBodies ? 1 : 0;

    // N-body: fixed physics substep, independent of frame rate. DT_MAX is
    // T_inner/64 for the innermost orbit (r = 8e12 m around Sgr A*), so even
    // a stalled 100 ms frame at timeScale 5 integrates accurately.
    const float DT_MAX = 9.3e4f;
    float dt_total = 2500000.0f * deltaTime * std::clamp(g_sim.timeScale, 0.0f, 5.0f);
    int n_sub = std::clamp((int)std::ceil(dt_total / DT_MAX), 1, 32);
    sysPtr->dt_phys = (dt_total > 0.0f) ? dt_total / n_sub : 0.0f;

    // Auto-exposure: read THIS slot before zeroing it. The slot was last
    // written by GPU frame N-3, which the frame semaphore guarantees has
    // completed — reading a younger slot races a GPU that runs 2+ frames
    // behind the CPU and sees only the CPU's own memset zeros.
    uint32_t* lumData = (uint32_t*)lumBuffer[currentFrame].contents;
    uint32_t sumEncoded = lumData[0];
    uint32_t pixCount = lumData[1];
    if (pixCount > 100) {
      float avgLogLum = (float(sumEncoded) / float(pixCount)) / 100.0f - 12.0f;
      float targetExposure = std::clamp(0.18f / exp2f(avgLogLum), 0.05f, 8.0f);
      currentExposure += (targetExposure - currentExposure) * std::min(deltaTime * 3.0f, 1.0f);
    }
    if (qaMode && (frameCount % 30 == 0 || (stateChanged && frameCount > 5)))
      printf("QA f%d: sum=%u pix=%u exposure=%.3f accumN=%d changed=%d dims=%dx%d cam=(%.3g,%.3g,%.3g)\n",
             frameCount, sumEncoded, pixCount, currentExposure, accumN, (int)stateChanged,
             drawableW, drawableH, camera.position().x, camera.position().y, camera.position().z);
    sysPtr->exposure = currentExposure;
    memset(lumBuffer[currentFrame].contents, 0, sizeof(uint32_t) * 2);

    // EHT lens: (re)build the beam-blur kernel when the FWHM changes.
    if (g_sim.lensMode == LENS_EHT) {
      float d_rs = glm::length(camPos) / (float)SagA_rs;
      float pxPerRad = float(drawableH) * 0.5f / tan(radians(30.0f));
      float sigmaPx = (g_sim.ehtBeamFwhm / 2.355f) / std::max(d_rs, 1.0f) * pxPerRad * 0.5f; // half-res chain
      sigmaPx = std::clamp(sigmaPx, 0.5f, 48.0f);
      if (!mpsEht || std::abs(sigmaPx - ehtSigmaPx) > 0.05f) {
        mpsEht = [[MPSImageGaussianBlur alloc] initWithDevice:device sigma:sigmaPx];
        ehtSigmaPx = sigmaPx;
      }
    }

    GridUniforms *gPtr = (GridUniforms *)gridUniformBuffer[currentFrame].contents;
    gPtr->viewProj = camera.getViewProj(float(drawableW) / float(drawableH));

    @autoreleasepool {
      id<CAMetalDrawable> drawable = [metalLayer nextDrawable];
      if (!drawable) {
        dispatch_semaphore_signal(frameSemaphore);
        return;
      }

      id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];

      // 1. N-Body Physics — velocity-Verlet (KDK), substepped at fixed dt.
      if (g_sim.Gravity && sysPtr->dt_phys > 0.0f) {
        MTLSize physTpg = MTLSizeMake(32, 1, 1);
        MTLSize physGrid = MTLSizeMake(activeObjectCount, 1, 1);
        for (int ss = 0; ss < n_sub; ss++) {
          id<MTLComputeCommandEncoder> drift = [cmd computeCommandEncoder];
          [drift setComputePipelineState:physicsDriftPSO];
          [drift setBuffer:objBuffer offset:0 atIndex:0];
          [drift setBuffer:objUniformBuffer[currentFrame] offset:0 atIndex:1];
          [drift setBuffer:sysUniformBuffer[currentFrame] offset:0 atIndex:2];
          [drift dispatchThreads:physGrid threadsPerThreadgroup:physTpg];
          [drift endEncoding];

          id<MTLComputeCommandEncoder> kick = [cmd computeCommandEncoder];
          [kick setComputePipelineState:physicsKickPSO];
          [kick setBuffer:objBuffer offset:0 atIndex:0];
          [kick setBuffer:objUniformBuffer[currentFrame] offset:0 atIndex:1];
          [kick setBuffer:sysUniformBuffer[currentFrame] offset:0 atIndex:2];
          [kick dispatchThreads:physGrid threadsPerThreadgroup:physTpg];
          [kick endEncoding];
        }
      }

      // 2. Geodesic Raytrace (jittered) -> rtTex
      MTLSize tpg = MTLSizeMake(32, 8, 1);
      id<MTLComputeCommandEncoder> ray = [cmd computeCommandEncoder];
      [ray setComputePipelineState:raytracePSO];
      [ray setTexture:rtTex atIndex:0];
      [ray setBuffer:camBuffer[currentFrame] offset:0 atIndex:0];
      [ray setBuffer:objBuffer offset:0 atIndex:1];
      [ray setBuffer:objUniformBuffer[currentFrame] offset:0 atIndex:2];
      [ray setBuffer:sysUniformBuffer[currentFrame] offset:0 atIndex:3];
      [ray dispatchThreads:MTLSizeMake(drawableW, drawableH, 1) threadsPerThreadgroup:tpg];
      [ray endEncoding];

      // 3. Temporal accumulation: rtTex + accumTex[prev] -> accumTex[cur]
      int curAcc = currentAccumIdx;
      int prevAcc = 1 - currentAccumIdx;
      {
        id<MTLComputeCommandEncoder> ta = [cmd computeCommandEncoder];
        [ta setComputePipelineState:temporalPSO];
        [ta setTexture:rtTex atIndex:0];
        [ta setTexture:accumTex[prevAcc] atIndex:1];
        [ta setTexture:accumTex[curAcc] atIndex:2];
        [ta setBuffer:sysUniformBuffer[currentFrame] offset:0 atIndex:0];
        [ta dispatchThreads:MTLSizeMake(drawableW, drawableH, 1) threadsPerThreadgroup:tpg];
        [ta endEncoding];
      }

      // 4. Bloom / EHT-beam extraction (half-res) + MPS Gaussian
      bool wantBlurChain = g_sim.enBloom || g_sim.lensMode == LENS_EHT;
      if (wantBlurChain) {
        id<MTLComputeCommandEncoder> be = [cmd computeCommandEncoder];
        [be setComputePipelineState:bloomExtractPSO];
        [be setTexture:accumTex[curAcc] atIndex:0];
        [be setTexture:bloomTex atIndex:1];
        [be setBuffer:sysUniformBuffer[currentFrame] offset:0 atIndex:0];
        int bw = std::max(1, drawableW / 2);
        int bh = std::max(1, drawableH / 2);
        [be dispatchThreads:MTLSizeMake(bw, bh, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [be endEncoding];

        MPSImageGaussianBlur *blur = (g_sim.lensMode == LENS_EHT && mpsEht) ? mpsEht : mpsBloom;
        [blur encodeToCommandBuffer:cmd sourceTexture:bloomTex destinationTexture:bloomBlurTex];
      }

      // 5. Luminance metering (auto-exposure), on the accumulated HDR
      {
        id<MTLComputeCommandEncoder> lum = [cmd computeCommandEncoder];
        [lum setComputePipelineState:lumReducePSO];
        [lum setTexture:accumTex[curAcc] atIndex:0];
        [lum setBuffer:lumBuffer[currentFrame] offset:0 atIndex:0];
        int lumW = (drawableW + 3) / 4;
        int lumH = (drawableH + 3) / 4;
        [lum dispatchThreads:MTLSizeMake(lumW, lumH, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [lum endEncoding];
      }

      // 6. Post-processing (exposure -> bloom -> flare -> vignette -> ACES -> grain)
      id<MTLComputeCommandEncoder> post = [cmd computeCommandEncoder];
      [post setComputePipelineState:postPSO];
      [post setTexture:accumTex[curAcc] atIndex:0];
      [post setTexture:drawable.texture atIndex:1];
      [post setTexture:bloomBlurTex atIndex:2];
      [post setBuffer:sysUniformBuffer[currentFrame] offset:0 atIndex:0];
      [post dispatchThreads:MTLSizeMake(drawableW, drawableH, 1) threadsPerThreadgroup:tpg];
      [post endEncoding];

      currentAccumIdx = 1 - currentAccumIdx;

      // 7. Grid render pass + overlays + ImGui
      MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
      rpd.colorAttachments[0].texture = drawable.texture;
      rpd.colorAttachments[0].loadAction = MTLLoadActionLoad;
      rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
      rpd.depthAttachment.texture = depthTexture;
      rpd.depthAttachment.loadAction = MTLLoadActionClear;
      rpd.depthAttachment.clearDepth = 1.0;
      rpd.depthAttachment.storeAction = MTLStoreActionStore;

      id<MTLRenderCommandEncoder> ren = [cmd renderCommandEncoderWithDescriptor:rpd];
      if (g_sim.enGrid) {
        [ren setRenderPipelineState:gridRenderer->pipelineState];
        [ren setDepthStencilState:gridRenderer->depthState];
        [ren setVertexBuffer:gridRenderer->vertexBuffer offset:0 atIndex:0];
        [ren setVertexBuffer:gridUniformBuffer[currentFrame] offset:0 atIndex:1];
        [ren setVertexBuffer:objBuffer offset:0 atIndex:2];
        [ren setVertexBuffer:objUniformBuffer[currentFrame] offset:0 atIndex:3];
        [ren setVertexBuffer:sysUniformBuffer[currentFrame] offset:0 atIndex:4];
        [ren setFragmentTexture:accumTex[curAcc] atIndex:0];
        [ren setFragmentBuffer:sysUniformBuffer[currentFrame] offset:0 atIndex:0];
        [ren drawIndexedPrimitives:MTLPrimitiveTypeLine indexCount:gridRenderer->indexCount indexType:MTLIndexTypeUInt32 indexBuffer:gridRenderer->indexBuffer indexBufferOffset:0];
      }

      ImGui_ImplMetal_NewFrame(rpd);
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();

      drawOverlays(spin_c, charge_c, kr);
      drawControlPanel(kr, spin_c, charge_c);

      ImGui::Render();
      ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), cmd, ren);
      [ren endEncoding];

      // Reference-still capture: the ACCUMULATED scene-linear HDR (pre-tonemap,
      // pre-UI), written as a half-float OpenEXR. macOS ImageIO supports
      // com.ilm.openexr-image natively, so this needs no third-party library.
      bool exrThisFrame = exrPending;
      if (exrThisFrame) exrPending = false;
      id<MTLBuffer> exrReadback = nil;
      NSUInteger exrBpr = 0;
      if (exrThisFrame) {
          exrBpr = drawableW * 8;   // RGBA16Float
          exrReadback = [device newBufferWithLength:exrBpr * drawableH
                                            options:MTLResourceStorageModeShared];
          id<MTLBlitCommandEncoder> eb = [cmd blitCommandEncoder];
          [eb copyFromTexture:accumTex[curAcc]
                  sourceSlice:0 sourceLevel:0
                 sourceOrigin:MTLOriginMake(0, 0, 0)
                   sourceSize:MTLSizeMake(drawableW, drawableH, 1)
                     toBuffer:exrReadback
            destinationOffset:0
       destinationBytesPerRow:exrBpr
     destinationBytesPerImage:exrBpr * drawableH];
          [eb endEncoding];
      }

      // Screenshot capture: blit in the SAME command buffer, before present.
      bool captureThisFrame = screenshotPending;
      if (captureThisFrame) screenshotPending = false;

      NSUInteger captureW = 0, captureH = 0, captureBpr = 0;
      id<MTLBuffer> captureReadback = nil;
      if (captureThisFrame) {
          captureW = drawable.texture.width;
          captureH = drawable.texture.height;
          captureBpr = captureW * 8;   // RGBA16Float
          captureReadback = [device newBufferWithLength:captureBpr * captureH
                                                options:MTLResourceStorageModeShared];

          id<MTLBlitCommandEncoder> capBlit = [cmd blitCommandEncoder];
          [capBlit copyFromTexture:drawable.texture
                       sourceSlice:0 sourceLevel:0
                      sourceOrigin:MTLOriginMake(0, 0, 0)
                        sourceSize:MTLSizeMake(captureW, captureH, 1)
                          toBuffer:captureReadback
                 destinationOffset:0
            destinationBytesPerRow:captureBpr
          destinationBytesPerImage:captureBpr * captureH];
          [capBlit endEncoding];
      }

      [cmd presentDrawable:drawable];

      dispatch_semaphore_t sem = frameSemaphore;
      [cmd addCompletedHandler:^(id<MTLCommandBuffer> _) {
          dispatch_semaphore_signal(sem);
      }];
      [cmd commit];

      if (exrThisFrame) {
          [cmd waitUntilCompleted];
          writeEXR(exrReadback, drawableW, drawableH, exrBpr, accumN, g_sim.projection);
      }

      if (captureThisFrame) {
          [cmd waitUntilCompleted];
          uint8_t *data = (uint8_t *)captureReadback.contents;
          char path[512];
          const char* dir = getenv("BH_SHOT_DIR");
          snprintf(path, sizeof(path), "%s/bh_%d_%d.ppm",
                   dir ? dir : "/tmp", (int)getpid(), frameCount);
          // O_EXCL + O_NOFOLLOW: never truncate an existing file or follow a
          // pre-planted symlink in a world-writable directory.
          int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0644);
          FILE *f = (fd >= 0) ? fdopen(fd, "wb") : nullptr;
          if (f) {
              fprintf(f, "P6\n%lu %lu\n255\n", captureW, captureH);
              // The drawable is extended-linear half-float. PPM is an SDR
              // container, so encode the [0,1] range with the sRGB transfer
              // function and report the true peak — values above 1.0 are the
              // EDR highlights the display shows but the file cannot hold.
              std::vector<uint8_t> rgb(captureW * captureH * 3);
              uint8_t *dst = rgb.data();
              float peak = 0.0f;
              for (NSUInteger y = 0; y < captureH; y++) {
                  const __fp16 *row = (const __fp16 *)(data + y * captureBpr);
                  for (NSUInteger x = 0; x < captureW; x++) {
                      for (int k = 0; k < 3; k++) {
                          float v = (float)row[x * 4 + k];
                          peak = std::max(peak, v);
                          float c = std::clamp(v, 0.0f, 1.0f);
                          c = (c <= 0.0031308f) ? c * 12.92f
                                                : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
                          *dst++ = (uint8_t)(c * 255.0f + 0.5f);
                      }
                  }
              }
              fwrite(rgb.data(), 1, rgb.size(), f);
              fclose(f);
              printf("Capture frame %d: %s (%lux%lu) peak=%.3f headroom=%.2f\n",
                     frameCount, path, captureW, captureH, peak, edrHeadroom);
          } else {
              if (fd >= 0) close(fd);
              printf("Failed to open %s for writing\n", path);
          }
      }
    }
  }

  // --- Learning overlays: analytic curves drawn over the raytraced image. ---
  void drawOverlays(float spin, float charge, const KerrRadii& kr) {
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    glm::mat4 vp = camera.getViewProj(float(drawableW) / float(drawableH));
    float winScaleX = 1.0f, winScaleY = 1.0f;
    {
      int ww, wh;
      glfwGetWindowSize(winRef, &ww, &wh);
      if (ww > 0 && wh > 0) { winScaleX = float(ww); winScaleY = float(wh); }
    }
    auto project = [&](glm::vec3 world, ImVec2* out) -> bool {
      glm::vec4 clip = vp * glm::vec4(world, 1.0f);
      if (clip.w <= 0.0f) return false;
      out->x = (clip.x / clip.w * 0.5f + 0.5f) * winScaleX;
      out->y = (0.5f - clip.y / clip.w * 0.5f) * winScaleY;
      return true;
    };

    if (g_sim.ovMarkers) {
      // Coordinate-space circles in the equatorial plane (NOT lensed images:
      // the horizon's APPARENT silhouette is ~2.6x larger — that difference
      // is the teaching point, made explicit by the critical-curve overlay).
      struct Marker { float r_rs; ImU32 col; const char* label; };
      Marker marks[] = {
        { kr.r_horizon, linCol(255,  70,  70, 200), "event horizon (coord.)" },
        { 1.0f,         linCol(190, 110, 255, 170), "ergosphere (equator)" },
        { kr.r_photon,  linCol(255, 210,  80, 200), "photon orbit" },
        { kr.r_isco,    linCol( 90, 220, 140, 200), "ISCO" },
      };
      int n_marks = (std::abs(spin) > 0.01f) ? 4 : 4;
      for (int m = 0; m < n_marks; m++) {
        if (m == 1 && std::abs(spin) <= 0.01f) continue;   // no ergoregion at a=0
        float R = marks[m].r_rs * (float)SagA_rs;
        ImVec2 prev; bool havePrev = false;
        for (int k = 0; k <= 96; k++) {
          float t = 2.0f * PI_F * k / 96.0f;
          ImVec2 p;
          if (project(glm::vec3(R * cosf(t), 0.0f, R * sinf(t)), &p)) {
            if (havePrev) dl->AddLine(prev, p, marks[m].col, 1.5f);
            prev = p; havePrev = true;
          } else havePrev = false;
        }
        ImVec2 lp;
        if (project(glm::vec3(marks[m].r_rs * (float)SagA_rs, 0.0f, 0.0f), &lp))
          dl->AddText(ImVec2(lp.x + 4, lp.y - 12), marks[m].col, marks[m].label);
      }
    }

    if (g_sim.ovCriticalCurve) {
      // Bardeen analytic shadow boundary, projected as angular offsets from
      // the BH direction. Inclination = camera polar angle from the spin axis.
      float inc = std::acos(std::clamp(camera.position().y / glm::length(camera.position()), -1.0f, 1.0f));
      if (inc > PI_F * 0.5f) inc = PI_F - inc;   // symmetric above/below plane
      auto curve = compute_critical_curve(spin, charge, inc);
      ImVec2 bhScreen;
      if (!curve.empty() && project(glm::vec3(0.0f), &bhScreen)) {
        float d_rs = glm::length(camera.position()) / (float)SagA_rs;
        float thf = tanf(radians(30.0f));
        float pxPerRadY = winScaleY * 0.5f / thf;
        float pxPerRadX = winScaleX * 0.5f / (thf * (float)drawableW / (float)drawableH);
        bool below = camera.position().y < 0.0f;
        // Finite-distance aberration: the impact parameter b maps to the
        // apparent angle sin(psi) = (b/d) sqrt(-g_tt(d)) for a static
        // observer, not the small-angle b/d of the infinity limit.
        float lapse = std::sqrt(std::max(1.0f - 1.0f / std::max(d_rs, 1.5f), 0.1f));
        ImVec2 prev; bool havePrev = false;
        for (auto& ab : curve) {
          // (alpha, beta) are in M units -> /2 for rs.
          float ax = (ab.x * 0.5f) / d_rs * lapse;
          float by = (ab.y * 0.5f) / d_rs * lapse * (below ? -1.0f : 1.0f);
          ImVec2 p(bhScreen.x - ax * pxPerRadX, bhScreen.y - by * pxPerRadY);
          if (havePrev) dl->AddLine(prev, p, linCol(80, 255, 120, 220), 1.5f);
          prev = p; havePrev = true;
        }
        dl->AddText(ImVec2(bhScreen.x + 8, bhScreen.y + 8), linCol(80, 255, 120, 220),
                    "critical curve (analytic shadow edge)");
      }
    }

    // Lens legends: every false-color lens ships with its colorbar + caption.
    if (g_sim.lensMode == LENS_RING_ORDER && g_sim.bhMode == BH_MP_BINARY) {
      ImVec2 o(12, winScaleY - 96);
      dl->AddRectFilled(ImVec2(o.x - 6, o.y - 6), ImVec2(o.x + 360, o.y + 86), linCol(0, 0, 0, 160), 6.0f);
      struct { ImU32 c; const char* t; } brows[] = {
        { linCol(230,  89,  26), "captured by hole 1" },
        { linCol( 26, 140, 230), "captured by hole 2" },
        { linCol(242, 230,  51), "trapped (chaotic orbit, budget exhausted)" },
        { linCol(140,  51, 166), "escaped, shaded by winding count" },
      };
      for (int i = 0; i < 4; i++) {
        dl->AddRectFilled(ImVec2(o.x, o.y + i * 20), ImVec2(o.x + 14, o.y + 14 + i * 20), brows[i].c);
        dl->AddText(ImVec2(o.x + 20, o.y + i * 20), linCol(235, 235, 235), brows[i].t);
      }
      dl->AddText(ImVec2(12, winScaleY - 116), linCol(255, 255, 255, 220),
                  "Capture basins: the boundary between them is fractal");
    } else if (g_sim.lensMode == LENS_RING_ORDER) {
      ImVec2 o(12, winScaleY - 116);
      dl->AddRectFilled(ImVec2(o.x - 6, o.y - 6), ImVec2(o.x + 330, o.y + 106), linCol(0, 0, 0, 160), 6.0f);
      struct { ImU32 c; const char* t; } rows[] = {
        { linCol(230, 158,  20, 255), "n = 0  direct disk image" },
        { linCol( 26, 191, 217, 255), "n = 1  lensed (far side / underside)" },
        { linCol(204,  51, 217, 255), "n = 2  photon-ring image" },
        { linCol(242,  26,  38, 255), "n >= 3" },
        { linCol( 89, 107, 140, 255), "absorbed in foreground (jet/glow)" },
      };
      for (int i = 0; i < 5; i++) {
        dl->AddRectFilled(ImVec2(o.x, o.y + i * 20), ImVec2(o.x + 14, o.y + 14 + i * 20), rows[i].c);
        dl->AddText(ImVec2(o.x + 20, o.y + i * 20), linCol(235, 235, 235, 255), rows[i].t);
      }
      dl->AddText(ImVec2(12, winScaleY - 136), linCol(255, 255, 255, 220),
                  "Image order = number of equatorial crossings before the disk hit");
    } else if (g_sim.lensMode == LENS_REDSHIFT) {
      ImVec2 o(12, winScaleY - 64);
      dl->AddRectFilled(ImVec2(o.x - 6, o.y - 26), ImVec2(o.x + 276, o.y + 30), linCol(0, 0, 0, 160), 6.0f);
      for (int i = 0; i < 256; i++) {
        float t = i / 255.0f;
        float g = 0.4f + 1.2f * t;
        ImU32 c;
        if (g < 1.0f) { float u = (g - 0.4f) / 0.6f; c = linCol((int)(217 + (242 - 217) * u), (int)(26 + (242 - 26) * u), (int)(23 + (242 - 23) * u), 255); }
        else          { float u = (g - 1.0f) / 0.6f; c = linCol((int)(242 - (242 - 33) * u), (int)(242 - (242 - 77) * u), (int)(242 - (242 - 227) * u), 255); }
        dl->AddRectFilled(ImVec2(o.x + i, o.y), ImVec2(o.x + i + 1, o.y + 14), c);
      }
      dl->AddText(ImVec2(o.x, o.y - 20), linCol(255, 255, 255, 220), "g = E_obs / E_emit (redshift factor at first disk hit)");
      dl->AddText(ImVec2(o.x - 2, o.y + 14), linCol(235, 235, 235, 255), "0.4");
      dl->AddText(ImVec2(o.x + 120, o.y + 14), linCol(235, 235, 235, 255), "1.0");
      dl->AddText(ImVec2(o.x + 244, o.y + 14), linCol(235, 235, 235, 255), "1.6");
    } else if (g_sim.lensMode == LENS_CHECKER) {
      dl->AddText(ImVec2(12, winScaleY - 40), linCol(255, 255, 255, 220),
                  "Checkerboard sky: each repeated red patch is another image order of the point behind the camera");
    } else if (g_sim.lensMode == LENS_EHT) {
      dl->AddText(ImVec2(12, winScaleY - 40), linCol(255, 220, 160, 230),
                  "EHT view: image convolved with the telescope restoring beam (beam FWHM slider)");
      // Beam-size glyph, radio-paper style.
      float d_rs = glm::length(camera.position()) / (float)SagA_rs;
      float thf = tanf(radians(30.0f));
      float pxPerRad = winScaleY * 0.5f / thf;
      float beamPx = (g_sim.ehtBeamFwhm / d_rs) * pxPerRad * 0.5f;
      dl->AddCircle(ImVec2(40, winScaleY - 80), std::max(beamPx, 2.0f), linCol(255, 255, 255, 180), 32, 1.5f);
    }

    if (g_sim.ovGeodesicFan) {
      if (fanSpin != spin || fanCharge != charge) {
        fanRays = compute_geodesic_fan(spin, charge);
        fanSpin = spin; fanCharge = charge;
      }
      ImGui::SetNextWindowPos(ImVec2(winScaleX - 360, 30), ImGuiCond_FirstUseEver);
      ImGui::SetNextWindowSize(ImVec2(340, 400), ImGuiCond_FirstUseEver);
      ImGui::Begin("Geodesic Fan (equatorial plane)");
      ImVec2 p0 = ImGui::GetCursorScreenPos();
      ImVec2 avail = ImGui::GetContentRegionAvail();
      float side = std::min(avail.x, avail.y);
      ImDrawList* fdl = ImGui::GetWindowDrawList();
      auto toScreen = [&](ImVec2 w) {
        return ImVec2(p0.x + (w.x + 16.0f) / 32.0f * side,
                      p0.y + (16.0f - w.y) / 32.0f * side);
      };
      fdl->AddRectFilled(p0, ImVec2(p0.x + side, p0.y + side), linCol(8, 8, 14, 255));
      // Reference circles: horizon (filled), photon orbit, ISCO.
      ImVec2 c = toScreen(ImVec2(0, 0));
      float pxPerRs = side / 32.0f;
      fdl->AddCircleFilled(c, kr.r_horizon * pxPerRs, linCol(30, 30, 34, 255), 48);
      fdl->AddCircle(c, kr.r_photon * pxPerRs, linCol(255, 210, 80, 160), 48, 1.0f);
      fdl->AddCircle(c, kr.r_isco * pxPerRs, linCol(90, 220, 140, 160), 48, 1.0f);
      for (auto& ray : fanRays) {
        ImU32 col = ray.fate == 1 ? linCol(242, 80, 60, 200) : linCol(110, 190, 255, 170);
        for (size_t i = 1; i < ray.pts.size(); i++) {
          fdl->AddLine(toScreen(ray.pts[i - 1]), toScreen(ray.pts[i]), col, 1.0f);
        }
      }
      ImGui::Dummy(ImVec2(side, side));
      ImGui::TextWrapped("Parallel light rays (left to right). Red: captured. Blue: deflected and escaped. Yellow circle: photon orbit; green: ISCO.");
      ImGui::End();
    }
  }

  void drawControlPanel(const KerrRadii& kr, float spin_c, float charge_c) {
    ImGui::SetNextWindowSize(ImVec2(390, 680), ImGuiCond_FirstUseEver);
    ImGui::Begin("Geodesic Engine");

    if (ImGui::CollapsingHeader("Presets", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextWrapped("Quick configurations for common spacetime geometries.");
        auto physicsBase = [&]() {
            g_sim.enBloom = false; g_sim.enFlare = false; g_sim.enMotionBlur = false; g_sim.enFilmGrain = false;
            g_sim.enNebula = false; g_sim.enScintillation = false; g_sim.enVignette = false;
            g_sim.enGrid = false; g_sim.photonRingBoost = 0.0f;
            g_sim.lensMode = LENS_STANDARD; g_sim.beamingMode = BEAM_FULL;
            g_sim.diskModel = DISK_THIN; g_sim.bhMode = BH_KERR_NEWMAN;
        };
        if (ImGui::Button("Schwarzschild")) {
            physicsBase();
            g_sim.blackHoleSpin = 0.0f; g_sim.blackHoleCharge = 0.0f;
            g_sim.enCharge = false; g_sim.enJets = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Kerr")) {
            physicsBase();
            g_sim.blackHoleSpin = 0.7f; g_sim.blackHoleCharge = 0.0f;
            g_sim.enCharge = false; g_sim.enJets = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Extreme Kerr")) {
            physicsBase();
            g_sim.blackHoleSpin = 0.998f; g_sim.blackHoleCharge = 0.0f;
            g_sim.enCharge = false; g_sim.enJets = true;
        }
        if (ImGui::Button("Charged (RN)")) {
            physicsBase();
            g_sim.blackHoleSpin = 0.0f; g_sim.blackHoleCharge = 0.5f;
            g_sim.enCharge = true; g_sim.enJets = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Kerr-Newman")) {
            physicsBase();
            g_sim.blackHoleSpin = 0.6f; g_sim.blackHoleCharge = 0.3f;
            g_sim.enCharge = true; g_sim.enJets = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Cinematic")) {
            g_sim.blackHoleSpin = 0.85f; g_sim.blackHoleCharge = 0.0f;
            g_sim.enCharge = false; g_sim.enJets = true;
            g_sim.enBloom = true; g_sim.enFlare = true; g_sim.enMotionBlur = true; g_sim.enFilmGrain = true;
            g_sim.enNebula = true; g_sim.enScintillation = true; g_sim.enVignette = true;
            g_sim.enGrid = true;
            g_sim.photonRingBoost = 2.0f;
            g_sim.lensMode = LENS_STANDARD; g_sim.beamingMode = BEAM_FULL;
        }
        if (ImGui::Button("EHT M87* view")) {
            physicsBase();
            g_sim.blackHoleSpin = 0.9f; g_sim.blackHoleCharge = 0.0f;
            g_sim.enCharge = false; g_sim.enJets = false;
            g_sim.lensMode = LENS_EHT; g_sim.ehtBeamFwhm = 2.6f;
            camera.elevation = 0.30f;   // ~17 deg inclination (EHT M87*)
        }
        ImGui::SameLine();
        if (ImGui::Button("MP binary")) {
            physicsBase();
            g_sim.bhMode = BH_MP_BINARY;
            g_sim.mpM1 = 0.25f; g_sim.mpM2 = 0.25f; g_sim.mpSep = 6.0f;
            g_sim.mpGlow = 0.5f; g_sim.enNebula = true; g_sim.nebulaIntensity = 1.0f;
            // View down the axis perpendicular to the separation, so both
            // holes are equidistant and the eyebrow structure is symmetric.
            camera.radius = (float)(SagA_rs * 14.0);
            camera.elevation = 1.5708f;
            camera.azimuth = 1.5708f;
        }
        ImGui::SameLine();
        if (ImGui::Button("Volumetric torus")) {
            physicsBase();
            g_sim.blackHoleSpin = 0.9f; g_sim.blackHoleCharge = 0.0f;
            g_sim.enCharge = false; g_sim.enJets = false;
            g_sim.diskModel = DISK_VOLUMETRIC;
            g_sim.torusR0 = 7.0f; g_sim.torusH = 6.0f; g_sim.torusL0 = 1.0f;
            g_sim.torusAlpha = 0.0f; g_sim.torusAbsorb = 1.0f;
            g_sim.torusDensity = 1.0f; g_sim.torusTemp = 6500.0f;
            camera.elevation = 1.15f;
        }
        ImGui::SameLine();
        if (ImGui::Button("Luminet 1979")) {
            physicsBase();
            g_sim.blackHoleSpin = 0.0f; g_sim.blackHoleCharge = 0.0f;
            g_sim.enCharge = false; g_sim.enJets = false;
            g_sim.diskModel = DISK_THIN;
            camera.elevation = 1.40f;   // near edge-on, the classic view
        }
    }

    if (ImGui::CollapsingHeader("General Relativity", ImGuiTreeNodeFlags_DefaultOpen)) {
        const char* spacetimes[] = { "Kerr-Newman (single hole)",
                                     "Majumdar-Papapetrou binary" };
        ImGui::Combo("Spacetime", &g_sim.bhMode, spacetimes, 2);
        if (g_sim.bhMode == BH_MP_BINARY) {
            ImGui::TextWrapped("Two extremally-charged holes in EXACT static equilibrium "
                               "(gravity balanced by electrostatic repulsion). No Carter "
                               "constant exists here: the capture basin is a chaotic "
                               "scatterer with a fractal boundary and self-similar "
                               "'eyebrows' facing each companion. Use the Image-order lens "
                               "to see the basin structure, or the Checkerboard sky for the lensing.");
            ImGui::SliderFloat("Mass 1", &g_sim.mpM1, 0.02f, 1.0f);
            ImGui::SliderFloat("Mass 2", &g_sim.mpM2, 0.0f, 1.0f);
            ImGui::SliderFloat("Separation", &g_sim.mpSep, 0.0f, 30.0f);
            ImGui::SliderFloat("Horizon Glow", &g_sim.mpGlow, 0.0f, 2.0f);
            ImGui::TextDisabled("  each hole: horizon (areal) = m, shadow b = 4m");
            ImGui::Separator();
        }
        ImGui::TextWrapped("Kerr-Newman metric parameters. Spin < 0 renders a retrograde disk (retrograde ISCO applies).");
        ImGui::SliderFloat("Black Hole Spin (a)", &g_sim.blackHoleSpin, -1.0f, 1.0f);
        ImGui::SliderFloat("Electric Charge (Q)", &g_sim.blackHoleCharge, 0.0f, 1.0f);
        ImGui::SameLine(); ImGui::Checkbox("##enCharge", &g_sim.enCharge);
        if (g_sim.enCharge && charge_c < g_sim.blackHoleCharge - 1e-4f)
            ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                               "  clamped to Q = %.3f (a^2 + Q^2 <= 1)", charge_c);
        ImGui::SliderFloat("Simulation Speed", &g_sim.timeScale, 0.0f, 5.0f);
        ImGui::Checkbox("N-Body Gravitation", &g_sim.Gravity);
    }

    if (ImGui::CollapsingHeader("Accretion Flow", ImGuiTreeNodeFlags_DefaultOpen)) {
        const char* models[] = { "Thin disk (Novikov-Thorne surface)",
                                 "Volumetric torus (GRMHD-style)" };
        ImGui::Combo("Model", &g_sim.diskModel, models, 2);
        if (g_sim.diskModel == DISK_THIN) {
            ImGui::TextWrapped("Optically-thick surface at the equator: Page-Thorne flux (zero at ISCO, peak at 49/36 r_in), blackbody color with the full g-factor temperature shift.");
            ImGui::SliderFloat("Surface Brightness", &g_sim.diskDensity, 0.0f, 1.0f);
            ImGui::SameLine(); ImGui::Checkbox("##enDisk", &g_sim.enDisk);
            ImGui::SliderFloat("Outer Radius (rs)", &g_sim.diskOuterRadius, 4.0f, 24.0f);
            ImGui::SliderFloat("Peak Temp (K)", &g_sim.diskTempK, 1000.0f, 20000.0f);
        } else {
            ImGui::TextWrapped("Optically-thin plasma torus integrated with the covariant radiative-transfer equation d(I/nu^3)/dlambda = j/nu^2 - (nu alpha) I/nu^3. Geometry and rotation law follow the EHT code-comparison parameterization (Gold et al. 2020).");
            ImGui::SliderFloat("Emission Scale", &g_sim.torusDensity, 0.0f, 4.0f);
            ImGui::SliderFloat("Ring Radius (M)", &g_sim.torusR0, 2.0f, 30.0f);
            ImGui::SliderFloat("Vertical Compression", &g_sim.torusH, 0.0f, 12.0f);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                "0 = spherical envelope; larger values flatten it into a torus.");
            ImGui::SliderFloat("Angular Momentum l0", &g_sim.torusL0, 0.0f, 6.0f);
            ImGui::SliderFloat("Spectral Index", &g_sim.torusAlpha, -3.0f, 3.0f);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                "j_nu ~ nu^-alpha. At alpha = -2 the Doppler boost cancels\n"
                "exactly against the frequency dependence and the image\n"
                "becomes symmetric despite relativistic rotation\n"
                "(the Gold et al. 2020 Test 2 identity).");
            ImGui::SliderFloat("Absorption", &g_sim.torusAbsorb, 0.0f, 8.0f);
            ImGui::SliderFloat("Color Temp (K)", &g_sim.torusTemp, 1000.0f, 20000.0f);
        }
    }

    if (ImGui::CollapsingHeader("Learning", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextWrapped("Alternative visualizations of the same spacetime.");
        const char* lenses[] = { "Standard (photographic)", "Image order (photon rings)",
                                 "Redshift map (g-factor)", "Checkerboard sky (lensing)",
                                 "EHT view (beam-blurred)" };
        ImGui::Combo("Lens", &g_sim.lensMode, lenses, 5);
        const char* projs[] = { "Perspective", "360 equirectangular (VR)", "Mollweide all-sky" };
        ImGui::Combo("Projection", &g_sim.projection, projs, 3);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            "Ray tracers get these almost free: only the pixel -> direction\n"
            "mapping changes. Render 360 at 2:1 for spherical-video players.");
        if (g_sim.lensMode == LENS_EHT)
            ImGui::SliderFloat("Beam FWHM (rs)", &g_sim.ehtBeamFwhm, 0.5f, 8.0f);
        const char* beams[] = { "Full (g^4 + color shift)", "Color shift only", "Off (bolometric)" };
        ImGui::Combo("Doppler/beaming", &g_sim.beamingMode, beams, 3);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            "Isolates the relativistic brightness asymmetry:\n"
            "Full = physical; Color shift = Interstellar convention;\n"
            "Off = Luminet-style bolometric image.");
        ImGui::Checkbox("Orbit markers", &g_sim.ovMarkers);
        ImGui::SameLine();
        ImGui::Checkbox("Critical curve", &g_sim.ovCriticalCurve);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            "Bardeen's analytic shadow boundary. The rendered shadow edge\n"
            "must land exactly on this curve — it is also a live validation.");
        ImGui::Checkbox("Geodesic fan", &g_sim.ovGeodesicFan);
        ImGui::SameLine();
        ImGui::Checkbox("Spacetime grid", &g_sim.enGrid);
        ImGui::SliderFloat("Photon Ring Boost", &g_sim.photonRingBoost, 0.0f, 4.0f);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            "Amplifies higher-order (n >= 1) disk images.\n"
            "0 = physical: real photon rings are demagnified ~e^-pi per order.");
    }

    if (ImGui::CollapsingHeader("Spacetime Effects")) {
        ImGui::SliderFloat("GW Amplitude", &g_sim.gwAmplitude, 0.0f, 2.0f);
        ImGui::SameLine(); ImGui::Checkbox("##enGW", &g_sim.enGW);
        ImGui::SliderFloat("Jet Intensity", &g_sim.jetIntensity, 0.0f, 2.0f);
        ImGui::SameLine(); ImGui::Checkbox("##enJets", &g_sim.enJets);
        ImGui::Checkbox("Companion stars", &g_sim.enStarBodies);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            "N-body stars rendered with occlusion + gravitational lensing.");
    }

    if (ImGui::CollapsingHeader("Cinematic Effects")) {
        ImGui::TextWrapped("Non-physical visual effects for cinematic rendering.");
        ImGui::SliderFloat("Nebula Intensity", &g_sim.nebulaIntensity, 0.0f, 2.0f);
        ImGui::SameLine(); ImGui::Checkbox("##enNebula", &g_sim.enNebula);
        ImGui::SliderFloat("Star Scintillation", &g_sim.starScintillation, 0.0f, 1.0f);
        ImGui::SameLine(); ImGui::Checkbox("##enScint", &g_sim.enScintillation);
        ImGui::SliderFloat("Bloom Threshold", &g_sim.bloomThreshold, 0.0f, 1.0f);
        ImGui::SameLine(); ImGui::Checkbox("##enBloom", &g_sim.enBloom);
        ImGui::SliderFloat("Anamorphic Flare", &g_sim.flareIntensity, 0.0f, 2.0f);
        ImGui::SameLine(); ImGui::Checkbox("##enFlare", &g_sim.enFlare);
        ImGui::SliderFloat("Motion Blur", &g_sim.motionBlur, 0.0f, 0.99f);
        ImGui::SameLine(); ImGui::Checkbox("##enMBlur", &g_sim.enMotionBlur);
        ImGui::SliderFloat("Film Grain", &g_sim.filmGrain, 0.0f, 0.1f);
        ImGui::SameLine(); ImGui::Checkbox("##enGrain", &g_sim.enFilmGrain);
        ImGui::SliderFloat("Vignette", &g_sim.vignetteIntensity, 0.0f, 1.0f);
        ImGui::SameLine(); ImGui::Checkbox("##enVig", &g_sim.enVignette);
        ImGui::Checkbox("Extended dynamic range (HDR)", &g_sim.enEDR);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
            "Lets the tonemap shoulder run past reference white so the\n"
            "Doppler-boosted inner limb glares on an XDR display.");
        ImGui::Text("  display headroom: %.2fx %s", edrHeadroom,
                    edrHeadroom > 1.05f ? "(EDR active)" : "(SDR)");
    }

    ImGui::Separator();
    {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f),
                           "Kerr-Newman (a = %.3f, Q = %.3f)", spin_c, charge_c);
        ImGui::Text("  Event Horizon:   %.3f rs", kr.r_horizon);
        ImGui::Text("  Photon Orbit:    %.3f rs %s", kr.r_photon,
                    spin_c < 0.0f ? "(retrograde)" : "");
        ImGui::Text("  ISCO:            %.3f rs (%s)", kr.r_isco,
                    spin_c < 0.0f ? "retrograde" : "prograde");
        if (std::abs(spin_c) > 0.01f)
            ImGui::Text("  Ergosphere:      %.3f -> 1.000 rs (pole -> eq.)", kr.r_horizon);
        // Telemetry: camera state in physical terms.
        float d_rs = glm::length(camera.position()) / (float)SagA_rs;
        float dil = std::sqrt(std::max(1.0f - 1.0f / std::max(d_rs, 1.001f), 0.0f));
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.6f, 0.85f, 1.0f, 1.0f), "Camera");
        ImGui::Text("  r = %.1f rs   static time dilation dtau/dt = %.4f", d_rs, dil);
        ImGui::Text("  Accumulated frames: %d%s", accumN,
                    accumN >= 32 ? " (supersampled)" : "");
    }
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "Auto-Exposure: %.2f", currentExposure);
    ImGui::Text("GPU Compute: %.1f FPS", ImGui::GetIO().Framerate);
    ImGui::End();
  }

  // Offline reference-still export: render at arbitrary resolution and
  // projection with N jittered accumulated samples, then write a scene-linear
  // EXR plus a tonemapped PNG-able PPM. Independent of the live render
  // targets, so a 360 panorama can be 2:1 regardless of the window.
  void servicePendingExports() {
    if (panoPending) {
      panoPending = false;
      int W = 4096, H = 2048;                    // 2:1, ~5.3 arcmin/pixel
      if (const char* e = getenv("BH_PANO_W")) { W = atoi(e); H = W / 2; }
      exportStill(W, H, 128, PROJ_EQUIRECT);
    }
    if (stillPending) {
      stillPending = false;
      int W = 3840, H = 2160;
      if (const char* e = getenv("BH_STILL_W")) { W = atoi(e); H = W * 9 / 16; }
      exportStill(W, H, 256, g_sim.projection);
    }
  }

  void exportStill(int W, int H, int samples, int projection) {
    // Spherical projections are 2:1 by definition — a 16:9 equirect is
    // stretched by every player, and the Mollweide ellipse test stops
    // matching the frame. Enforce it here so every caller is covered.
    if (projection != PROJ_PERSPECTIVE) H = W / 2;
    // BUG 3: Metal aborts (assertion, not a nil return) above 16384 texels.
    W = std::min(W, 16384); H = std::min(H, 16384);
    printf("Reference still: %dx%d, %d samples, projection %d ...\n",
           W, H, samples, projection);
    MTLTextureDescriptor *td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                     width:W height:H mipmapped:NO];
    td.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModePrivate;
    id<MTLTexture> rt = [device newTextureWithDescriptor:td];
    id<MTLTexture> acc[2] = { [device newTextureWithDescriptor:td],
                              [device newTextureWithDescriptor:td] };
    id<MTLBuffer> camB = [device newBufferWithLength:sizeof(CameraData) options:MTLResourceStorageModeShared];
    id<MTLBuffer> sysB = [device newBufferWithLength:sizeof(SystemUniforms) options:MTLResourceStorageModeShared];
    id<MTLBuffer> objB = [device newBufferWithLength:sizeof(ObjectsUniform) options:MTLResourceStorageModeShared];
    memcpy(camB.contents, camBuffer[currentFrame].contents, sizeof(CameraData));
    memcpy(sysB.contents, sysUniformBuffer[currentFrame].contents, sizeof(SystemUniforms));
    memcpy(objB.contents, objUniformBuffer[currentFrame].contents, sizeof(ObjectsUniform));
    CameraData *cp = (CameraData *)camB.contents;
    cp->aspect = float(W) / float(H);
    if (projection != PROJ_PERSPECTIVE) {
      // Level the horizon. A 360 viewer cannot roll or pitch their head to
      // compensate for a tilted projection — a pitched horizon baked into a
      // panorama is the classic vection-discomfort trigger — so lock the pole
      // to the spin axis and put the image centre on the horizontal.
      glm::vec3 pole(0.0f, 1.0f, 0.0f);
      glm::vec3 f = glm::vec3(cp->camForward);
      glm::vec3 fh = f - pole * glm::dot(f, pole);
      float L = glm::length(fh);
      glm::vec3 fwdP = (L > 1e-4f) ? fh / L : glm::vec3(1.0f, 0.0f, 0.0f);
      glm::vec3 rgtP = glm::normalize(glm::cross(fwdP, pole));
      glm::vec3 upP  = glm::cross(rgtP, fwdP);          // == pole exactly
      cp->camForward = glm::vec4(fwdP, 0.0f);
      cp->camRight   = glm::vec4(rgtP, 0.0f);
      cp->camUp      = glm::vec4(upP, 0.0f);
    }
    SystemUniforms *sp = (SystemUniforms *)sysB.contents;
    sp->projection = projection;

    int cur = 0;
    for (int i = 0; i < samples; i++) {
      sp->jitter_x = halton(i + 1, 2) - 0.5f;
      sp->jitter_y = halton(i + 1, 3) - 0.5f;
      sp->accum_alpha = (i == 0) ? 1.0f : 1.0f / float(i + 1);
      @autoreleasepool {
        id<MTLCommandBuffer> cb = [commandQueue commandBuffer];
        MTLSize tpg = MTLSizeMake(32, 8, 1);
        id<MTLComputeCommandEncoder> ray = [cb computeCommandEncoder];
        [ray setComputePipelineState:raytracePSO];
        [ray setTexture:rt atIndex:0];
        [ray setBuffer:camB offset:0 atIndex:0];
        [ray setBuffer:objBuffer offset:0 atIndex:1];
        [ray setBuffer:objB offset:0 atIndex:2];
        [ray setBuffer:sysB offset:0 atIndex:3];
        [ray dispatchThreads:MTLSizeMake(W, H, 1) threadsPerThreadgroup:tpg];
        [ray endEncoding];
        id<MTLComputeCommandEncoder> ta = [cb computeCommandEncoder];
        [ta setComputePipelineState:temporalPSO];
        [ta setTexture:rt atIndex:0];
        [ta setTexture:acc[1 - cur] atIndex:1];
        [ta setTexture:acc[cur] atIndex:2];
        [ta setBuffer:sysB offset:0 atIndex:0];
        [ta dispatchThreads:MTLSizeMake(W, H, 1) threadsPerThreadgroup:tpg];
        [ta endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
      }
      cur = 1 - cur;
      if ((i + 1) % 32 == 0) { printf("  %d/%d\r", i + 1, samples); fflush(stdout); }
    }
    NSUInteger bpr = W * 8;
    id<MTLBuffer> rb = [device newBufferWithLength:bpr * H options:MTLResourceStorageModeShared];
    @autoreleasepool {
      id<MTLCommandBuffer> cb = [commandQueue commandBuffer];
      id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
      [bl copyFromTexture:acc[1 - cur] sourceSlice:0 sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(W, H, 1)
                 toBuffer:rb destinationOffset:0
      destinationBytesPerRow:bpr destinationBytesPerImage:bpr * H];
      [bl endEncoding];
      [cb commit]; [cb waitUntilCompleted];
    }
    writeEXR(rb, W, H, bpr, samples - 1, projection);
  }

  void resize(int w, int h) {
    drawableW = w; drawableH = h;
    metalLayer.drawableSize = CGSizeMake(drawableW, drawableH);
    rtTex = nil;
    accumTex[0] = nil;
    accumTex[1] = nil;
    bloomTex = nil;
    bloomBlurTex = nil;
    depthTexture = nil;
    createResources();
  }
};

void mouseCallback(GLFWwindow *w, int btn, int act, int mods) {
  if (qaMode) return;   // deterministic captures: ignore live input
  // Releases must ALWAYS clear the drag state, even over the ImGui panel —
  // otherwise a release there leaves the camera stuck in drag mode.
  if (btn == GLFW_MOUSE_BUTTON_LEFT && act == GLFW_RELEASE) {
    camera.dragging = false; camera.panning = false;
    return;
  }
  if (ImGui::GetIO().WantCaptureMouse) return;
  if (btn == GLFW_MOUSE_BUTTON_LEFT && act == GLFW_PRESS) {
    glfwGetCursorPos(w, &camera.lastX, &camera.lastY);
    camera.panning = (mods & GLFW_MOD_SHIFT);
    camera.dragging = !camera.panning;
  }
}

void cursorCallback(GLFWwindow * /*w*/, double x, double y) {
  if (qaMode) return;
  float dx = float(x - camera.lastX); float dy = float(y - camera.lastY);
  camera.lastX = x; camera.lastY = y;    // keep tracking even over the panel
  if (ImGui::GetIO().WantCaptureMouse && !camera.dragging && !camera.panning) return;
  if (camera.dragging) {
    camera.azimuth -= dx * 0.005f; camera.elevation -= dy * 0.005f;
    camera.elevation = std::max(0.05f, std::min(camera.elevation, PI_F - 0.05f));
  }
  if (camera.panning) {
    vec3 fwd = normalize(camera.target - camera.position());
    vec3 right = normalize(cross(fwd, vec3(0, 1, 0)));
    vec3 up = cross(right, fwd);
    camera.target += (-right * dx + up * dy) * (camera.radius * 0.002f);
  }
}

void scrollCallback(GLFWwindow * /*w*/, double /*xoff*/, double yoff) {
  if (qaMode) return;
  if (ImGui::GetIO().WantCaptureMouse) return;
  camera.radius *= (1.0f - float(yoff) * 0.1f);
  const float minR = (float)(SagA_rs * 2.5);
  const float maxR = 5.0e14f;
  camera.radius = std::clamp(camera.radius, minR, maxR);
}

void keyCallback(GLFWwindow *w, int key, int /*scancode*/, int action, int /*mods*/) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) glfwSetWindowShouldClose(w, GLFW_TRUE);
  if (ImGui::GetIO().WantCaptureKeyboard) return;
  if (key == GLFW_KEY_P && action == GLFW_PRESS) screenshotPending = true;
  if (key == GLFW_KEY_E && action == GLFW_PRESS) exrPending = true;
  if (key == GLFW_KEY_3 && action == GLFW_PRESS) panoPending = true;
  if (key == GLFW_KEY_R && action == GLFW_PRESS) stillPending = true;
  // L cycles the learning lens.
  if (key == GLFW_KEY_L && action == GLFW_PRESS) g_sim.lensMode = (g_sim.lensMode + 1) % 5;
}

int main() {
  if (!glfwInit()) return -1;
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow *window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Metal Blackhole", nullptr, nullptr);
  if (!window) { std::cerr << "Failed to create GLFW window" << std::endl; glfwTerminate(); return 1; }
  glfwSetMouseButtonCallback(window, mouseCallback);
  glfwSetCursorPosCallback(window, cursorCallback);
  glfwSetScrollCallback(window, scrollCallback);
  glfwSetKeyCallback(window, keyCallback);

  // QA harness (headless-ish visual regression): BH_QA=1 captures frame 90,
  // exits at frame 100. BH_ELEV/BH_AZIM/BH_SPIN/BH_CHARGE/BH_LENS override state.
  qaMode = getenv("BH_QA") != nullptr;
  if (const char* e = getenv("BH_ELEV"))   camera.elevation = std::clamp((float)atof(e), 0.05f, PI_F - 0.05f);
  if (const char* e = getenv("BH_AZIM"))   camera.azimuth = (float)atof(e);
  if (const char* e = getenv("BH_SPIN"))   g_sim.blackHoleSpin = std::clamp((float)atof(e), -1.0f, 1.0f);
  if (const char* e = getenv("BH_CHARGE")) { g_sim.blackHoleCharge = std::clamp((float)atof(e), 0.0f, 1.0f); g_sim.enCharge = g_sim.blackHoleCharge > 0.0f; }
  if (const char* e = getenv("BH_LENS"))   g_sim.lensMode = std::clamp(atoi(e), 0, 4);
  if (const char* e = getenv("BH_BEAMING"))g_sim.beamingMode = std::clamp(atoi(e), 0, 2);
  if (getenv("BH_JETS")) g_sim.enJets = true;
  if (const char* e = getenv("BH_MODEL")) g_sim.diskModel = std::clamp(atoi(e), 0, 1);
  if (const char* e = getenv("BH_ALPHA")) g_sim.torusAlpha = (float)atof(e);
  if (const char* e = getenv("BH_ABSORB")) g_sim.torusAbsorb = (float)atof(e);
  if (getenv("BH_NOEDR")) g_sim.enEDR = false;
  if (const char* e = getenv("BH_BINARY")) {
    g_sim.bhMode = BH_MP_BINARY; g_sim.mpSep = (float)atof(e);
    g_sim.enNebula = true;
  }
  if (const char* e = getenv("BH_M2")) g_sim.mpM2 = (float)atof(e);
  if (const char* e = getenv("BH_PROJ")) g_sim.projection = std::clamp(atoi(e), 0, 2);
  if (const char* e = getenv("BH_RADIUS")) camera.radius = (float)(SagA_rs * atof(e));

  if (const char* e = getenv("BH_OVERLAYS")) {   // any of: m(arkers) c(ritical) f(an) g(rid)
    g_sim.ovMarkers       = strchr(e, 'm') != nullptr;
    g_sim.ovCriticalCurve = strchr(e, 'c') != nullptr;
    g_sim.ovGeodesicFan   = strchr(e, 'f') != nullptr;
    g_sim.enGrid          = strchr(e, 'g') != nullptr;
  }

  vector<SimObject> objects;
  const float GM = (float)(G_const * SagA_mass);

  SimObject bh = {}; bh.posRadius = vec4(0, 0, 0, (float)SagA_rs); bh.color = vec4(0,0,0,1); bh.mass = (float)SagA_mass; bh.velocity = vec4(0,0,0,0);

  // Close binary pair
  SimObject s1 = {}; s1.posRadius = vec4(0, 0, 8.0e12f, 1.2e12f); s1.color = vec4(1.0f, 0.8f, 0.3f, 1); s1.mass = 2.9e30f;
  s1.velocity = vec4(sqrt(GM / 8.0e12f), 0, 0, 0);
  SimObject s2 = {}; s2.posRadius = vec4(0, 0, -8.0e12f, 1.2e12f); s2.color = vec4(0.4f, 0.7f, 1.0f, 1); s2.mass = 2.9e30f;
  s2.velocity = vec4(-sqrt(GM / 8.0e12f), 0, 0, 0);

  // Distant stars — lensed via the exit-direction intersection
  float r4 = 80.0e12f; float v4 = sqrt(GM / r4);
  SimObject s4 = {}; s4.posRadius = vec4(r4, 0, 0, 3.0e12f); s4.color = vec4(0.5f, 0.7f, 1.0f, 1); s4.mass = 12.0e30f;
  s4.velocity = vec4(0, 0, v4, 0);

  float r5 = 120.0e12f; float v5 = sqrt(GM / r5);
  SimObject s5 = {}; s5.posRadius = vec4(-r5 * 0.7f, 5.0e12f, r5 * 0.7f, 4.0e12f); s5.color = vec4(1.0f, 0.4f, 0.15f, 1); s5.mass = 5.0e30f;
  s5.velocity = vec4(-v5 * 0.7f, 0, -v5 * 0.7f, 0);

  float r6 = 60.0e12f; float v6 = sqrt(GM / r6);
  SimObject s6 = {}; s6.posRadius = vec4(0, 15.0e12f, -r6, 2.0e12f); s6.color = vec4(0.85f, 0.9f, 1.0f, 1); s6.mass = 6.0e30f;
  s6.velocity = vec4(v6, 0, 0, 0);

  float r7 = 150.0e12f; float v7 = sqrt(GM / r7);
  SimObject s7 = {}; s7.posRadius = vec4(r7 * 0.5f, 0, r7 * 0.866f, 3.5e12f); s7.color = vec4(1.0f, 0.6f, 0.2f, 1); s7.mass = 4.0e30f;
  s7.velocity = vec4(-v7 * 0.866f, 0, v7 * 0.5f, 0);

  float r8 = 100.0e12f; float v8 = sqrt(GM / r8);
  SimObject s8 = {}; s8.posRadius = vec4(-r8, 0, 0, 2.5e12f); s8.color = vec4(0.7f, 0.4f, 1.0f, 1); s8.mass = 7.0e30f;
  s8.velocity = vec4(0, 0, v8, 0);

  objects.push_back(s1); objects.push_back(s2);
  objects.push_back(s4); objects.push_back(s5); objects.push_back(s6);
  objects.push_back(s7); objects.push_back(s8);
  objects.push_back(bh);
  MetalEngine engine(window, objects);
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    engine.render((int)objects.size());
    engine.servicePendingExports();
    if (qaMode) {
      if (frameCount == 90) {
        screenshotPending = true;
        if (getenv("BH_EXR")) exrPending = true;
        if (getenv("BH_PANO")) panoPending = true;
        if (getenv("BH_STILL")) stillPending = true;
      }
      if (frameCount >= 100) glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
  }
  glfwTerminate();
  return 0;
}
