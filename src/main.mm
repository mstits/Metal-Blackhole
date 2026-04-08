#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <QuartzCore/QuartzCore.h>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_metal.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace glm;
using namespace std;

// --- SHARED STRUCTS (single source of truth in ShaderCommon.h) ---
#include "ShaderCommon.h"

const int WINDOW_WIDTH = 1200;
const int WINDOW_HEIGHT = 800;

const double G_const = 6.67430e-11;
const double c_const = 299792458.0;
const double SagA_rs = 2.0 * G_const * 8.54e36 / (c_const * c_const);

const float PI_F = 3.14159265358979323846f;

// --- GLOBALS ---

bool Gravity = true;
float blackHoleSpin = 0.0f;       // Start Schwarzschild (simplest GR prediction)
float blackHoleCharge = 0.0f;
float simulationTime = 0.0f;
float timeScale = 1.0f;
float starScintillation = 0.5f;
float nebulaIntensity = 1.0f;
float bloomThreshold = 0.85f;
float flareIntensity = 0.5f;
float motionBlur = 0.85f;
float filmGrain = 0.02f;
float diskDensity = 0.45f;
float diskHeight = 0.6f;
float shadowIntensity = 0.5f;
float gwAmplitude = 0.5f;
float jetIntensity = 0.7f;
double lastFrameTime = 0.0;
float deltaTime = 0.016f;
int frameCount = 0;
bool screenshotPending = false;

// Feature toggles — physics mode by default (cinematic effects OFF)
bool enCharge = false;        // Off: start with pure Schwarzschild
bool enDisk = true;           // On: essential for seeing BH effects
bool enShadow = true;         // On: physical self-shadowing
bool enGW = true;             // On: educational (spacetime curvature)
bool enJets = false;          // Off: astrophysical, not fundamental GR
bool enNebula = false;        // Off: visual distraction
bool enScintillation = false; // Off: not GR physics
bool enBloom = false;         // Off: cinematic
bool enFlare = false;         // Off: cinematic
bool enMotionBlur = false;    // Off: cinematic
bool enFilmGrain = false;     // Off: cinematic
bool enVignette = false;      // Off: cinematic

#include "Camera.h"
Camera camera((float)(SagA_rs * 20.0));

class GridRenderer {
public:
  id<MTLRenderPipelineState> pipelineState;
  id<MTLDepthStencilState> depthState;
  id<MTLBuffer> vertexBuffer;
  id<MTLBuffer> indexBuffer;
  id<MTLBuffer> uniformBuffer;
  int indexCount = 0;

  GridRenderer(id<MTLDevice> device, id<MTLLibrary> library) {
    NSError *err = nil;
    MTLRenderPipelineDescriptor *pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = [library newFunctionWithName:@"grid_vertex"];
    pd.fragmentFunction = [library newFunctionWithName:@"grid_fragment"];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
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
    if (!pipelineState) { std::cerr << "Grid PSO Error: " << err.localizedDescription.UTF8String << std::endl; }
    
    MTLDepthStencilDescriptor *dd = [[MTLDepthStencilDescriptor alloc] init];
    dd.depthCompareFunction = MTLCompareFunctionLessEqual;
    dd.depthWriteEnabled = YES;
    depthState = [device newDepthStencilStateWithDescriptor:dd];

    uniformBuffer = [device newBufferWithLength:sizeof(GridUniforms) options:MTLResourceStorageModeShared];

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

class MetalEngine {
public:
  id<MTLDevice> device;
  id<MTLCommandQueue> commandQueue;
  id<MTLComputePipelineState> raytracePSO;
  id<MTLComputePipelineState> physicsPSO;
  id<MTLComputePipelineState> fluidPSO;
  id<MTLComputePipelineState> postPSO;
  id<MTLComputePipelineState> bloomExtractPSO;
  id<MTLComputePipelineState> lumReducePSO;

  id<MTLBuffer> camBuffer[3];
  id<MTLBuffer> objBuffer;          // Single buffer — GPU-only writes, no triple-buffering needed
  id<MTLBuffer> objUniformBuffer[3];
  id<MTLBuffer> sysUniformBuffer[3];
  id<MTLBuffer> gridUniformBuffer[3];
  id<MTLBuffer> lumBuffer[3];
  int currentFrame = 0;
  float currentExposure = 1.0f;

  id<MTLTexture> fluidTex[2];
  id<MTLTexture> intermediateTex[2];  // Double-buffered: eliminates blit copy
  id<MTLTexture> bloomTex;
  id<MTLTexture> bloomBlurTex;
  id<MTLTexture> depthTexture;
  int currentFluidIdx = 0;
  int currentIntermIdx = 0;

  MPSImageGaussianBlur *mpsBloom;

  CAMetalLayer *metalLayer;
  GridRenderer *gridRenderer;

  int drawableW, drawableH;
  GLFWwindow* winRef;
  dispatch_semaphore_t frameSemaphore;

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
    metalLayer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    metalLayer.framebufferOnly = NO;
    metalLayer.drawableSize = CGSizeMake(drawableW, drawableH);
    [view setLayer:metalLayer];
    [view setWantsLayer:YES];

    NSError *err = nil;
    NSString *execDir = [[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent];
    id<MTLLibrary> lib = nil;

    // Try precompiled .metallib first (faster startup, no GPU code injection risk)
    NSString *metallibPath = [execDir stringByAppendingPathComponent:@"geodesic.metallib"];
    if ([[NSFileManager defaultManager] fileExistsAtPath:metallibPath]) {
      lib = [device newLibraryWithURL:[NSURL fileURLWithPath:metallibPath] error:&err];
      if (lib) std::cout << "Loaded precompiled geodesic.metallib" << std::endl;
    }

    // Fallback: runtime source compilation (development mode)
    if (!lib) {
      NSString *headerPath = [execDir stringByAppendingPathComponent:@"ShaderCommon.h"];
      NSString *shaderPath = [execDir stringByAppendingPathComponent:@"geodesic.metal"];
      NSString *headerSrc = [NSString stringWithContentsOfFile:headerPath encoding:NSUTF8StringEncoding error:&err];
      if (!headerSrc) { std::cerr << "Cannot load ShaderCommon.h: " << err.localizedDescription.UTF8String << std::endl; exit(1); }
      NSString *shaderSrc = [NSString stringWithContentsOfFile:shaderPath encoding:NSUTF8StringEncoding error:&err];
      if (!shaderSrc) { std::cerr << "Cannot load geodesic.metal: " << err.localizedDescription.UTF8String << std::endl; exit(1); }
      NSString *fullSrc = [NSString stringWithFormat:@"%@\n%@", headerSrc, shaderSrc];
      MTLCompileOptions *compileOpts = [[MTLCompileOptions alloc] init];
      // INVARIANT: Must use MTLMathModeRelaxed, NOT MTLMathModeFast.
      // Fast math uses approximate sqrt/division which corrupts geodesic
      // integration, causing the accretion disk to disappear at edge-on angles.
      compileOpts.mathMode = MTLMathModeRelaxed;
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

    raytracePSO = createPSO("raytrace");
    physicsPSO = createPSO("update_physics");
    fluidPSO = createPSO("simulate_disk_fluid");
    postPSO = createPSO("post_process_suite");
    bloomExtractPSO = createPSO("bloom_extract");
    lumReducePSO = createPSO("luminance_reduce");

    gridRenderer = new GridRenderer(device, lib);
    mpsBloom = [[MPSImageGaussianBlur alloc] initWithDevice:device sigma:8.0f];

    // Object buffer is single (GPU physics writes; no CPU mutation after init)
    objBuffer = [device newBufferWithBytes:initialObjects.data() length:sizeof(SimObject) * initialObjects.size() options:MTLResourceStorageModeShared];

    for (int i = 0; i < 3; i++) {
      camBuffer[i] = [device newBufferWithLength:sizeof(CameraData) options:MTLResourceStorageModeShared];
      objUniformBuffer[i] = [device newBufferWithLength:sizeof(ObjectsUniform) options:MTLResourceStorageModeShared];
      sysUniformBuffer[i] = [device newBufferWithLength:sizeof(SystemUniforms) options:MTLResourceStorageModeShared];
      gridUniformBuffer[i] = [device newBufferWithLength:sizeof(GridUniforms) options:MTLResourceStorageModeShared];
      lumBuffer[i] = [device newBufferWithLength:sizeof(uint32_t) * 2 options:MTLResourceStorageModeShared];
      memset(lumBuffer[i].contents, 0, sizeof(uint32_t) * 2);
    }

    createResources();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOther(window, true);
    ImGui_ImplMetal_Init(device);
  }

  void createResources() {
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:1024 height:1024 mipmapped:NO];
    td.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModePrivate;
    fluidTex[0] = [device newTextureWithDescriptor:td];
    fluidTex[1] = [device newTextureWithDescriptor:td];

    MTLTextureDescriptor *itd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:drawableW height:drawableH mipmapped:NO];
    itd.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    itd.storageMode = MTLStorageModePrivate;
    intermediateTex[0] = [device newTextureWithDescriptor:itd];
    intermediateTex[1] = [device newTextureWithDescriptor:itd]; 

    // Half-res bloom textures
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
  }

  void render(int activeObjectCount) {
    dispatch_semaphore_wait(frameSemaphore, DISPATCH_TIME_FOREVER);

    int curW, curH;
    glfwGetFramebufferSize(winRef, &curW, &curH);
    if (curW != drawableW || curH != drawableH) {
      resize(curW, curH);
    }

    // FPS-independent timing
    double now = glfwGetTime();
    deltaTime = std::clamp(float(now - lastFrameTime), 0.0001f, 0.1f);  // cap at 100ms
    lastFrameTime = now;
    simulationTime += deltaTime * timeScale;
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

    ObjectsUniform *ouPtr = (ObjectsUniform *)objUniformBuffer[currentFrame].contents; ouPtr->count = activeObjectCount;
    SystemUniforms *sysPtr = (SystemUniforms *)sysUniformBuffer[currentFrame].contents;
    // SEC-04: Clamp all values to valid ranges at the GPU write site
    // Feature toggles: send 0 when disabled, preserving slider value for re-enable
    sysPtr->time = simulationTime; sysPtr->spin = std::clamp(blackHoleSpin, -1.0f, 1.0f);
    sysPtr->star_scint = enScintillation ? std::clamp(starScintillation, 0.0f, 1.0f) : 0.0f;
    sysPtr->nebula_int = enNebula ? std::clamp(nebulaIntensity, 0.0f, 2.0f) : 0.0f;
    sysPtr->charge = enCharge ? std::clamp(blackHoleCharge, 0.0f, 1.0f) : 0.0f;
    sysPtr->dt_sim = deltaTime * std::clamp(timeScale, 0.0f, 5.0f);
    sysPtr->bloom_threshold = enBloom ? std::clamp(bloomThreshold, 0.0f, 1.0f) : 999.0f;  // 999 = nothing passes when disabled
    sysPtr->flare_int = enFlare ? std::clamp(flareIntensity, 0.0f, 2.0f) : 0.0f;
    sysPtr->motion_blur = enMotionBlur ? std::clamp(motionBlur, 0.0f, 0.99f) : 0.0f;
    sysPtr->film_grain = enFilmGrain ? std::clamp(filmGrain, 0.0f, 0.1f) : 0.0f;
    sysPtr->disk_density = enDisk ? std::clamp(diskDensity, 0.0f, 1.0f) : 0.0f;
    sysPtr->disk_height = std::clamp(diskHeight, 0.1f, 2.0f);
    sysPtr->shadow_int = enShadow ? std::clamp(shadowIntensity, 0.0f, 1.0f) : 0.0f;
    sysPtr->gw_amp = enGW ? std::clamp(gwAmplitude, 0.0f, 2.0f) : 0.0f;
    sysPtr->jet_int = enJets ? std::clamp(jetIntensity, 0.0f, 2.0f) : 0.0f;

    // Auto-exposure: read luminance data from 2 frames ago (guaranteed complete)
    int readFrame = (currentFrame + 1) % 3;
    uint32_t* lumData = (uint32_t*)lumBuffer[readFrame].contents;
    uint32_t sumEncoded = lumData[0];
    uint32_t pixCount = lumData[1];
    if (pixCount > 100) {
      float avgLogLum = (float(sumEncoded) / float(pixCount)) / 1000.0f - 10.0f;
      float targetExposure = std::clamp(0.18f / exp2f(avgLogLum), 0.15f, 5.0f);
      currentExposure += (targetExposure - currentExposure) * std::min(deltaTime * 1.5f, 1.0f);
    }
    sysPtr->exposure = currentExposure;
    // Clear this frame's luminance accumulator before GPU writes
    memset(lumBuffer[currentFrame].contents, 0, sizeof(uint32_t) * 2);

    GridUniforms *gPtr = (GridUniforms *)gridUniformBuffer[currentFrame].contents;
    gPtr->viewProj = camera.getViewProj(float(drawableW) / float(drawableH));

    @autoreleasepool {
      id<CAMetalDrawable> drawable = [metalLayer nextDrawable];
      if (!drawable) {
        dispatch_semaphore_signal(frameSemaphore);
        return;
      }

      id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];

      // 1. Fluid Simulation
      id<MTLComputeCommandEncoder> fluid = [cmd computeCommandEncoder];
      [fluid setComputePipelineState:fluidPSO];
      [fluid setTexture:fluidTex[currentFluidIdx] atIndex:0];
      [fluid setTexture:fluidTex[1 - currentFluidIdx] atIndex:1];
      [fluid setBuffer:sysUniformBuffer[currentFrame] offset:0 atIndex:0];
      [fluid dispatchThreads:MTLSizeMake(1024, 1024, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [fluid endEncoding];
      currentFluidIdx = 1 - currentFluidIdx;

      // 2. N-Body Physics
      if (Gravity) {
        id<MTLComputeCommandEncoder> phys = [cmd computeCommandEncoder];
        [phys setComputePipelineState:physicsPSO];
        [phys setBuffer:objBuffer offset:0 atIndex:0];
        [phys setBuffer:objUniformBuffer[currentFrame] offset:0 atIndex:1];
        [phys setBuffer:sysUniformBuffer[currentFrame] offset:0 atIndex:2];
        [phys dispatchThreads:MTLSizeMake(activeObjectCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [phys endEncoding];
      }

      // 3. Geodesic Raytrace
      MTLSize tpg = MTLSizeMake(32, 8, 1);
      int curInterm = currentIntermIdx;
      int prevInterm = 1 - currentIntermIdx;

      id<MTLComputeCommandEncoder> ray = [cmd computeCommandEncoder];
      [ray setComputePipelineState:raytracePSO];
      [ray setTexture:intermediateTex[curInterm] atIndex:0];
      [ray setTexture:fluidTex[currentFluidIdx] atIndex:1];
      [ray setBuffer:camBuffer[currentFrame] offset:0 atIndex:0];
      [ray setBuffer:objBuffer offset:0 atIndex:1];
      [ray setBuffer:objUniformBuffer[currentFrame] offset:0 atIndex:2];
      [ray setBuffer:sysUniformBuffer[currentFrame] offset:0 atIndex:3];
      [ray dispatchThreads:MTLSizeMake(drawableW, drawableH, 1) threadsPerThreadgroup:tpg];
      [ray endEncoding];

      // 4. Bloom Extraction (half-res)
      {
        id<MTLComputeCommandEncoder> be = [cmd computeCommandEncoder];
        [be setComputePipelineState:bloomExtractPSO];
        [be setTexture:intermediateTex[curInterm] atIndex:0];
        [be setTexture:bloomTex atIndex:1];
        [be setBuffer:sysUniformBuffer[currentFrame] offset:0 atIndex:0];
        int bw = std::max(1, drawableW / 2);
        int bh = std::max(1, drawableH / 2);
        [be dispatchThreads:MTLSizeMake(bw, bh, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [be endEncoding];
      }

      // 5. MPS Gaussian Blur (cinematic bloom)
      [mpsBloom encodeToCommandBuffer:cmd sourceTexture:bloomTex destinationTexture:bloomBlurTex];

      // 6. Luminance Analysis (auto-exposure)
      {
        id<MTLComputeCommandEncoder> lum = [cmd computeCommandEncoder];
        [lum setComputePipelineState:lumReducePSO];
        [lum setTexture:intermediateTex[curInterm] atIndex:0];
        [lum setBuffer:lumBuffer[currentFrame] offset:0 atIndex:0];
        [lum dispatchThreads:MTLSizeMake(drawableW, drawableH, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [lum endEncoding];
      }

      // 7. Cinematic Post-Processing (bloom + exposure + motion blur + flare + ACES)
      id<MTLComputeCommandEncoder> post = [cmd computeCommandEncoder];
      [post setComputePipelineState:postPSO];
      [post setTexture:intermediateTex[curInterm] atIndex:0];
      [post setTexture:intermediateTex[prevInterm] atIndex:1];
      [post setTexture:drawable.texture atIndex:2];
      [post setTexture:bloomBlurTex atIndex:3];
      [post setBuffer:sysUniformBuffer[currentFrame] offset:0 atIndex:0];
      [post dispatchThreads:MTLSizeMake(drawableW, drawableH, 1) threadsPerThreadgroup:tpg];
      [post endEncoding];

      // Swap intermediate buffer index (no blit copy needed)
      currentIntermIdx = 1 - currentIntermIdx;

      // 5. Grid Render Pass
      MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
      rpd.colorAttachments[0].texture = drawable.texture;
      rpd.colorAttachments[0].loadAction = MTLLoadActionLoad;
      rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
      rpd.depthAttachment.texture = depthTexture;
      rpd.depthAttachment.loadAction = MTLLoadActionClear;
      rpd.depthAttachment.clearDepth = 1.0;
      rpd.depthAttachment.storeAction = MTLStoreActionStore;

      id<MTLRenderCommandEncoder> ren = [cmd renderCommandEncoderWithDescriptor:rpd];
      [ren setRenderPipelineState:gridRenderer->pipelineState];
      [ren setDepthStencilState:gridRenderer->depthState];
      [ren setVertexBuffer:gridRenderer->vertexBuffer offset:0 atIndex:0];
      [ren setVertexBuffer:gridUniformBuffer[currentFrame] offset:0 atIndex:1];
      [ren setVertexBuffer:objBuffer offset:0 atIndex:2];
      [ren setVertexBuffer:objUniformBuffer[currentFrame] offset:0 atIndex:3];
      [ren setVertexBuffer:sysUniformBuffer[currentFrame] offset:0 atIndex:4];
      // Pass the HDR scene texture so grid fragment can hide behind bright objects
      [ren setFragmentTexture:intermediateTex[curInterm] atIndex:0];
      [ren drawIndexedPrimitives:MTLPrimitiveTypeLine indexCount:gridRenderer->indexCount indexType:MTLIndexTypeUInt32 indexBuffer:gridRenderer->indexBuffer indexBufferOffset:0];
      
      ImGui_ImplMetal_NewFrame(rpd);
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();
      ImGui::SetNextWindowSize(ImVec2(380, 620), ImGuiCond_FirstUseEver);
      ImGui::Begin("Geodesic Engine");

      // --- PHYSICS PRESETS ---
      if (ImGui::CollapsingHeader("Presets", ImGuiTreeNodeFlags_DefaultOpen)) {
          ImGui::TextWrapped("Quick configurations for common spacetime geometries.");
          if (ImGui::Button("Schwarzschild")) {
              blackHoleSpin = 0.0f; blackHoleCharge = 0.0f;
              enCharge = false; enJets = false;
              enBloom = false; enFlare = false; enMotionBlur = false; enFilmGrain = false;
              enNebula = false; enScintillation = false;
          }
          ImGui::SameLine();
          if (ImGui::Button("Kerr")) {
              blackHoleSpin = 0.7f; blackHoleCharge = 0.0f;
              enCharge = false; enJets = false;
              enBloom = false; enFlare = false; enMotionBlur = false; enFilmGrain = false;
              enNebula = false; enScintillation = false;
          }
          ImGui::SameLine();
          if (ImGui::Button("Extreme Kerr")) {
              blackHoleSpin = 0.998f; blackHoleCharge = 0.0f;
              enCharge = false; enJets = true;
              enBloom = false; enFlare = false; enMotionBlur = false; enFilmGrain = false;
              enNebula = false; enScintillation = false;
          }
          if (ImGui::Button("Charged (RN)")) {
              blackHoleSpin = 0.0f; blackHoleCharge = 0.5f;
              enCharge = true; enJets = false;
              enBloom = false; enFlare = false; enMotionBlur = false; enFilmGrain = false;
              enNebula = false; enScintillation = false;
          }
          ImGui::SameLine();
          if (ImGui::Button("Kerr-Newman")) {
              blackHoleSpin = 0.6f; blackHoleCharge = 0.3f;
              enCharge = true; enJets = true;
              enBloom = false; enFlare = false; enMotionBlur = false; enFilmGrain = false;
              enNebula = false; enScintillation = false;
          }
          ImGui::SameLine();
          if (ImGui::Button("Cinematic")) {
              blackHoleSpin = 0.85f; blackHoleCharge = 0.0f;
              enCharge = false; enJets = true;
              enBloom = true; enFlare = true; enMotionBlur = true; enFilmGrain = true;
              enNebula = true; enScintillation = true;
          }
      }

      // --- GENERAL RELATIVITY ---
      if (ImGui::CollapsingHeader("General Relativity", ImGuiTreeNodeFlags_DefaultOpen)) {
          ImGui::TextWrapped("Kerr-Newman metric parameters. Spin causes frame dragging; charge modifies the event horizon.");
          ImGui::SliderFloat("Black Hole Spin (a)", &blackHoleSpin, -1.0f, 1.0f);
          ImGui::SliderFloat("Electric Charge (Q)", &blackHoleCharge, 0.0f, 1.0f);
          ImGui::SameLine(); ImGui::Checkbox("##enCharge", &enCharge);
          ImGui::SliderFloat("Simulation Speed", &timeScale, 0.0f, 5.0f);
          ImGui::Checkbox("N-Body Gravitation", &Gravity);
      }

      // --- ACCRETION PHYSICS ---
      if (ImGui::CollapsingHeader("Accretion Disk", ImGuiTreeNodeFlags_DefaultOpen)) {
          ImGui::TextWrapped("Novikov-Thorne thin disk. Temperature follows T ~ r^(-3/4). Inner edge at the ISCO.");
          ImGui::SliderFloat("Plasma Density", &diskDensity, 0.0f, 1.0f);
          ImGui::SameLine(); ImGui::Checkbox("##enDisk", &enDisk);
          ImGui::SliderFloat("Torus Height", &diskHeight, 0.1f, 2.0f);
          ImGui::SliderFloat("Shadow Depth", &shadowIntensity, 0.0f, 1.0f);
          ImGui::SameLine(); ImGui::Checkbox("##enShadow", &enShadow);
      }

      // --- SPACETIME EFFECTS ---
      if (ImGui::CollapsingHeader("Spacetime Effects")) {
          ImGui::SliderFloat("GW Amplitude", &gwAmplitude, 0.0f, 2.0f);
          ImGui::SameLine(); ImGui::Checkbox("##enGW", &enGW);
          ImGui::SliderFloat("Jet Intensity", &jetIntensity, 0.0f, 2.0f);
          ImGui::SameLine(); ImGui::Checkbox("##enJets", &enJets);
      }

      // --- CINEMATIC (collapsed by default — physics first) ---
      if (ImGui::CollapsingHeader("Cinematic Effects")) {
          ImGui::TextWrapped("Non-physical visual effects for cinematic rendering.");
          ImGui::SliderFloat("Nebula Intensity", &nebulaIntensity, 0.0f, 2.0f);
          ImGui::SameLine(); ImGui::Checkbox("##enNebula", &enNebula);
          ImGui::SliderFloat("Star Scintillation", &starScintillation, 0.0f, 1.0f);
          ImGui::SameLine(); ImGui::Checkbox("##enScint", &enScintillation);
          ImGui::SliderFloat("Bloom Threshold", &bloomThreshold, 0.0f, 1.0f);
          ImGui::SameLine(); ImGui::Checkbox("##enBloom", &enBloom);
          ImGui::SliderFloat("Anamorphic Flare", &flareIntensity, 0.0f, 2.0f);
          ImGui::SameLine(); ImGui::Checkbox("##enFlare", &enFlare);
          ImGui::SliderFloat("Motion Blur", &motionBlur, 0.0f, 0.99f);
          ImGui::SameLine(); ImGui::Checkbox("##enMBlur", &enMotionBlur);
          ImGui::SliderFloat("Film Grain", &filmGrain, 0.0f, 0.1f);
          ImGui::SameLine(); ImGui::Checkbox("##enGrain", &enFilmGrain);
      }

      ImGui::Separator();
      // Kerr metric physics readouts (computed from spin)
      {
          float a = std::abs(blackHoleSpin);
          float a2 = a * a;
          float r_h = (1.0f + std::sqrt(std::max(1.0f - a2, 0.0f))) * 0.5f;
          float cbrt = std::pow(std::max(1.0f - a2, 1e-6f), 1.0f/3.0f);
          float Z1 = 1.0f + cbrt * (std::pow(1.0f + a, 1.0f/3.0f) + std::pow(std::max(1.0f - a, 1e-6f), 1.0f/3.0f));
          float Z2 = std::sqrt(3.0f * a2 + Z1 * Z1);
          float r_isco = (3.0f + Z2 - std::sqrt(std::max((3.0f - Z1) * (3.0f + Z1 + 2.0f * Z2), 0.0f))) * 0.5f;
          float r_photon = (1.0f + std::cos(2.0f/3.0f * std::acos(-a)));
          ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f), "Kerr Metric (a = %.3f)", blackHoleSpin);
          ImGui::Text("  Event Horizon:   %.3f rs", r_h);
          ImGui::Text("  Photon Sphere:   %.3f rs", r_photon);
          ImGui::Text("  ISCO (prograde): %.3f rs", r_isco);
          if (a > 0.01f) ImGui::Text("  Ergosphere:      1.000 rs");
      }
      ImGui::Separator();
      ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "Auto-Exposure: %.2f", currentExposure);
      ImGui::Text("GPU Compute: %.1f FPS", ImGui::GetIO().Framerate);
      ImGui::End();
      ImGui::Render();
      ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), cmd, ren);
      [ren endEncoding];

      // Press P to capture screenshot (saves to /tmp/bh_diag_<frame>.ppm)
      bool captureThisFrame = screenshotPending;
      if (captureThisFrame) screenshotPending = false;

      [cmd presentDrawable:drawable];

      // Signal semaphore when GPU finishes this frame
      dispatch_semaphore_t sem = frameSemaphore;
      [cmd addCompletedHandler:^(id<MTLCommandBuffer> _) {
          dispatch_semaphore_signal(sem);
      }];
      [cmd commit];

      // Synchronous screenshot capture AFTER main command buffer completes
      if (captureThisFrame) {
          [cmd waitUntilCompleted];
          NSUInteger w = drawable.texture.width, h = drawable.texture.height;
          NSUInteger bpr = w * 4;
          NSUInteger sz = bpr * h;
          id<MTLBuffer> readback = [device newBufferWithLength:sz options:MTLResourceStorageModeShared];

          id<MTLCommandBuffer> blitCmd = [commandQueue commandBuffer];
          id<MTLBlitCommandEncoder> blit = [blitCmd blitCommandEncoder];
          [blit copyFromTexture:drawable.texture
                    sourceSlice:0 sourceLevel:0
                   sourceOrigin:MTLOriginMake(0, 0, 0)
                     sourceSize:MTLSizeMake(w, h, 1)
                       toBuffer:readback
              destinationOffset:0
         destinationBytesPerRow:bpr
       destinationBytesPerImage:sz];
          [blit endEncoding];
          [blitCmd commit];
          [blitCmd waitUntilCompleted];

          uint8_t *data = (uint8_t *)readback.contents;
          char path[256];
          snprintf(path, sizeof(path), "/tmp/bh_diag_%d.ppm", frameCount);
          FILE *f = fopen(path, "wb");
          if (f) {
              fprintf(f, "P6\n%lu %lu\n255\n", w, h);
              for (NSUInteger y = 0; y < h; y++) {
                  for (NSUInteger x = 0; x < w; x++) {
                      size_t idx = y * bpr + x * 4;
                      fputc(data[idx+2], f);  // BGRA → R
                      fputc(data[idx+1], f);  // G
                      fputc(data[idx+0], f);  // B
                  }
              }
              fclose(f);
              printf("QA capture frame %d: %s (%lux%lu)\n", frameCount, path, w, h);
          } else {
              printf("Failed to open %s for writing\n", path);
          }
      }
    }
  }

  void resize(int w, int h) {
    drawableW = w; drawableH = h;
    metalLayer.drawableSize = CGSizeMake(drawableW, drawableH);
    // Release old textures before re-creating
    intermediateTex[0] = nil;
    intermediateTex[1] = nil;
    bloomTex = nil;
    bloomBlurTex = nil;
    depthTexture = nil;
    createResources();
  }
};

void mouseCallback(GLFWwindow *w, int btn, int act, int mods) {
  if (ImGui::GetIO().WantCaptureMouse) return;
  if (btn == GLFW_MOUSE_BUTTON_LEFT) {
    if (act == GLFW_PRESS) {
      glfwGetCursorPos(w, &camera.lastX, &camera.lastY);
      camera.panning = (mods & GLFW_MOD_SHIFT);
      camera.dragging = !camera.panning;
    } else { camera.dragging = false; camera.panning = false; }
  }
}

void cursorCallback(GLFWwindow * /*w*/, double x, double y) {
  if (ImGui::GetIO().WantCaptureMouse) return;
  float dx = float(x - camera.lastX); float dy = float(y - camera.lastY);
  camera.lastX = x; camera.lastY = y;
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
  if (ImGui::GetIO().WantCaptureMouse) return;
  camera.radius *= (1.0f - float(yoff) * 0.1f);
  if (camera.radius < (float)(SagA_rs * 2.5)) camera.radius = (float)(SagA_rs * 2.5);
}

void keyCallback(GLFWwindow *w, int key, int /*scancode*/, int action, int /*mods*/) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) glfwSetWindowShouldClose(w, GLFW_TRUE);
  if (key == GLFW_KEY_P && action == GLFW_PRESS) screenshotPending = true;
}

int main() {
  if (!glfwInit()) return -1;
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow *window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Black Hole Simulation - Final Frontier", nullptr, nullptr);
  if (!window) { std::cerr << "Failed to create GLFW window" << std::endl; glfwTerminate(); return 1; }  // SEC-06
  glfwSetMouseButtonCallback(window, mouseCallback);
  glfwSetCursorPosCallback(window, cursorCallback);
  glfwSetScrollCallback(window, scrollCallback);
  glfwSetKeyCallback(window, keyCallback);
  vector<SimObject> objects;
  const float GM = (float)(G_const * 8.5e36);

  // --- Black Hole ---
  SimObject bh = {}; bh.posRadius = vec4(0, 0, 0, (float)SagA_rs); bh.color = vec4(0,0,0,1); bh.mass = (float)8.5e36; bh.velocity = vec4(0,0,0,0);

  // Close binary pair (existing)
  SimObject s1 = {}; s1.posRadius = vec4(0, 0, 8.0e12f, 1.2e12f); s1.color = vec4(1.0f, 0.8f, 0.3f, 1); s1.mass = 2.9e30f;
  s1.velocity = vec4(sqrt(GM / 8.0e12f), 0, 0, 0);
  SimObject s2 = {}; s2.posRadius = vec4(0, 0, -8.0e12f, 1.2e12f); s2.color = vec4(0.4f, 0.7f, 1.0f, 1); s2.mass = 2.9e30f;
  s2.velocity = vec4(-sqrt(GM / 8.0e12f), 0, 0, 0);

  // Distant stars — gravitationally lensed via exit-direction check
  // Blue giant behind BH
  float r4 = 80.0e12f; float v4 = sqrt(GM / r4);
  SimObject s4 = {}; s4.posRadius = vec4(r4, 0, 0, 3.0e12f); s4.color = vec4(0.5f, 0.7f, 1.0f, 1); s4.mass = 12.0e30f;
  s4.velocity = vec4(0, 0, v4, 0);

  // Red giant, opposite side
  float r5 = 120.0e12f; float v5 = sqrt(GM / r5);
  SimObject s5 = {}; s5.posRadius = vec4(-r5 * 0.7f, 5.0e12f, r5 * 0.7f, 4.0e12f); s5.color = vec4(1.0f, 0.4f, 0.15f, 1); s5.mass = 5.0e30f;
  s5.velocity = vec4(-v5 * 0.7f, 0, -v5 * 0.7f, 0);

  // White-blue star, high inclination
  float r6 = 60.0e12f; float v6 = sqrt(GM / r6);
  SimObject s6 = {}; s6.posRadius = vec4(0, 15.0e12f, -r6, 2.0e12f); s6.color = vec4(0.85f, 0.9f, 1.0f, 1); s6.mass = 6.0e30f;
  s6.velocity = vec4(v6, 0, 0, 0);

  // Orange star, far orbit
  float r7 = 150.0e12f; float v7 = sqrt(GM / r7);
  SimObject s7 = {}; s7.posRadius = vec4(r7 * 0.5f, 0, r7 * 0.866f, 3.5e12f); s7.color = vec4(1.0f, 0.6f, 0.2f, 1); s7.mass = 4.0e30f;
  s7.velocity = vec4(-v7 * 0.866f, 0, v7 * 0.5f, 0);

  // Violet star
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
  }
  glfwTerminate();
  return 0;
}
