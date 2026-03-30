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

const int WINDOW_WIDTH = 1200;
const int WINDOW_HEIGHT = 800;

const double G_const = 6.67430e-11;
const double c_const = 299792458.0;
const double SagA_rs = 2.0 * G_const * 8.54e36 / (c_const * c_const);

const float PI_F = 3.14159265358979323846f;

// --- SHARED STRUCTS (MUST MATCH GEODESIC.METAL) ---

struct alignas(16) SimObject {
    vec4 posRadius;
    vec4 color;
    float  mass;
    float  _pad0[3];
    vec4 velocity;
};

struct alignas(16) CameraData {
    vec4 camPos;
    vec4 camRight;
    vec4 camUp;
    vec4 camForward;
    float  tanHalfFov;
    float  aspect;
    int    moving;
    int    _pad4;
};

struct alignas(16) SystemUniforms {
    float time;
    float spin;
    float star_scint;
    float nebula_int;
    float charge;
    float dt_sim;
    float bloom_threshold;
    float flare_int;
    float motion_blur;
    float film_grain;
    float disk_density;
    float disk_height;
    float shadow_int;
    float gw_amp;
    float _pad[2];
};

struct alignas(16) ObjectsUniform {
    int count;
};

struct alignas(16) GridUniforms {
    mat4 viewProj;
};

// --- GLOBALS ---

bool Gravity = true;
float blackHoleSpin = 0.85f;
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

struct Camera {
  vec3 target = vec3(0.0f);
  float radius = (float)(SagA_rs * 20.0);
  float azimuth = 0.5f;
  float elevation = 0.4f;
  bool dragging = false;
  bool panning = false;
  double lastX = 0.0;
  double lastY = 0.0;

  vec3 position() const {
    float clampedEl = std::max(0.05f, std::min(elevation, PI_F - 0.05f));
    vec3 offset(radius * sin(clampedEl) * cos(azimuth), radius * cos(clampedEl),
                radius * sin(clampedEl) * sin(azimuth));
    return target + offset;
  }

  mat4 getViewProj() const {
    float aspect = float(WINDOW_WIDTH) / float(WINDOW_HEIGHT);
    mat4 proj = perspective(radians(60.0f), aspect, 1e9f, 5e13f);
    mat4 view = lookAt(position(), target, vec3(0, 1, 0));
    return proj * view;
  }
} camera;

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

    const int N = 120;
    const float spacing = 1.0e11f;
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

  id<MTLBuffer> camBuffer[3];
  id<MTLBuffer> objBuffer[3];
  id<MTLBuffer> objUniformBuffer[3];
  id<MTLBuffer> sysUniformBuffer[3];
  id<MTLBuffer> gridUniformBuffer[3];
  int currentFrame = 0;

  id<MTLTexture> fluidTex[2];
  id<MTLTexture> intermediateTex;
  id<MTLTexture> accumTex;
  id<MTLTexture> depthTexture;
  int currentFluidIdx = 0;

  CAMetalLayer *metalLayer;
  GridRenderer *gridRenderer;

  int drawableW, drawableH;
  GLFWwindow* winRef;

  MetalEngine(GLFWwindow *window, const vector<SimObject> &initialObjects) {
    winRef = window;
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
    NSString *src = [NSString stringWithContentsOfFile:@"geodesic.metal" encoding:NSUTF8StringEncoding error:&err];
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
    if (!lib) { std::cerr << "Metal Library Error: " << err.localizedDescription.UTF8String << std::endl; exit(1); }

    auto createPSO = [&](const char* name) {
        id<MTLFunction> func = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
        if (!func) { std::cerr << "Metal Function Not Found: " << name << std::endl; exit(1); }
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:func error:&err];
        if (!pso) { std::cerr << "PSO Creation Error for " << name << ": " << err.localizedDescription.UTF8String << std::endl; exit(1); }
        return pso;
    };

    raytracePSO = createPSO("raytrace");
    physicsPSO = createPSO("update_physics");
    fluidPSO = createPSO("simulate_disk_fluid");
    postPSO = createPSO("post_process_suite");

    gridRenderer = new GridRenderer(device, lib);

    for (int i = 0; i < 3; i++) {
      camBuffer[i] = [device newBufferWithLength:sizeof(CameraData) options:MTLResourceStorageModeShared];
      objBuffer[i] = [device newBufferWithBytes:initialObjects.data() length:sizeof(SimObject) * initialObjects.size() options:MTLResourceStorageModeShared];
      objUniformBuffer[i] = [device newBufferWithLength:sizeof(ObjectsUniform) options:MTLResourceStorageModeShared];
      sysUniformBuffer[i] = [device newBufferWithLength:sizeof(SystemUniforms) options:MTLResourceStorageModeShared];
      gridUniformBuffer[i] = [device newBufferWithLength:sizeof(GridUniforms) options:MTLResourceStorageModeShared];
    }

    createResources();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOther(window, true);
    ImGui_ImplMetal_Init(device);
  }

  void createResources() {
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:2048 height:2048 mipmapped:NO];
    td.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModePrivate;
    fluidTex[0] = [device newTextureWithDescriptor:td];
    fluidTex[1] = [device newTextureWithDescriptor:td];

    MTLTextureDescriptor *itd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:drawableW height:drawableH mipmapped:NO];
    itd.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    itd.storageMode = MTLStorageModePrivate;
    intermediateTex = [device newTextureWithDescriptor:itd];
    accumTex = [device newTextureWithDescriptor:itd]; 

    MTLTextureDescriptor *dd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float width:drawableW height:drawableH mipmapped:NO];
    dd.usage = MTLTextureUsageRenderTarget;
    dd.storageMode = MTLStorageModePrivate;
    depthTexture = [device newTextureWithDescriptor:dd];
  }

  void render(int activeObjectCount) {
    int curW, curH;
    glfwGetFramebufferSize(winRef, &curW, &curH);
    if (curW != drawableW || curH != drawableH) {
      resize(curW, curH);
    }

    simulationTime += 0.016f * timeScale;
    currentFrame = (currentFrame + 1) % 3;

    CameraData *cPtr = (CameraData *)camBuffer[currentFrame].contents;
    vec3 camPos = camera.position();
    vec3 fwd = normalize(camera.target - camPos);
    vec3 right = normalize(cross(fwd, vec3(0, 1, 0)));
    vec3 up = cross(right, fwd);
    cPtr->camPos = vec4(camPos, 0.0f); cPtr->camRight = vec4(right, 0.0f); cPtr->camUp = vec4(up, 0.0f);
    cPtr->camForward = vec4(fwd, 0.0f); cPtr->tanHalfFov = tan(radians(30.0f));
    cPtr->aspect = float(drawableW) / float(drawableH);

    ObjectsUniform *ouPtr = (ObjectsUniform *)objUniformBuffer[currentFrame].contents; ouPtr->count = activeObjectCount;
    SystemUniforms *sysPtr = (SystemUniforms *)sysUniformBuffer[currentFrame].contents;
    sysPtr->time = simulationTime; sysPtr->spin = blackHoleSpin;
    sysPtr->star_scint = starScintillation; sysPtr->nebula_int = nebulaIntensity;
    sysPtr->charge = blackHoleCharge; sysPtr->dt_sim = 0.016f * timeScale;
    sysPtr->bloom_threshold = bloomThreshold; sysPtr->flare_int = flareIntensity;
    sysPtr->motion_blur = motionBlur; sysPtr->film_grain = filmGrain;
    sysPtr->disk_density = diskDensity; sysPtr->disk_height = diskHeight;
    sysPtr->shadow_int = shadowIntensity; sysPtr->gw_amp = gwAmplitude;

    GridUniforms *gPtr = (GridUniforms *)gridUniformBuffer[currentFrame].contents;
    gPtr->viewProj = camera.getViewProj();

    @autoreleasepool {
      id<CAMetalDrawable> drawable = [metalLayer nextDrawable];
      if (!drawable) return;

      id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];

      // 1. Fluid Simulation
      id<MTLComputeCommandEncoder> fluid = [cmd computeCommandEncoder];
      [fluid setComputePipelineState:fluidPSO];
      [fluid setTexture:fluidTex[currentFluidIdx] atIndex:0];
      [fluid setTexture:fluidTex[1 - currentFluidIdx] atIndex:1];
      [fluid setBuffer:sysUniformBuffer[currentFrame] offset:0 atIndex:0];
      [fluid dispatchThreadgroups:MTLSizeMake(128, 128, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [fluid endEncoding];
      currentFluidIdx = 1 - currentFluidIdx;

      // 2. N-Body Physics
      if (Gravity) {
        id<MTLComputeCommandEncoder> phys = [cmd computeCommandEncoder];
        [phys setComputePipelineState:physicsPSO];
        [phys setBuffer:objBuffer[currentFrame] offset:0 atIndex:0];
        [phys setBuffer:objUniformBuffer[currentFrame] offset:0 atIndex:1];
        [phys dispatchThreadgroups:MTLSizeMake((activeObjectCount+31)/32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [phys endEncoding];
      }

      // 3. Quantum Gargantua Raytrace
      MTLSize threads = MTLSizeMake(16, 16, 1);
      MTLSize groups = MTLSizeMake((drawableW + 15) / 16, (drawableH + 15) / 16, 1);

      id<MTLComputeCommandEncoder> ray = [cmd computeCommandEncoder];
      [ray setComputePipelineState:raytracePSO];
      [ray setTexture:intermediateTex atIndex:0];
      [ray setTexture:fluidTex[currentFluidIdx] atIndex:1];
      [ray setBuffer:camBuffer[currentFrame] offset:0 atIndex:0];
      [ray setBuffer:objBuffer[currentFrame] offset:0 atIndex:1];
      [ray setBuffer:objUniformBuffer[currentFrame] offset:0 atIndex:2];
      [ray setBuffer:sysUniformBuffer[currentFrame] offset:0 atIndex:3];
      [ray dispatchThreadgroups:groups threadsPerThreadgroup:threads];
      [ray endEncoding];

      // 4. Cinematic Post-Processing
      id<MTLComputeCommandEncoder> post = [cmd computeCommandEncoder];
      [post setComputePipelineState:postPSO];
      [post setTexture:intermediateTex atIndex:0];
      [post setTexture:accumTex atIndex:1];
      [post setTexture:drawable.texture atIndex:2];
      [post setBuffer:sysUniformBuffer[currentFrame] offset:0 atIndex:0];
      [post dispatchThreadgroups:groups threadsPerThreadgroup:threads];
      [post endEncoding];

      // Copy to Accumulation for Motion Blur
      id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
      [blit copyFromTexture:intermediateTex toTexture:accumTex];
      [blit endEncoding];

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
      [ren setVertexBuffer:objBuffer[currentFrame] offset:0 atIndex:2];
      [ren setVertexBuffer:objUniformBuffer[currentFrame] offset:0 atIndex:3];
      [ren setVertexBuffer:sysUniformBuffer[currentFrame] offset:0 atIndex:4];
      [ren drawIndexedPrimitives:MTLPrimitiveTypeLine indexCount:gridRenderer->indexCount indexType:MTLIndexTypeUInt32 indexBuffer:gridRenderer->indexBuffer indexBufferOffset:0];
      
      ImGui_ImplMetal_NewFrame(rpd);
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();
      ImGui::SetNextWindowSize(ImVec2(350, 450), ImGuiCond_FirstUseEver);
      ImGui::Begin("Quantum Gargantua Engine");
      if (ImGui::CollapsingHeader("Physics Metrics", ImGuiTreeNodeFlags_DefaultOpen)) {
          ImGui::SliderFloat("Black Hole Spin (a)", &blackHoleSpin, -1.0f, 1.0f);
          ImGui::SliderFloat("Electric Charge (Q)", &blackHoleCharge, 0.0f, 1.0f);
          ImGui::SliderFloat("Simulation Speed", &timeScale, 0.0f, 5.0f);
          ImGui::Checkbox("N-Body Gravitation", &Gravity);
      }
      if (ImGui::CollapsingHeader("Disk Fluid Dynamics", ImGuiTreeNodeFlags_DefaultOpen)) {
          ImGui::SliderFloat("Plasma Density", &diskDensity, 0.0f, 1.0f);
          ImGui::SliderFloat("Torus Height", &diskHeight, 0.1f, 2.0f);
          ImGui::SliderFloat("Shadow Depth", &shadowIntensity, 0.0f, 1.0f);
      }
      if (ImGui::CollapsingHeader("Gravitational Waves", ImGuiTreeNodeFlags_DefaultOpen)) {
          ImGui::SliderFloat("GW Amplitude", &gwAmplitude, 0.0f, 2.0f);
      }
      if (ImGui::CollapsingHeader("Cinematic Optics", ImGuiTreeNodeFlags_DefaultOpen)) {
          ImGui::SliderFloat("Nebula Intensity", &nebulaIntensity, 0.0f, 2.0f);
          ImGui::SliderFloat("Star Scintillation", &starScintillation, 0.0f, 1.0f);
          ImGui::SliderFloat("Bloom Threshold", &bloomThreshold, 0.0f, 1.0f);
          ImGui::SliderFloat("Anamorphic Flare", &flareIntensity, 0.0f, 2.0f);
          ImGui::SliderFloat("Motion Blur", &motionBlur, 0.0f, 0.99f);
          ImGui::SliderFloat("Film Grain", &filmGrain, 0.0f, 0.1f);
      }
      ImGui::Separator();
      ImGui::Text("GPU Compute Performance: %.1f FPS", ImGui::GetIO().Framerate);
      ImGui::End();
      ImGui::Render();
      ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), cmd, ren);
      [ren endEncoding];

      [cmd presentDrawable:drawable];
      [cmd commit];
    }
  }

  void resize(int w, int h) {
    drawableW = w; drawableH = h;
    metalLayer.drawableSize = CGSizeMake(drawableW, drawableH);
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

void cursorCallback(GLFWwindow *w, double x, double y) {
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

void scrollCallback(GLFWwindow *w, double xoff, double yoff) {
  if (ImGui::GetIO().WantCaptureMouse) return;
  camera.radius -= float(yoff) * 5e11f;
  if (camera.radius < (float)(SagA_rs * 2.5)) camera.radius = (float)(SagA_rs * 2.5);
}

int main() {
  if (!glfwInit()) return -1;
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow *window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Black Hole Simulation - Final Frontier", nullptr, nullptr);
  glfwSetMouseButtonCallback(window, mouseCallback);
  glfwSetCursorPosCallback(window, cursorCallback);
  glfwSetScrollCallback(window, scrollCallback);
  vector<SimObject> objects;
  objects.push_back({vec4(0, 0, 8.0e12f, 1.2e12f), vec4(1, 0.8, 0.3, 1), 2.9e30f, {0,0,0}, vec4(1e7, 0, 0, 0)});
  objects.push_back({vec4(0, 0, -8.0e12f, 1.2e12f), vec4(0.4, 0.7, 1, 1), 2.9e30f, {0,0,0}, vec4(-1e7, 0, 0, 0)});
  objects.push_back({vec4(0, 0, 0, (float)SagA_rs), vec4(0,0,0,1), (float)8.5e36, {0,0,0}, vec4(0,0,0,0)});
  MetalEngine engine(window, objects);
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    engine.render((int)objects.size());
  }
  glfwTerminate();
  return 0;
}
