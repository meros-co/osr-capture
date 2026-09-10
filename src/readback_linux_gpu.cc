// Linux GPU readback backend (design: LINUX_GPU_READBACK.md).
//
// A single dedicated GL thread owns a headless EGL + GLES 3.0 context and services a FIFO job queue
// (GL contexts are thread-bound; readback calls arrive on ANY libuv worker). Per frame:
//   consume: eglCreateImageKHR(EGL_LINUX_DMA_BUF_EXT) -> GL texture -> fragment-shader convert
//            (+ optional box downscale) into cached FBO targets -> fenceDraw -> glReadPixels into a
//            per-key PBO (async DMA) -> fenceRead + flush -> client-wait fenceDraw only -> destroy the
//            image/texture and RETURN (the dmabuf is no longer read; Electron may release the frame).
//   finish:  client-wait fenceRead -> map the PBO on the GL thread -> the CALLING libuv worker does the
//            large memcpy out of the (cached, system-memory) mapping -> an unmap job recycles the PBO.
// The two slow stages (readback DMA, copy-out) therefore overlap the GL thread's cheap draw work, so
// multiple concurrent 4K outputs pipeline through the one context without corruption (all GL state is
// GL-thread-only; keys are independent).
//
// BT.601 full range with convert.cc's exact integer coefficients, so GPU and CPU paths agree.
#include "readback_linux_gpu.h"

#if defined(__linux__)

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <thread>

// ---- constant fallbacks (older EGL/GLES headers) ---------------------------------------------------------
#ifndef EGL_PLATFORM_SURFACELESS_MESA
#define EGL_PLATFORM_SURFACELESS_MESA 0x31DD
#endif
#ifndef EGL_PLATFORM_DEVICE_EXT
#define EGL_PLATFORM_DEVICE_EXT 0x313F
#endif
#ifndef EGL_NO_CONFIG_KHR
#define EGL_NO_CONFIG_KHR ((EGLConfig)0)
#endif
#ifndef EGL_OPENGL_ES3_BIT_KHR
#define EGL_OPENGL_ES3_BIT_KHR 0x0040
#endif
#ifndef EGL_LINUX_DMA_BUF_EXT
#define EGL_LINUX_DMA_BUF_EXT 0x3270
#endif
#ifndef EGL_LINUX_DRM_FOURCC_EXT
#define EGL_LINUX_DRM_FOURCC_EXT 0x3271
#endif
#ifndef EGL_DMA_BUF_PLANE0_FD_EXT
#define EGL_DMA_BUF_PLANE0_FD_EXT 0x3272
#define EGL_DMA_BUF_PLANE0_OFFSET_EXT 0x3273
#define EGL_DMA_BUF_PLANE0_PITCH_EXT 0x3274
#define EGL_DMA_BUF_PLANE1_FD_EXT 0x3275
#define EGL_DMA_BUF_PLANE1_OFFSET_EXT 0x3276
#define EGL_DMA_BUF_PLANE1_PITCH_EXT 0x3277
#define EGL_DMA_BUF_PLANE2_FD_EXT 0x3278
#define EGL_DMA_BUF_PLANE2_OFFSET_EXT 0x3279
#define EGL_DMA_BUF_PLANE2_PITCH_EXT 0x327A
#endif
#ifndef EGL_DMA_BUF_PLANE3_FD_EXT
#define EGL_DMA_BUF_PLANE3_FD_EXT 0x3440
#define EGL_DMA_BUF_PLANE3_OFFSET_EXT 0x3441
#define EGL_DMA_BUF_PLANE3_PITCH_EXT 0x3442
#endif
#ifndef EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT
#define EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT 0x3443
#define EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT 0x3444
#define EGL_DMA_BUF_PLANE1_MODIFIER_LO_EXT 0x3445
#define EGL_DMA_BUF_PLANE1_MODIFIER_HI_EXT 0x3446
#define EGL_DMA_BUF_PLANE2_MODIFIER_LO_EXT 0x3447
#define EGL_DMA_BUF_PLANE2_MODIFIER_HI_EXT 0x3448
#define EGL_DMA_BUF_PLANE3_MODIFIER_LO_EXT 0x3449
#define EGL_DMA_BUF_PLANE3_MODIFIER_HI_EXT 0x344A
#endif

namespace osrcap {
namespace linuxgpu {

namespace {

// DRM fourccs, defined locally so libdrm-dev is not a build dependency. Electron's OSR BGRA buffer is
// byte order B,G,R,A = DRM_FORMAT_ARGB8888 ('AR24'); XRGB8888 is the opaque variant some stacks report.
// (ABGR8888 is deliberately NOT tried: it would import a BGRA buffer with swapped channels.)
constexpr uint32_t Fourcc(char a, char b, char c, char d) {
    return (uint32_t)(uint8_t)a | ((uint32_t)(uint8_t)b << 8) | ((uint32_t)(uint8_t)c << 16) | ((uint32_t)(uint8_t)d << 24);
}
constexpr uint32_t kFourccArgb8888 = Fourcc('A', 'R', '2', '4');
constexpr uint32_t kFourccXrgb8888 = Fourcc('X', 'R', '2', '4');
constexpr uint64_t kModifierInvalid = 0x00ffffffffffffffULL;  // DRM_FORMAT_MOD_INVALID

constexpr GLuint64 kFenceTimeoutNs = 2000000000ULL;  // 2s watchdog, matches the Windows fence watchdogs
constexpr long long kDemotionBaseMs = 1000;          // first backoff; doubles per successive demotion
constexpr long long kDemotionMaxMs = 60000;          // ceiling, so even a broken GPU still re-probes
constexpr int kImportFailDemotion = 3;               // consecutive import failures before demoting to CPU

bool HasToken(const char* list, const char* token) {
    if (!list || !token) return false;
    size_t len = std::strlen(token);
    const char* p = list;
    while ((p = std::strstr(p, token)) != nullptr) {
        // whole-word match (extension names are space-separated)
        if ((p == list || p[-1] == ' ') && (p[len] == '\0' || p[len] == ' ')) return true;
        p += len;
    }
    return false;
}

// ---- diagnostics (FS_CAP_GPU_DIAG=1 or FS_CAP_STATS=1) ---------------------------------------------------
// Rate-limited (~1/s) resource-lifecycle telemetry so a wedge on real hardware pinpoints itself:
//   live(...) climbing        -> a per-frame leak (EGLImage / GL texture / fence / dup'd fd held)
//   q= growing + last(consume) frozen -> the GL thread is stalled (fence wait / driver hang)
//   calls(...) frozen but live/q flat -> the CALLERS stopped invoking us (frames no longer forwarded:
//                                        upstream frame pool drained, or FreeShow's paint drive is absent)
// Plus a first-8-frames per-stage trace (import / convert / readback / released).
int64_t NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Diag {
    bool enabled = false;
    // cumulative created/destroyed pairs; live = created - destroyed
    std::atomic<uint64_t> imgCreated{0}, imgDestroyed{0};        // EGLImages
    std::atomic<uint64_t> texCreated{0}, texDestroyed{0};        // per-frame dmabuf source textures
    std::atomic<uint64_t> fenceCreated{0}, fenceDestroyed{0};    // GLsync objects
    std::atomic<uint64_t> pboCreated{0}, pboDestroyed{0};        // per-key PBOs (cached; live == active keys)
    std::atomic<uint64_t> dupCreated{0}, dupClosedUs{0};         // plane fds WE dup'd / WE closed (EGL closes the rest)
    std::atomic<uint64_t> consumeCalls{0}, consumeFails{0}, finishCalls{0}, finishFails{0};
    std::atomic<int64_t> lastConsumeMs{0}, lastFinishMs{0};      // NowMs() of the most recent invocation
    std::atomic<uint32_t> lastGlErr{0}, lastEglErr{0};           // most recent NONZERO error observed
    std::atomic<uint64_t> frameSeq{0};                           // consume ordinal (drives the first-8 trace)
    std::atomic<int64_t> lastLogMs{0};

    Diag() {
        const char* d = std::getenv("FS_CAP_GPU_DIAG");
        const char* s = std::getenv("FS_CAP_STATS");
        enabled = (d && d[0] && d[0] != '0') || (s && s[0] && s[0] != '0');
    }
};

Diag& D() {
    static Diag d;
    return d;
}

// First-8-frames stage trace. `seq` is the consume ordinal captured at entry.
void Trace8(uint64_t seq, const std::string& key, const char* msg) {
    if (!D().enabled || seq >= 8) return;
    std::fprintf(stderr, "[osr-capture] gpu-trace f%llu key=%s %s\n", (unsigned long long)seq, key.c_str(), msg);
}

// All draws are fullscreen triangles addressed by gl_FragCoord + texelFetch, so the mapping is pure
// memory-row identity: texel row 0 = first row in dmabuf memory, glReadPixels row 0 = first row of the
// output — the same orientation the old CPU memcpy produced. SRC() applies the FS_LINUX_READBACK_FLIP=1
// escape hatch (uFlipY) in case a driver imports with inverted orientation.
const char* kVert = R"GLSL(#version 300 es
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)GLSL";

const char* kFragPrelude = R"GLSL(#version 300 es
precision highp float;
precision highp int;
uniform sampler2D uTex;
uniform int uFlipY;
uniform int uSrcH;
uniform ivec2 uSrcSize;
uniform ivec2 uDstSize;
out vec4 o;
ivec2 SRC(int x, int y) { return ivec2(x, uFlipY == 1 ? (uSrcH - 1 - y) : y); }
// Box-average the source rect that maps to destination pixel (dx, dy). uDstSize is left unset (0) by the
// non-scaling shaders, so BOX degenerates to the plain fetch they used before.
vec4 BOX(int dx, int dy) {
    if (uDstSize.x <= 0 || uSrcSize == uDstSize) return texelFetch(uTex, SRC(dx, dy), 0);
    int sx0 = dx * uSrcSize.x / uDstSize.x;
    int sx1 = min(max(sx0 + 1, (dx + 1) * uSrcSize.x / uDstSize.x), uSrcSize.x);
    int sy0 = dy * uSrcSize.y / uDstSize.y;
    int sy1 = min(max(sy0 + 1, (dy + 1) * uSrcSize.y / uDstSize.y), uSrcSize.y);
    vec4 acc = vec4(0.0);
    int n = 0;
    for (int sy = sy0; sy < sy1; ++sy)
        for (int sx = sx0; sx < sx1; ++sx) {
            acc += texelFetch(uTex, SRC(sx, sy), 0);
            ++n;
        }
    return acc / float(n);
}
// BT.601 full range; the exact integer coefficients of convert.cc (x/256 on 0..255 inputs).
vec3 YUV(vec3 c) {
    vec3 s = c * 255.0;
    float y = (77.0 * s.r + 150.0 * s.g + 29.0 * s.b) / 256.0;
    float u = (-43.0 * s.r - 85.0 * s.g + 128.0 * s.b) / 256.0 + 128.0;
    float v = (128.0 * s.r - 107.0 * s.g - 21.0 * s.b) / 256.0 + 128.0;
    return vec3(y, u, v) / 255.0;
}
// ---- video layer: a live input frame composited UNDER the page, so it never has to travel through the
// browser's GPU thread to reach the output. The layer texture is uploaded as GL_RGBA/GL_UNSIGNED_BYTE, so
// channel .r is byte 0 of a texel, .g byte 1, .b byte 2, .a byte 3 (the mirror of kFragCopyBgra's
// reasoning, and the opposite of the Windows layer texture, which is BGRA).
uniform sampler2D uVideo;
uniform int uVideoOn;
uniform ivec2 uVideoSize;
uniform int uVideoFormat;
vec3 videoAt(int x, int y, ivec2 dstSize) {
    int vx = (dstSize.x == uVideoSize.x) ? x : x * uVideoSize.x / dstSize.x;
    int vy = (dstSize.y == uVideoSize.y) ? y : y * uVideoSize.y / dstSize.y;
    vx = clamp(vx, 0, uVideoSize.x - 1);  // dodge out-of-range texelFetch UB on a rounded-up dst pixel
    vy = clamp(vy, 0, uVideoSize.y - 1);
    if (uVideoFormat == 0) return texelFetch(uVideo, ivec2(vx, vy), 0).bgr;  // bytes B,G,R,A
    if (uVideoFormat == 3) return texelFetch(uVideo, ivec2(vx, vy), 0).rgb;  // bytes R,G,B,A
    vec4 s = texelFetch(uVideo, ivec2(vx / 2, vy), 0);                       // UYVY: bytes U,Y0,V,Y1
    float luma = ((vx & 1) == 0) ? s.g : s.a;
    float yy = (luma - 16.0 / 255.0) * (255.0 / 219.0);
    float u = (s.r - 128.0 / 255.0) * (255.0 / 224.0);
    float v = (s.b - 128.0 / 255.0) * (255.0 / 224.0);
    float bt = (uVideoSize.y >= 720) ? 1.0 : 0.0;  // BT.601 below 720 lines, BT.709 at or above
    float kr = mix(1.402, 1.5748, bt), kb = mix(1.772, 1.8556, bt);
    float gu = mix(0.344136, 0.1873, bt), gv = mix(0.714136, 0.4681, bt);
    return clamp(vec3(yy + kr * v, yy - gu * u - gv * v, yy + kb * u), 0.0, 1.0);
}
// page over video, premultiplied: opaque page pixels are untouched. (x, y) are DESTINATION PIXEL
// coordinates, which for a packed format are not the fragment coordinates — see each shader.
vec4 overVideo(vec4 page, int x, int y, ivec2 dstSize) {
    if (uVideoOn == 0 || page.a >= 0.999) return page;
    return vec4(page.rgb + videoAt(x, y, dstSize) * (1.0 - page.a), 1.0);
}
// One destination pixel of a scaling pass: the box average, over the video layer.
vec4 PIX(int dx, int dy) { return overVideo(BOX(dx, dy), dx, dy, uDstSize); }
)GLSL";

// dst texel x covers source pixels 2x, 2x+1; chroma from the even pixel (matches ConvertBgraToUyvyRaw).
// Output RGBA8 bytes = U, Y0, V, Y1 -> the UYVY wire layout when read back as GL_RGBA/UNSIGNED_BYTE.
// This pass is 1:1, so uSrcSize IS the destination size the video layer maps onto.
const char* kFragUyvy = R"GLSL(
void main() {
    ivec2 d = ivec2(gl_FragCoord.xy);
    int x0 = d.x * 2, x1 = d.x * 2 + 1;
    vec4 p0 = overVideo(texelFetch(uTex, SRC(x0, d.y), 0), x0, d.y, uSrcSize);
    vec4 p1 = overVideo(texelFetch(uTex, SRC(x1, d.y), 0), x1, d.y, uSrcSize);
    vec3 yuv0 = YUV(p0.rgb);
    float y1 = YUV(p1.rgb).x;
    o = vec4(yuv0.y, yuv0.x, yuv0.z, y1);
}
)GLSL";

// UYVA plane 2: pack 4 source alphas per RGBA8 texel (dst width = ceil(w/4)); the copy-out trims the GL
// row pitch back to w bytes when w % 4 != 0. Coordinates clamped to dodge out-of-range texelFetch UB.
// Each of the four packed pixels is composited on its own, so an alpha that the video layer made opaque
// reads back as 1.0 (matching the UYVY plane beside it).
const char* kFragAlpha = R"GLSL(
vec4 PAGE(int x, int y) { return overVideo(texelFetch(uTex, SRC(x, y), 0), x, y, uSrcSize); }
void main() {
    ivec2 d = ivec2(gl_FragCoord.xy);
    int w = textureSize(uTex, 0).x;
    int x0 = d.x * 4;
    o = vec4(PAGE(min(x0, w - 1), d.y).a,
             PAGE(min(x0 + 1, w - 1), d.y).a,
             PAGE(min(x0 + 2, w - 1), d.y).a,
             PAGE(min(x0 + 3, w - 1), d.y).a);
}
)GLSL";

// format 0: memory bytes must be B,G,R,A -> framebuffer (R,G,B,A) = sampled (b,g,r,a).
const char* kFragCopyBgra = R"GLSL(
void main() {
    ivec2 d = ivec2(gl_FragCoord.xy);
    o = overVideo(texelFetch(uTex, SRC(d.x, d.y), 0), d.x, d.y, uSrcSize).bgra;
}
)GLSL";

// format 3: memory bytes R,G,B,A -> passthrough.
const char* kFragCopyRgba = R"GLSL(
void main() {
    ivec2 d = ivec2(gl_FragCoord.xy);
    o = overVideo(texelFetch(uTex, SRC(d.x, d.y), 0), d.x, d.y, uSrcSize);
}
)GLSL";

// The shared scale source. Consumers smaller than the render each box-filtered the whole frame for
// themselves; this produces one composited downscale at the largest of their sizes, in the SOURCE's own
// channel order and already un-flipped, so a pass sampling it (with uFlipY and uVideoOn off) behaves
// exactly as if it were sampling the render.
const char* kFragShare = R"GLSL(
void main() {
    ivec2 d = ivec2(gl_FragCoord.xy);
    o = PIX(d.x, d.y);
}
)GLSL";

// Box-filter downscale to BGRA (same semantics as convert.cc DownscaleBgra / the Windows scale pass).
const char* kFragScale = R"GLSL(
void main() {
    ivec2 d = ivec2(gl_FragCoord.xy);
    o = PIX(d.x, d.y).bgra;
}
)GLSL";

// ---- per-consumer targets: the same conversions, but sampling through BOX so one pass scales AND converts.
// uDstSize is the target's size in DESTINATION PIXELS; the viewport is the packed texel width (see DrawToSized).

const char* kFragScaleUyvy = R"GLSL(
void main() {
    ivec2 d = ivec2(gl_FragCoord.xy);
    vec3 yuv0 = YUV(PIX(d.x * 2, d.y).rgb);
    float y1 = YUV(PIX(d.x * 2 + 1, d.y).rgb).x;
    o = vec4(yuv0.y, yuv0.x, yuv0.z, y1);
}
)GLSL";

const char* kFragScaleAlpha = R"GLSL(
void main() {
    ivec2 d = ivec2(gl_FragCoord.xy);
    int w = uDstSize.x;
    int x0 = d.x * 4;
    o = vec4(PIX(min(x0, w - 1), d.y).a,
             PIX(min(x0 + 1, w - 1), d.y).a,
             PIX(min(x0 + 2, w - 1), d.y).a,
             PIX(min(x0 + 3, w - 1), d.y).a);
}
)GLSL";

const char* kFragScaleRgba = R"GLSL(
void main() {
    ivec2 d = ivec2(gl_FragCoord.xy);
    o = PIX(d.x, d.y);
}
)GLSL";

// I420 (BT.601 limited range), the exact integer arithmetic of kScaleI420HLSL in readback_win.cc. SHR8 is
// an explicit floor-divide because GLSL ES leaves >> on a negative signed value implementation-defined,
// while the HLSL it must match shifts arithmetically.
const char* kFragI420Luma = R"GLSL(
int SHR8(int n) { return n >= 0 ? (n >> 8) : -((-n + 255) >> 8); }
float LUMA(vec4 c) {
    int r = int(c.r * 255.0 + 0.5), g = int(c.g * 255.0 + 0.5), b = int(c.b * 255.0 + 0.5);
    return float(clamp(SHR8(66 * r + 129 * g + 25 * b + 128) + 16, 16, 235)) / 255.0;
}
void main() {
    ivec2 d = ivec2(gl_FragCoord.xy);
    int w = uDstSize.x;
    int x0 = d.x * 4;
    o = vec4(LUMA(PIX(min(x0, w - 1), d.y)),
             LUMA(PIX(min(x0 + 1, w - 1), d.y)),
             LUMA(PIX(min(x0 + 2, w - 1), d.y)),
             LUMA(PIX(min(x0 + 3, w - 1), d.y)));
}
)GLSL";

// Chroma helpers shared by the U and V shaders: CHROMA(cx, cy) averages the 2x2 destination pixels the
// chroma sample covers, each of them itself a BOX of the source composited over the video layer (the
// composite is per destination pixel, before the 2x2 average — as in kScaleI420HLSL).
const char* kFragI420ChromaPrelude = R"GLSL(
int SHR8(int n) { return n >= 0 ? (n >> 8) : -((-n + 255) >> 8); }
vec4 CHROMA(int cx, int cy) {
    int x0 = min(cx * 2, uDstSize.x - 1), x1 = min(cx * 2 + 1, uDstSize.x - 1);
    int y0 = min(cy * 2, uDstSize.y - 1), y1 = min(cy * 2 + 1, uDstSize.y - 1);
    return (PIX(x0, y0) + PIX(x1, y0) + PIX(x0, y1) + PIX(x1, y1)) * 0.25;
}
)GLSL";

const char* kFragI420U = R"GLSL(
float U8(vec4 c) {
    int r = int(c.r * 255.0 + 0.5), g = int(c.g * 255.0 + 0.5), b = int(c.b * 255.0 + 0.5);
    return float(clamp(SHR8(-38 * r - 74 * g + 112 * b + 128) + 128, 16, 240)) / 255.0;
}
void main() {
    ivec2 d = ivec2(gl_FragCoord.xy);
    int cwm = max(uDstSize.x / 2 - 1, 0);
    int c0 = d.x * 4;
    o = vec4(U8(CHROMA(min(c0, cwm), d.y)),
             U8(CHROMA(min(c0 + 1, cwm), d.y)),
             U8(CHROMA(min(c0 + 2, cwm), d.y)),
             U8(CHROMA(min(c0 + 3, cwm), d.y)));
}
)GLSL";

const char* kFragI420V = R"GLSL(
float V8(vec4 c) {
    int r = int(c.r * 255.0 + 0.5), g = int(c.g * 255.0 + 0.5), b = int(c.b * 255.0 + 0.5);
    return float(clamp(SHR8(112 * r - 94 * g - 18 * b + 128) + 128, 16, 240)) / 255.0;
}
void main() {
    ivec2 d = ivec2(gl_FragCoord.xy);
    int cwm = max(uDstSize.x / 2 - 1, 0);
    int c0 = d.x * 4;
    o = vec4(V8(CHROMA(min(c0, cwm), d.y)),
             V8(CHROMA(min(c0 + 1, cwm), d.y)),
             V8(CHROMA(min(c0 + 2, cwm), d.y)),
             V8(CHROMA(min(c0 + 3, cwm), d.y)));
}
)GLSL";

struct Program {
    GLuint id = 0;
    GLint uTex = -1, uFlipY = -1, uSrcH = -1, uSrcSize = -1, uDstSize = -1;
    GLint uVideo = -1, uVideoOn = -1, uVideoSize = -1, uVideoFormat = -1;
};

// One plane of one per-consumer target inside the PBO. `rowGl` is the GL row pitch of the render target
// (always a whole number of RGBA8 texels); `rowDst` is what the destination wants per row, which is
// smaller whenever the packing rounded the texel width up.
struct TargetPlane {
    size_t off = 0;
    uint32_t rowGl = 0, rowDst = 0, rows = 0;
};
constexpr int kMaxTargetPlanes = 3;  // I420 (Y, U, V) is the widest case
struct TargetLayout {
    int nPlanes = 0;
    TargetPlane plane[kMaxTargetPlanes];
    size_t dstBytes = 0;  // TargetBytes(spec)
};
// Per-target GPU state, cached per key exactly like texMain/texScaled (reallocated only on a size change).
struct TargetState {
    GLuint tex[kMaxTargetPlanes] = {0, 0, 0};
    GLuint fbo[kMaxTargetPlanes] = {0, 0, 0};
    int w[kMaxTargetPlanes] = {0, 0, 0};
    int h[kMaxTargetPlanes] = {0, 0, 0};
    TargetLayout layout;
};

void DeleteTargetState(TargetState& ts) {
    for (int i = 0; i < kMaxTargetPlanes; ++i) {
        if (ts.fbo[i]) glDeleteFramebuffers(1, &ts.fbo[i]);
        if (ts.tex[i]) glDeleteTextures(1, &ts.tex[i]);
        ts.fbo[i] = 0;
        ts.tex[i] = 0;
        ts.w[i] = ts.h[i] = 0;
    }
}

// Cached per-key GL state + the pending (consumed, not yet finished) frame. GL-THREAD-ONLY.
struct KeyState {
    // render targets (RGBA8 texture + FBO), reallocated when the frame size changes
    GLuint texMain = 0, fboMain = 0;
    int mainW = 0, mainH = 0;
    GLuint texAlpha = 0, fboAlpha = 0;
    int alphaW = 0, alphaH = 0;
    GLuint texScaled = 0, fboScaled = 0;
    int scaledW = 0, scaledH = 0;
    // the shared scale source (see kFragShare), reused across frames like the other targets
    GLuint texShare = 0, fboShare = 0;
    int shareW = 0, shareH = 0;
    // video layer, cached per key (one texture per output: two outputs with differently sized layers must
    // not thrash a single shared allocation). Reallocated only when the texel dimensions change.
    GLuint videoTex = 0;
    int videoTexW = 0, videoTexH = 0;
    std::vector<TargetState> targets;
    GLuint pbo = 0;
    size_t pboCap = 0;
    // pending frame
    bool pending = false;
    GLsync fenceRead = nullptr;
    void* mapped = nullptr;
    int format = 0;
    uint32_t w = 0, h = 0;
    size_t mainBytes = 0;                       // final tightly-packed main-plane bytes (uyvy or bgra/rgba)
    size_t alphaOff = 0;                        // PBO offset of the alpha rows (format 2)
    uint32_t alphaRowGl = 0;                    // GL row pitch of the alpha rows (ceil(w/4)*4)
    size_t scaledOff = 0, scaledBytes = 0;      // PBO offset/bytes of the downscaled BGRA (0 = none)
    std::vector<TargetLayout> targetLayout;     // per-consumer targets, in caller order
    size_t total = 0;                           // whole PBO payload
};

// Layout/copy info handed from the GL thread's map job to the caller's memcpy.
struct MapResult {
    const uint8_t* ptr = nullptr;
    int format = 0;
    uint32_t w = 0, h = 0;
    size_t mainBytes = 0, alphaOff = 0, scaledOff = 0, scaledBytes = 0;
    uint32_t alphaRowGl = 0;
    std::vector<TargetLayout> targets;
};

class GlThread {
public:
    static GlThread& Instance() {
        static GlThread* t = new GlThread();  // leaked on purpose: outlives all envs, no GL teardown races
        return *t;
    }

    // Start the thread + context once; returns initOk. Safe from any thread.
    bool EnsureInit() {
        std::unique_lock<std::mutex> lk(m_);
        if (!started_) {
            started_ = true;
            th_ = std::thread([this] { ThreadMain(); });
            th_.detach();
        }
        cv_.wait(lk, [this] { return initDone_; });
        return initOk_;
    }
    bool InitOk() const { return initDone_ && initOk_; }
    const std::string& InitErr() const { return initErr_; }
    const std::string& PlatformName() const { return platformName_; }
    bool HaveModifiers() const { return haveModifiers_; }

    void Post(std::function<void()> fn) {
        std::lock_guard<std::mutex> lk(m_);
        q_.push_back(std::move(fn));
        cv_.notify_all();
    }
    size_t QueueDepth() {
        std::lock_guard<std::mutex> lk(m_);
        return q_.size();
    }
    // Run `fn` on the GL thread and wait for it. Must not be called FROM the GL thread. Returns false if
    // the job did not finish in time, in which case it may still be running: the shared wait state outlives
    // this call so a late completion has nothing of ours to touch.
    //
    // Every GPU wait inside a job is already bounded by kFenceTimeoutNs, and a job contains at most the
    // draw fence and the read fence, so a job that has not returned in twice that plus one more interval
    // for the CPU work around them is not slow - the GL thread is wedged, and waiting forever here would
    // take every caller down with it.
    bool Run(const std::function<void()>& fn) {
        struct WaitState {
            std::mutex m;
            std::condition_variable cv;
            bool done = false;
        };
        auto state = std::make_shared<WaitState>();
        Post([state, fn] {
            fn();
            std::lock_guard<std::mutex> lk(state->m);
            state->done = true;
            state->cv.notify_all();
        });
        const auto limit = std::chrono::nanoseconds(kFenceTimeoutNs * 3);
        std::unique_lock<std::mutex> lk(state->m);
        if (state->cv.wait_for(lk, limit, [&] { return state->done; })) return true;
        fprintf(stderr, "[osr-capture] GL thread did not finish a job within %llums; giving up on it\n",
                (unsigned long long)(kFenceTimeoutNs * 3 / 1000000ULL));
        return false;
    }

    // ---- GL-thread-only operations (call via Run/Post) ---------------------------------------------------
    bool ConsumeOnThread(const std::vector<DmabufPlane>& planes, uint64_t modifier, uint32_t w, uint32_t h,
                         int format, const std::string& key, uint32_t dstW, uint32_t dstH,
                         const std::vector<TargetSpec>& targets, const VideoLayer& video, std::string& err,
                         bool& importFailed);
    bool FinishMapOnThread(const std::string& key, MapResult& res, std::string& err);
    void FinishUnmapOnThread(const std::string& key);
    void ReleaseKeyOnThread(const std::string& key);

    std::atomic<int> importFails{0};
    // A demotion is a timed backoff, never permanent: an import failure can be transient (a GPU reset,
    // a driver reload), and a permanent flag meant one bad moment cost CPU conversion of every frame for
    // the life of the process. Each successive demotion doubles the wait, so a genuinely broken GPU
    // settles into probing rarely instead of once per frame.
    std::atomic<long long> demotedUntilMs{0};
    std::atomic<int> demotions{0};

    static long long NowMs() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    bool Demoted() const { return NowMs() < demotedUntilMs.load(); }

private:
    void ThreadMain() {
        bool ok = InitGl();
        {
            std::lock_guard<std::mutex> lk(m_);
            initOk_ = ok;
            initDone_ = true;
            cv_.notify_all();
        }
        if (!ok) return;  // no jobs will be posted (EnsureInit returned false)
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [this] { return !q_.empty(); });
                job = std::move(q_.front());
                q_.pop_front();
            }
            job();
        }
    }

    bool InitGl();
    bool CompileProgram(Program& p, const char* fragBody, std::string& err);
    bool EnsureTarget(GLuint& tex, GLuint& fbo, int& curW, int& curH, int w, int h, std::string& err);
    // Packed formats render fewer texels than they describe pixels, so the viewport and uDstSize differ.
    void DrawToSized(const Program& p, GLuint fbo, int vpW, int vpH, int dstW, int dstH, GLuint srcTex, int srcW, int srcH, bool fromShare = false);
    void DrawTo(const Program& p, GLuint fbo, int dstW, int dstH, GLuint srcTex, int srcW, int srcH, bool fromShare = false) {
        DrawToSized(p, fbo, dstW, dstH, dstW, dstH, srcTex, srcW, srcH, fromShare);
    }
    EGLImageKHR ImportDmabuf(const std::vector<DmabufPlane>& planes, uint64_t modifier, uint32_t w, uint32_t h, std::string& err);
    void DropPending(KeyState& ks);
    // Upload this frame's video layer into the key's texture and arm the draw-time uniforms. Always
    // called (an empty layer just disarms them), so no draw ever inherits the previous frame's layer.
    bool SetVideoLayerOnThread(KeyState& ks, const VideoLayer& video, std::string& err);
    GLuint DummyVideoTex();  // 1x1 black, so texture unit 1 is always complete even with no layer

    std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::function<void()>> q_;
    std::thread th_;
    bool started_ = false, initDone_ = false, initOk_ = false;
    std::string initErr_, platformName_;

    // EGL/GL state (GL-thread-only after init)
    EGLDisplay dpy_ = EGL_NO_DISPLAY;
    EGLContext ctx_ = EGL_NO_CONTEXT;
    EGLSurface surf_ = EGL_NO_SURFACE;
    bool haveModifiers_ = false;
    int flipY_ = 0;  // FS_LINUX_READBACK_FLIP=1
    PFNEGLCREATEIMAGEKHRPROC eglCreateImageKHR_ = nullptr;
    PFNEGLDESTROYIMAGEKHRPROC eglDestroyImageKHR_ = nullptr;
    PFNGLEGLIMAGETARGETTEXTURE2DOESPROC glEGLImageTargetTexture2DOES_ = nullptr;
    Program pUyvy_, pAlpha_, pBgra_, pRgba_, pScale_, pShare_;
    Program pScaleUyvy_, pScaleAlpha_, pScaleRgba_, pI420Luma_, pI420U_, pI420V_;
    std::map<std::string, KeyState> keys_;
    // the layer state DrawToSized binds; set per consume by SetVideoLayerOnThread
    GLuint videoTex_ = 0, dummyVideoTex_ = 0;
    bool videoOn_ = false;
    int videoW_ = 0, videoH_ = 0, videoFormat_ = 0;
};

bool GlThread::CompileProgram(Program& p, const char* fragBody, std::string& err) {
    std::string frag = std::string(kFragPrelude) + fragBody;
    auto compile = [&](GLenum type, const char* src) -> GLuint {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint ok = 0;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[512] = {0};
            glGetShaderInfoLog(s, sizeof(log) - 1, nullptr, log);
            err = std::string("shader compile failed: ") + log;
            glDeleteShader(s);
            return 0;
        }
        return s;
    };
    GLuint vs = compile(GL_VERTEX_SHADER, kVert);
    if (!vs) return false;
    GLuint fs = compile(GL_FRAGMENT_SHADER, frag.c_str());
    if (!fs) {
        glDeleteShader(vs);
        return false;
    }
    p.id = glCreateProgram();
    glAttachShader(p.id, vs);
    glAttachShader(p.id, fs);
    glLinkProgram(p.id);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok = 0;
    glGetProgramiv(p.id, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512] = {0};
        glGetProgramInfoLog(p.id, sizeof(log) - 1, nullptr, log);
        err = std::string("program link failed: ") + log;
        glDeleteProgram(p.id);
        p.id = 0;
        return false;
    }
    p.uTex = glGetUniformLocation(p.id, "uTex");
    p.uFlipY = glGetUniformLocation(p.id, "uFlipY");
    p.uSrcH = glGetUniformLocation(p.id, "uSrcH");
    p.uSrcSize = glGetUniformLocation(p.id, "uSrcSize");
    p.uDstSize = glGetUniformLocation(p.id, "uDstSize");
    p.uVideo = glGetUniformLocation(p.id, "uVideo");
    p.uVideoOn = glGetUniformLocation(p.id, "uVideoOn");
    p.uVideoSize = glGetUniformLocation(p.id, "uVideoSize");
    p.uVideoFormat = glGetUniformLocation(p.id, "uVideoFormat");
    return true;
}

bool GlThread::InitGl() {
    const char* force = std::getenv("FS_LINUX_READBACK");
    if (force && std::strcmp(force, "cpu") == 0) {
        initErr_ = "forced by FS_LINUX_READBACK=cpu";
        return false;
    }
    const char* flip = std::getenv("FS_LINUX_READBACK_FLIP");
    flipY_ = (flip && flip[0] == '1') ? 1 : 0;

    // ---- display ladder: surfaceless (Mesa) -> platform device (NVIDIA) -> default -----------------------
    const char* clientExts = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
    auto eglGetPlatformDisplayEXT_ = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    EGLint maj = 0, min = 0;
    if (eglGetPlatformDisplayEXT_ && HasToken(clientExts, "EGL_MESA_platform_surfaceless")) {
        EGLDisplay d = eglGetPlatformDisplayEXT_(EGL_PLATFORM_SURFACELESS_MESA, (void*)EGL_DEFAULT_DISPLAY, nullptr);
        if (d != EGL_NO_DISPLAY && eglInitialize(d, &maj, &min)) {
            dpy_ = d;
            platformName_ = "surfaceless";
        }
    }
    if (dpy_ == EGL_NO_DISPLAY && eglGetPlatformDisplayEXT_ && HasToken(clientExts, "EGL_EXT_platform_device")) {
        auto eglQueryDevicesEXT_ = (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
        if (eglQueryDevicesEXT_) {
            EGLDeviceEXT devices[16];
            EGLint n = 0;
            if (eglQueryDevicesEXT_(16, devices, &n)) {
                for (EGLint i = 0; i < n && dpy_ == EGL_NO_DISPLAY; ++i) {
                    EGLDisplay d = eglGetPlatformDisplayEXT_(EGL_PLATFORM_DEVICE_EXT, devices[i], nullptr);
                    if (d != EGL_NO_DISPLAY && eglInitialize(d, &maj, &min)) {
                        dpy_ = d;
                        platformName_ = "device";
                    }
                }
            }
        }
    }
    if (dpy_ == EGL_NO_DISPLAY) {
        EGLDisplay d = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (d != EGL_NO_DISPLAY && eglInitialize(d, &maj, &min)) {
            dpy_ = d;
            platformName_ = "default";
        }
    }
    if (dpy_ == EGL_NO_DISPLAY) {
        initErr_ = "no EGL display could be initialized";
        return false;
    }

    const char* dpyExts = eglQueryString(dpy_, EGL_EXTENSIONS);
    if (!HasToken(dpyExts, "EGL_EXT_image_dma_buf_import")) {
        initErr_ = "EGL_EXT_image_dma_buf_import not supported";
        return false;
    }
    haveModifiers_ = HasToken(dpyExts, "EGL_EXT_image_dma_buf_import_modifiers");
    bool surfaceless = HasToken(dpyExts, "EGL_KHR_surfaceless_context");
    bool noConfig = HasToken(dpyExts, "EGL_KHR_no_config_context");

    if (!eglBindAPI(EGL_OPENGL_ES_API)) {
        initErr_ = "eglBindAPI(GLES) failed";
        return false;
    }

    // Config: EGL_NO_CONFIG_KHR when possible; otherwise (and whenever we need a pbuffer because
    // surfaceless contexts are unsupported) a plain RGBA8 pbuffer-capable ES3 config.
    EGLConfig cfg = EGL_NO_CONFIG_KHR;
    if (!noConfig || !surfaceless) {
        const EGLint cfgAttribs[] = {EGL_SURFACE_TYPE, surfaceless ? 0 : EGL_PBUFFER_BIT,
                                     EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
                                     EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
                                     EGL_NONE};
        EGLint numCfg = 0;
        if (!eglChooseConfig(dpy_, cfgAttribs, &cfg, 1, &numCfg) || numCfg < 1) {
            initErr_ = "no ES3 EGLConfig";
            return false;
        }
    }
    const EGLint ctxAttribs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    ctx_ = eglCreateContext(dpy_, cfg, EGL_NO_CONTEXT, ctxAttribs);
    if (ctx_ == EGL_NO_CONTEXT) {
        initErr_ = "eglCreateContext(GLES3) failed";
        return false;
    }
    if (!surfaceless) {
        const EGLint pbAttribs[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
        surf_ = eglCreatePbufferSurface(dpy_, cfg, pbAttribs);
        if (surf_ == EGL_NO_SURFACE) {
            initErr_ = "eglCreatePbufferSurface failed (and no EGL_KHR_surfaceless_context)";
            return false;
        }
    }
    if (!eglMakeCurrent(dpy_, surf_, surf_, ctx_)) {
        initErr_ = "eglMakeCurrent failed";
        return false;
    }

    const char* glExts = (const char*)glGetString(GL_EXTENSIONS);
    if (!HasToken(glExts, "GL_OES_EGL_image")) {
        initErr_ = "GL_OES_EGL_image not supported";
        return false;
    }
    eglCreateImageKHR_ = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
    eglDestroyImageKHR_ = (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImageKHR");
    glEGLImageTargetTexture2DOES_ = (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress("glEGLImageTargetTexture2DOES");
    if (!eglCreateImageKHR_ || !eglDestroyImageKHR_ || !glEGLImageTargetTexture2DOES_) {
        initErr_ = "missing EGLImage entry points";
        return false;
    }

    std::string serr;
    const std::string i420u = std::string(kFragI420ChromaPrelude) + kFragI420U;
    const std::string i420v = std::string(kFragI420ChromaPrelude) + kFragI420V;
    if (!CompileProgram(pUyvy_, kFragUyvy, serr) || !CompileProgram(pAlpha_, kFragAlpha, serr) ||
        !CompileProgram(pBgra_, kFragCopyBgra, serr) || !CompileProgram(pRgba_, kFragCopyRgba, serr) ||
        !CompileProgram(pScale_, kFragScale, serr) || !CompileProgram(pShare_, kFragShare, serr) ||
        !CompileProgram(pScaleUyvy_, kFragScaleUyvy, serr) ||
        !CompileProgram(pScaleAlpha_, kFragScaleAlpha, serr) || !CompileProgram(pScaleRgba_, kFragScaleRgba, serr) ||
        !CompileProgram(pI420Luma_, kFragI420Luma, serr) || !CompileProgram(pI420U_, i420u.c_str(), serr) ||
        !CompileProgram(pI420V_, i420v.c_str(), serr)) {
        initErr_ = serr;
        return false;
    }
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_SCISSOR_TEST);
    return true;
}

bool GlThread::EnsureTarget(GLuint& tex, GLuint& fbo, int& curW, int& curH, int w, int h, std::string& err) {
    if (tex && curW == w && curH == h) return true;
    if (!tex) glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    if (!fbo) glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        err = "FBO incomplete";
        return false;
    }
    curW = w;
    curH = h;
    return true;
}

void GlThread::DrawToSized(const Program& p, GLuint fbo, int vpW, int vpH, int dstW, int dstH, GLuint srcTex, int srcW, int srcH, bool fromShare) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, vpW, vpH);
    glUseProgram(p.id);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, srcTex);
    if (p.uTex >= 0) glUniform1i(p.uTex, 0);
    // the shared scale source is already un-flipped and already has the layer composited into it
    if (p.uFlipY >= 0) glUniform1i(p.uFlipY, fromShare ? 0 : flipY_);
    if (p.uSrcH >= 0) glUniform1i(p.uSrcH, srcH);
    if (p.uSrcSize >= 0) glUniform2i(p.uSrcSize, srcW, srcH);
    if (p.uDstSize >= 0) glUniform2i(p.uDstSize, dstW, dstH);
    // unit 1 is the video layer; leave the active unit back at 0, which the rest of the file assumes.
    // (DummyVideoTex touches the active unit itself, so resolve it BEFORE selecting unit 1.)
    const bool layer = videoOn_ && !fromShare;
    const GLuint vtex = layer ? videoTex_ : DummyVideoTex();
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, vtex);
    glActiveTexture(GL_TEXTURE0);
    if (p.uVideo >= 0) glUniform1i(p.uVideo, 1);
    if (p.uVideoOn >= 0) glUniform1i(p.uVideoOn, layer ? 1 : 0);
    if (p.uVideoSize >= 0) glUniform2i(p.uVideoSize, videoW_, videoH_);
    if (p.uVideoFormat >= 0) glUniform1i(p.uVideoFormat, videoFormat_);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

GLuint GlThread::DummyVideoTex() {
    if (dummyVideoTex_) return dummyVideoTex_;
    glGenTextures(1, &dummyVideoTex_);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, dummyVideoTex_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    const uint8_t black[4] = {0, 0, 0, 0};
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, black);
    glActiveTexture(GL_TEXTURE0);
    return dummyVideoTex_;
}

bool GlThread::SetVideoLayerOnThread(KeyState& ks, const VideoLayer& video, std::string& err) {
    videoOn_ = false;
    videoTex_ = 0;
    videoW_ = videoH_ = videoFormat_ = 0;
    if (!video.w || !video.h || !video.data) return true;
    const uint32_t texels = video.format == 1 ? video.w / 2 : video.w;  // UYVY packs two pixels per texel
    const uint32_t rowBytes = texels * 4;
    if (!texels || (size_t)rowBytes * video.h > video.bytes) {
        err = "video layer buffer too small";
        return false;
    }
    if (!ks.videoTex) glGenTextures(1, &ks.videoTex);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, ks.videoTex);
    if (ks.videoTexW != (int)texels || ks.videoTexH != (int)video.h) {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, (GLsizei)texels, (GLsizei)video.h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        ks.videoTexW = (int)texels;
        ks.videoTexH = (int)video.h;
    }
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (GLsizei)texels, (GLsizei)video.h, GL_RGBA, GL_UNSIGNED_BYTE, video.data);
    glActiveTexture(GL_TEXTURE0);
    videoTex_ = ks.videoTex;
    videoOn_ = true;
    videoW_ = (int)video.w;
    videoH_ = (int)video.h;
    videoFormat_ = video.format;
    return true;
}

EGLImageKHR GlThread::ImportDmabuf(const std::vector<DmabufPlane>& planes, uint64_t modifier, uint32_t w, uint32_t h, std::string& err) {
    static const EGLint fdAttr[4] = {EGL_DMA_BUF_PLANE0_FD_EXT, EGL_DMA_BUF_PLANE1_FD_EXT, EGL_DMA_BUF_PLANE2_FD_EXT, EGL_DMA_BUF_PLANE3_FD_EXT};
    static const EGLint offAttr[4] = {EGL_DMA_BUF_PLANE0_OFFSET_EXT, EGL_DMA_BUF_PLANE1_OFFSET_EXT, EGL_DMA_BUF_PLANE2_OFFSET_EXT, EGL_DMA_BUF_PLANE3_OFFSET_EXT};
    static const EGLint pitchAttr[4] = {EGL_DMA_BUF_PLANE0_PITCH_EXT, EGL_DMA_BUF_PLANE1_PITCH_EXT, EGL_DMA_BUF_PLANE2_PITCH_EXT, EGL_DMA_BUF_PLANE3_PITCH_EXT};
    static const EGLint modLoAttr[4] = {EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT, EGL_DMA_BUF_PLANE1_MODIFIER_LO_EXT, EGL_DMA_BUF_PLANE2_MODIFIER_LO_EXT, EGL_DMA_BUF_PLANE3_MODIFIER_LO_EXT};
    static const EGLint modHiAttr[4] = {EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT, EGL_DMA_BUF_PLANE1_MODIFIER_HI_EXT, EGL_DMA_BUF_PLANE2_MODIFIER_HI_EXT, EGL_DMA_BUF_PLANE3_MODIFIER_HI_EXT};

    size_t nPlanes = planes.size() > 4 ? 4 : planes.size();
    bool useModifier = haveModifiers_ && modifier != kModifierInvalid;
    const uint32_t fourccs[2] = {kFourccArgb8888, kFourccXrgb8888};
    EGLint lastErr = EGL_SUCCESS;
    for (uint32_t fourcc : fourccs) {
        // fd OWNERSHIP (EGL_EXT_image_dma_buf_import): on a SUCCESSFUL import the EGL implementation
        // takes ownership of the plane fds and may close them at any time. Electron still owns the
        // original fds (texture.release() closes them) — importing them directly is a per-frame DOUBLE
        // CLOSE that corrupts the process fd table and, within a few frames, closes recycled fds
        // (Chromium frame-pool dmabufs / IPC) out from under Chromium: the OSR frame pool can no longer
        // recycle its buffers and paints stop entirely. So: pass EGL freshly dup'd fds — on success EGL
        // owns and closes the DUPS; on failure (ownership not transferred) WE close them. Electron's fds
        // are never touched. Fresh dups per attempt so a driver that misbehaves on a failed attempt
        // can't poison the second fourcc try.
        int dupFds[4] = {-1, -1, -1, -1};
        bool dupOk = true;
        for (size_t p = 0; p < nPlanes; ++p) {
            dupFds[p] = fcntl(planes[p].fd, F_DUPFD_CLOEXEC, 3);
            if (dupFds[p] < 0) {
                dupOk = false;
                break;
            }
            D().dupCreated.fetch_add(1);
        }
        if (!dupOk) {
            for (size_t p = 0; p < nPlanes; ++p)
                if (dupFds[p] >= 0) {
                    close(dupFds[p]);
                    D().dupClosedUs.fetch_add(1);
                }
            err = "dup(dmabuf fd) failed";
            return EGL_NO_IMAGE_KHR;
        }
        EGLint a[64];
        int i = 0;
        a[i++] = EGL_WIDTH;
        a[i++] = (EGLint)w;
        a[i++] = EGL_HEIGHT;
        a[i++] = (EGLint)h;
        a[i++] = EGL_LINUX_DRM_FOURCC_EXT;
        a[i++] = (EGLint)fourcc;
        for (size_t p = 0; p < nPlanes; ++p) {
            a[i++] = fdAttr[p];
            a[i++] = dupFds[p];
            a[i++] = offAttr[p];
            a[i++] = (EGLint)planes[p].offset;
            a[i++] = pitchAttr[p];
            a[i++] = (EGLint)planes[p].stride;
            if (useModifier) {
                a[i++] = modLoAttr[p];
                a[i++] = (EGLint)(modifier & 0xffffffffu);
                a[i++] = modHiAttr[p];
                a[i++] = (EGLint)(modifier >> 32);
            }
        }
        a[i] = EGL_NONE;
        EGLImageKHR img = eglCreateImageKHR_(dpy_, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, (EGLClientBuffer) nullptr, a);
        if (img != EGL_NO_IMAGE_KHR) {
            D().imgCreated.fetch_add(1);  // the dup'd fds are now EGL's to close
            return img;
        }
        lastErr = eglGetError();
        D().lastEglErr.store((uint32_t)lastErr);
        for (size_t p = 0; p < nPlanes; ++p) {  // failed import: ownership NOT transferred — close our dups
            close(dupFds[p]);
            D().dupClosedUs.fetch_add(1);
        }
    }
    char buf[128];
    std::snprintf(buf, sizeof(buf), "eglCreateImageKHR(dmabuf) failed (egl 0x%x, modifier 0x%llx, planes %zu)",
                  (unsigned)lastErr, (unsigned long long)modifier, nPlanes);
    err = buf;
    return EGL_NO_IMAGE_KHR;
}

void GlThread::DropPending(KeyState& ks) {
    if (ks.mapped) {
        glBindBuffer(GL_PIXEL_PACK_BUFFER, ks.pbo);
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        ks.mapped = nullptr;
    }
    if (ks.fenceRead) {
        glDeleteSync(ks.fenceRead);
        D().fenceDestroyed.fetch_add(1);
        ks.fenceRead = nullptr;
    }
    ks.pending = false;
}

bool GlThread::ConsumeOnThread(const std::vector<DmabufPlane>& planes, uint64_t modifier, uint32_t w, uint32_t h,
                               int format, const std::string& key, uint32_t dstW, uint32_t dstH,
                               const std::vector<TargetSpec>& targets, const VideoLayer& video,
                               std::string& err, bool& importFailed) {
    importFailed = false;
    uint64_t seq = D().frameSeq.fetch_add(1);
    if (planes.empty() || w == 0 || h == 0) {
        err = "no dmabuf planes / empty frame";
        return false;
    }
    if ((format == 1 || format == 2) && (w & 1)) {
        err = "odd width unsupported for UYVY/UYVA";
        return false;
    }
    for (const TargetSpec& t : targets) {
        if ((t.format == 1 || t.format == 2) && (t.w & 1)) {
            err = "odd target width unsupported for UYVY/UYVA";
            return false;
        }
        if (t.format == 4 && t.w && t.h && (t.w < 2 || t.h < 2)) {
            err = "I420 target smaller than 2x2";
            return false;
        }
    }
    KeyState& ks = keys_[key];
    DropPending(ks);  // stale consume without a finish (shouldn't happen with single-in-flight per key)

    // ---- video layer (before the import, so a bad layer costs no EGLImage) --------------------------------
    if (!SetVideoLayerOnThread(ks, video, err)) {
        Trace8(seq, key, "videoLayer=FAIL");
        return false;
    }

    // ---- import ------------------------------------------------------------------------------------------
    EGLImageKHR img = ImportDmabuf(planes, modifier, w, h, err);
    if (img == EGL_NO_IMAGE_KHR) {
        importFailed = true;
        Trace8(seq, key, "import=FAIL");
        return false;
    }
    GLuint srcTex = 0;
    glGenTextures(1, &srcTex);
    D().texCreated.fetch_add(1);
    glBindTexture(GL_TEXTURE_2D, srcTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    for (int i = 0; i < 8 && glGetError() != GL_NO_ERROR; ++i) {  // clear stale errors before the bind check
    }
    glEGLImageTargetTexture2DOES_(GL_TEXTURE_2D, (GLeglImageOES)img);
    if (glGetError() != GL_NO_ERROR) {
        glDeleteTextures(1, &srcTex);
        D().texDestroyed.fetch_add(1);
        eglDestroyImageKHR_(dpy_, img);
        D().imgDestroyed.fetch_add(1);
        err = "glEGLImageTargetTexture2DOES failed";
        importFailed = true;
        Trace8(seq, key, "import=ok bind=FAIL released(img+tex)");
        return false;
    }

    // ---- layout ------------------------------------------------------------------------------------------
    int mainTexW;
    size_t mainBytes;
    if (format == 1 || format == 2) {
        mainTexW = (int)(w / 2);  // RGBA8 texel = U Y0 V Y1
        mainBytes = (size_t)w * 2 * h;
    } else {
        mainTexW = (int)w;
        mainBytes = (size_t)w * 4 * h;
    }
    uint32_t alphaTexW = (w + 3) / 4;
    uint32_t alphaRowGl = alphaTexW * 4;
    size_t alphaOff = 0, alphaBytes = 0;
    if (format == 2) {
        alphaOff = mainBytes;
        alphaBytes = (size_t)alphaRowGl * h;
    }
    bool wantScaled = dstW > 0 && dstH > 0;
    size_t scaledOff = mainBytes + alphaBytes;
    size_t scaledBytes = wantScaled ? (size_t)dstW * dstH * 4 : 0;
    size_t total = scaledOff + scaledBytes;

    // ---- targets + PBO -----------------------------------------------------------------------------------
    bool ok = EnsureTarget(ks.texMain, ks.fboMain, ks.mainW, ks.mainH, mainTexW, (int)h, err);
    if (ok && format == 2) ok = EnsureTarget(ks.texAlpha, ks.fboAlpha, ks.alphaW, ks.alphaH, (int)alphaTexW, (int)h, err);
    if (ok && wantScaled) ok = EnsureTarget(ks.texScaled, ks.fboScaled, ks.scaledW, ks.scaledH, (int)dstW, (int)dstH, err);

    // ---- per-consumer targets: lay their planes out after the scaled region and allocate the FBOs -------
    struct TargetDraw {
        GLuint fbo;
        int vpW, vpH, dstW, dstH;
        const Program* prog;
        size_t off;
        bool share;  // this plane may read the shared scale source
    };
    std::vector<TargetDraw> tdraws;
    for (size_t i = targets.size(); i < ks.targets.size(); ++i) DeleteTargetState(ks.targets[i]);
    ks.targets.resize(targets.size());
    size_t off = total;
    for (size_t i = 0; i < targets.size() && ok; ++i) {
        TargetState& ts = ks.targets[i];
        ts.layout = TargetLayout();
        const TargetSpec& t = targets[i];
        if (!t.w || !t.h) continue;
        const int tw = (int)t.w, th = (int)t.h;
        ts.layout.dstBytes = TargetBytes(t);
        // plane idx renders vpW x vpH RGBA8 texels; the copy-out trims each GL row (vpW * 4) back to rowDst
        auto plane = [&](int idx, int vpW, int vpH, uint32_t rowDst, const Program& pr) {
            if (!ok) return;
            ok = EnsureTarget(ts.tex[idx], ts.fbo[idx], ts.w[idx], ts.h[idx], vpW, vpH, err);
            if (!ok) return;
            TargetPlane& pl = ts.layout.plane[idx];
            pl.off = off;
            pl.rowGl = (uint32_t)vpW * 4;
            pl.rowDst = rowDst;
            pl.rows = (uint32_t)vpH;
            off += (size_t)pl.rowGl * pl.rows;
            ts.layout.nPlanes = idx + 1;
            tdraws.push_back({ts.fbo[idx], vpW, vpH, tw, th, &pr, pl.off, t.format == 0 || t.format == 1 || t.format == 3});
        };
        if (t.format == 1 || t.format == 2) {
            plane(0, tw / 2, th, (uint32_t)tw * 2, pScaleUyvy_);
            if (t.format == 2) plane(1, (tw + 3) / 4, th, (uint32_t)tw, pScaleAlpha_);
        } else if (t.format == 4) {
            const int cw = tw / 2;
            plane(0, (tw + 3) / 4, th, (uint32_t)tw, pI420Luma_);
            plane(1, (cw + 3) / 4, th / 2, (uint32_t)cw, pI420U_);
            plane(2, (cw + 3) / 4, th / 2, (uint32_t)cw, pI420V_);
        } else {
            plane(0, tw, th, (uint32_t)tw * 4, t.format == 3 ? pScaleRgba_ : pScale_);
        }
    }
    total = off;

    if (ok) {
        if (!ks.pbo) {
            glGenBuffers(1, &ks.pbo);
            D().pboCreated.fetch_add(1);
        }
        glBindBuffer(GL_PIXEL_PACK_BUFFER, ks.pbo);
        if (ks.pboCap != total) {
            glBufferData(GL_PIXEL_PACK_BUFFER, (GLsizeiptr)total, nullptr, GL_STREAM_READ);
            ks.pboCap = total;
        }
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    }
    if (!ok) {
        glDeleteTextures(1, &srcTex);
        D().texDestroyed.fetch_add(1);
        eglDestroyImageKHR_(dpy_, img);
        D().imgDestroyed.fetch_add(1);
        Trace8(seq, key, "import=ok targets=FAIL released(img+tex)");
        return false;
    }

    // ---- draws (the only stage that READS the dmabuf) ----------------------------------------------------
    // Consumers smaller than the render each box-filter the whole frame, so two or more of them read the
    // source two or more times over. Read it once into a shared scale source at the largest of their sizes
    // and let them filter from that. Packed UYVA and I420 targets keep reading the source directly - their
    // plane packing is tied to the source layout - so the share only exists when two plain or UYVY
    // consumers can use it, and when it does not exist nothing below changes.
    int shareW = 0, shareH = 0, shareUsers = 0;
    auto canShare = [](int fmt) { return fmt == 0 || fmt == 1 || fmt == 3; };
    if (wantScaled && (int)dstW < (int)w) {
        shareUsers++;
        if ((int)dstW > shareW) { shareW = (int)dstW; shareH = (int)dstH; }
    }
    for (const TargetSpec& t : targets) {
        if (!t.w || (int)t.w >= (int)w || !canShare(t.format)) continue;
        shareUsers++;
        if ((int)t.w > shareW) { shareW = (int)t.w; shareH = (int)t.h; }
    }
    bool haveShare = false;
    if (shareUsers >= 2 && shareW > 0 && shareH > 0) {
        std::string serr;
        if (EnsureTarget(ks.texShare, ks.fboShare, ks.shareW, ks.shareH, shareW, shareH, serr)) {
            DrawTo(pShare_, ks.fboShare, shareW, shareH, srcTex, (int)w, (int)h);
            haveShare = true;
        }
    }
    const GLuint smallTex = haveShare ? ks.texShare : srcTex;
    const int smallW = haveShare ? shareW : (int)w;
    const int smallH = haveShare ? shareH : (int)h;

    const Program& mainProg = (format == 1 || format == 2) ? pUyvy_ : (format == 3 ? pRgba_ : pBgra_);
    DrawTo(mainProg, ks.fboMain, mainTexW, (int)h, srcTex, (int)w, (int)h);
    if (format == 2) DrawTo(pAlpha_, ks.fboAlpha, (int)alphaTexW, (int)h, srcTex, (int)w, (int)h);
    if (wantScaled) {
        const bool viaShare = haveShare && (int)dstW <= smallW && (int)dstH <= smallH;
        DrawTo(pScale_, ks.fboScaled, (int)dstW, (int)dstH, viaShare ? smallTex : srcTex, viaShare ? smallW : (int)w, viaShare ? smallH : (int)h, viaShare);
    }
    for (const TargetDraw& td : tdraws) {
        const bool viaShare = haveShare && td.share && td.dstW <= smallW && td.dstH <= smallH;
        DrawToSized(*td.prog, td.fbo, td.vpW, td.vpH, td.dstW, td.dstH, viaShare ? smallTex : srcTex, viaShare ? smallW : (int)w, viaShare ? smallH : (int)h, viaShare);
    }
    GLsync fenceDraw = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    D().fenceCreated.fetch_add(1);

    // ---- async readback into the PBO (reads the FBO targets, NOT the dmabuf) -----------------------------
    glBindBuffer(GL_PIXEL_PACK_BUFFER, ks.pbo);
    glBindFramebuffer(GL_FRAMEBUFFER, ks.fboMain);
    glReadPixels(0, 0, mainTexW, (GLsizei)h, GL_RGBA, GL_UNSIGNED_BYTE, (void*)0);
    if (format == 2) {
        glBindFramebuffer(GL_FRAMEBUFFER, ks.fboAlpha);
        glReadPixels(0, 0, (GLsizei)alphaTexW, (GLsizei)h, GL_RGBA, GL_UNSIGNED_BYTE, (void*)(uintptr_t)alphaOff);
    }
    if (wantScaled) {
        glBindFramebuffer(GL_FRAMEBUFFER, ks.fboScaled);
        glReadPixels(0, 0, (GLsizei)dstW, (GLsizei)dstH, GL_RGBA, GL_UNSIGNED_BYTE, (void*)(uintptr_t)scaledOff);
    }
    for (const TargetDraw& td : tdraws) {
        glBindFramebuffer(GL_FRAMEBUFFER, td.fbo);
        glReadPixels(0, 0, (GLsizei)td.vpW, (GLsizei)td.vpH, GL_RGBA, GL_UNSIGNED_BYTE, (void*)(uintptr_t)td.off);
    }
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    GLsync fenceRead = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    D().fenceCreated.fetch_add(1);
    glFlush();

    // ---- wait for the DRAWS only, then release the dmabuf (readback DMA continues in the background) -----
    GLenum r = glClientWaitSync(fenceDraw, GL_SYNC_FLUSH_COMMANDS_BIT, kFenceTimeoutNs);
    glDeleteSync(fenceDraw);
    D().fenceDestroyed.fetch_add(1);
    glDeleteTextures(1, &srcTex);
    D().texDestroyed.fetch_add(1);
    eglDestroyImageKHR_(dpy_, img);
    D().imgDestroyed.fetch_add(1);
    if (D().enabled) {
        GLenum ge = glGetError();
        if (ge != GL_NO_ERROR) D().lastGlErr.store((uint32_t)ge);
    }
    if (r != GL_ALREADY_SIGNALED && r != GL_CONDITION_SATISFIED) {
        glDeleteSync(fenceRead);
        D().fenceDestroyed.fetch_add(1);
        err = "gl draw fence watchdog";
        Trace8(seq, key, "import=ok convert=queued drawFence=TIMEOUT released(img+tex)");
        return false;
    }
    Trace8(seq, key, "import=ok convert=ok readback=queued drawFence=ok released(img+tex) pending=1");

    ks.pending = true;
    ks.fenceRead = fenceRead;
    ks.format = format;
    ks.w = w;
    ks.h = h;
    ks.mainBytes = mainBytes;
    ks.alphaOff = alphaOff;
    ks.alphaRowGl = alphaRowGl;
    ks.scaledOff = scaledOff;
    ks.scaledBytes = scaledBytes;
    ks.targetLayout.clear();
    for (const TargetState& ts : ks.targets) ks.targetLayout.push_back(ts.layout);
    ks.total = total;
    return true;
}

bool GlThread::FinishMapOnThread(const std::string& key, MapResult& res, std::string& err) {
    auto it = keys_.find(key);
    if (it == keys_.end() || !it->second.pending) {
        err = "no pending consume for key";
        return false;
    }
    KeyState& ks = it->second;
    if (ks.fenceRead) {
        GLenum r = glClientWaitSync(ks.fenceRead, GL_SYNC_FLUSH_COMMANDS_BIT, kFenceTimeoutNs);
        glDeleteSync(ks.fenceRead);
        D().fenceDestroyed.fetch_add(1);
        ks.fenceRead = nullptr;
        if (r != GL_ALREADY_SIGNALED && r != GL_CONDITION_SATISFIED) {
            ks.pending = false;
            err = "gl readback fence watchdog";
            return false;
        }
    }
    glBindBuffer(GL_PIXEL_PACK_BUFFER, ks.pbo);
    void* p = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, (GLsizeiptr)ks.total, GL_MAP_READ_BIT);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    if (!p) {
        ks.pending = false;
        err = "glMapBufferRange failed";
        return false;
    }
    ks.mapped = p;
    res.ptr = (const uint8_t*)p;
    res.format = ks.format;
    res.w = ks.w;
    res.h = ks.h;
    res.mainBytes = ks.mainBytes;
    res.alphaOff = ks.alphaOff;
    res.alphaRowGl = ks.alphaRowGl;
    res.scaledOff = ks.scaledOff;
    res.scaledBytes = ks.scaledBytes;
    res.targets = ks.targetLayout;
    return true;
}

void GlThread::FinishUnmapOnThread(const std::string& key) {
    auto it = keys_.find(key);
    if (it == keys_.end()) return;
    DropPending(it->second);
}

void GlThread::ReleaseKeyOnThread(const std::string& key) {
    auto it = keys_.find(key);
    if (it == keys_.end()) return;
    KeyState& ks = it->second;
    DropPending(ks);
    if (ks.fboMain) glDeleteFramebuffers(1, &ks.fboMain);
    if (ks.fboAlpha) glDeleteFramebuffers(1, &ks.fboAlpha);
    if (ks.fboScaled) glDeleteFramebuffers(1, &ks.fboScaled);
    if (ks.fboShare) glDeleteFramebuffers(1, &ks.fboShare);
    if (ks.texMain) glDeleteTextures(1, &ks.texMain);
    if (ks.texAlpha) glDeleteTextures(1, &ks.texAlpha);
    if (ks.texScaled) glDeleteTextures(1, &ks.texScaled);
    if (ks.texShare) glDeleteTextures(1, &ks.texShare);
    if (ks.videoTex) {
        if (videoTex_ == ks.videoTex) {  // never leave the draw state pointing at a deleted texture
            videoTex_ = 0;
            videoOn_ = false;
        }
        glDeleteTextures(1, &ks.videoTex);
    }
    for (TargetState& ts : ks.targets) DeleteTargetState(ts);
    if (ks.pbo) {
        glDeleteBuffers(1, &ks.pbo);
        D().pboDestroyed.fetch_add(1);
    }
    keys_.erase(it);
}

std::atomic<bool> g_loggedBackend{false};
std::atomic<bool> g_loggedDemotion{false};

void LogBackendOnce(GlThread& t, bool ok) {
    bool expected = false;
    if (!g_loggedBackend.compare_exchange_strong(expected, true)) return;
    if (ok) {
        std::fprintf(stderr, "[osr-capture] linux readback backend: egl-gles3 (platform=%s, modifiers=%s)\n",
                     t.PlatformName().c_str(), t.HaveModifiers() ? "yes" : "no");
    } else {
        std::fprintf(stderr, "[osr-capture] linux readback backend: cpu (%s)\n", t.InitErr().c_str());
    }
}

// Rate-limited (~1/s) lifecycle log; called from the public Consume/Finish (any thread). See the Diag
// comment for how to read it.
void MaybeLogDiag(GlThread& t) {
    Diag& d = D();
    if (!d.enabled) return;
    int64_t now = NowMs();
    int64_t last = d.lastLogMs.load();
    if (now - last < 1000 || !d.lastLogMs.compare_exchange_strong(last, now)) return;
    uint64_t imgC = d.imgCreated.load(), imgD = d.imgDestroyed.load();
    uint64_t texC = d.texCreated.load(), texD = d.texDestroyed.load();
    uint64_t fenC = d.fenceCreated.load(), fenD = d.fenceDestroyed.load();
    uint64_t pboC = d.pboCreated.load(), pboD = d.pboDestroyed.load();
    uint64_t dupC = d.dupCreated.load(), dupU = d.dupClosedUs.load();
    int64_t lc = d.lastConsumeMs.load(), lf = d.lastFinishMs.load();
    std::fprintf(stderr,
                 "[osr-capture] gpu-diag q=%zu live(img=%lld tex=%lld fence=%lld pbo=%lld dupFd~%lld) "
                 "cum(img %llu/%llu tex %llu/%llu fence %llu/%llu pbo %llu/%llu dup %llu/%llu) "
                 "calls(consume=%llu fail=%llu finish=%llu fail=%llu) "
                 "last(consume=%lldms finish=%lldms ago) err(gl=0x%x egl=0x%x)\n",
                 t.QueueDepth(),
                 (long long)(imgC - imgD), (long long)(texC - texD), (long long)(fenC - fenD),
                 (long long)(pboC - pboD), (long long)(dupC - dupU - imgC) /* dups EGL owns are ~1/img */,
                 (unsigned long long)imgC, (unsigned long long)imgD, (unsigned long long)texC, (unsigned long long)texD,
                 (unsigned long long)fenC, (unsigned long long)fenD, (unsigned long long)pboC, (unsigned long long)pboD,
                 (unsigned long long)dupC, (unsigned long long)dupU,
                 (unsigned long long)d.consumeCalls.load(), (unsigned long long)d.consumeFails.load(),
                 (unsigned long long)d.finishCalls.load(), (unsigned long long)d.finishFails.load(),
                 (long long)(lc ? now - lc : -1), (long long)(lf ? now - lf : -1),
                 (unsigned)d.lastGlErr.load(), (unsigned)d.lastEglErr.load());
}

}  // namespace

bool Init() {
    GlThread& t = GlThread::Instance();
    bool ok = t.EnsureInit();
    LogBackendOnce(t, ok);
    return ok && !t.Demoted();
}

bool Initialized() { return GlThread::Instance().InitOk(); }

bool Available() {
    GlThread& t = GlThread::Instance();
    return t.InitOk() && !t.Demoted();
}

const char* BackendName() {
    GlThread& t = GlThread::Instance();
    if (t.InitOk()) return t.Demoted() ? "cpu" : "egl-gles3";
    return g_loggedBackend.load() ? "cpu" : "none";
}

bool Consume(const std::vector<DmabufPlane>& planes, uint64_t modifier, uint32_t w, uint32_t h,
             int format, const std::string& key, uint32_t dstW, uint32_t dstH,
             const std::vector<TargetSpec>& targets, const VideoLayer& video, std::string& err, bool& importFailed) {
    importFailed = false;
    GlThread& t = GlThread::Instance();
    if (!t.InitOk() || t.Demoted()) {
        err = "linux gpu readback unavailable";
        return false;
    }
    D().consumeCalls.fetch_add(1);
    D().lastConsumeMs.store(NowMs());
    bool ok = false;
    std::string e;
    bool impFail = false;
    t.Run([&] { ok = t.ConsumeOnThread(planes, modifier, w, h, format, key, dstW, dstH, targets, video, e, impFail); });
    MaybeLogDiag(t);
    if (ok) {
        t.importFails.store(0);
        t.demotions.store(0);
        return true;
    }
    D().consumeFails.fetch_add(1);
    err = e;
    importFailed = impFail;
    // Back off after repeated import failures (e.g. WSL's virtual GPU): stop burning a GPU
    // round-trip per frame; callers take the CPU path via Available() until the deadline passes.
    if (impFail && t.importFails.fetch_add(1) + 1 >= kImportFailDemotion) {
        t.importFails.store(0);
        const int n = t.demotions.fetch_add(1) + 1;
        long long waitMs = kDemotionBaseMs;
        for (int i = 1; i < n && waitMs < kDemotionMaxMs; ++i) waitMs *= 2;
        if (waitMs > kDemotionMaxMs) waitMs = kDemotionMaxMs;
        t.demotedUntilMs.store(GlThread::NowMs() + waitMs);
        bool expected = false;
        if (g_loggedDemotion.compare_exchange_strong(expected, true)) {
            std::fprintf(stderr, "[osr-capture] linux gpu readback on cpu for %lldms after %d import failures (%s)\n",
                         waitMs, kImportFailDemotion, e.c_str());
        }
    }
    return false;
}

bool Consume(const std::vector<DmabufPlane>& planes, uint64_t modifier, uint32_t w, uint32_t h,
             int format, const std::string& key, uint32_t dstW, uint32_t dstH,
             const std::vector<TargetSpec>& targets, std::string& err, bool& importFailed) {
    return Consume(planes, modifier, w, h, format, key, dstW, dstH, targets, VideoLayer(), err, importFailed);
}

bool Consume(const std::vector<DmabufPlane>& planes, uint64_t modifier, uint32_t w, uint32_t h,
             int format, const std::string& key, uint32_t dstW, uint32_t dstH, std::string& err, bool& importFailed) {
    return Consume(planes, modifier, w, h, format, key, dstW, dstH, std::vector<TargetSpec>(), err, importFailed);
}

bool Finish(const std::string& key, uint8_t* dst, size_t dstSize, uint8_t* scaledDst, size_t scaledSize,
            const std::vector<TargetDst>& targetDsts, std::string& err) {
    GlThread& t = GlThread::Instance();
    if (!t.InitOk()) {
        err = "linux gpu readback unavailable";
        return false;
    }
    D().finishCalls.fetch_add(1);
    D().lastFinishMs.store(NowMs());
    bool ok = false;
    std::string e;
    MapResult res;
    t.Run([&] { ok = t.FinishMapOnThread(key, res, e); });
    if (!ok) {
        D().finishFails.fetch_add(1);
        err = e;
        return false;
    }

    // Large copy-out on the CALLING (libuv) thread — the GL thread stays free for other keys' draws.
    if (res.format == 2) {
        size_t n = res.mainBytes < dstSize ? res.mainBytes : dstSize;
        std::memcpy(dst, res.ptr, n);  // UYVY plane
        if (dstSize > res.mainBytes) {
            uint8_t* adst = dst + res.mainBytes;
            size_t aCap = dstSize - res.mainBytes;
            const uint8_t* asrc = res.ptr + res.alphaOff;
            if (res.alphaRowGl == res.w) {
                size_t an = (size_t)res.w * res.h;
                std::memcpy(adst, asrc, an < aCap ? an : aCap);
            } else {
                // w % 4 != 0: trim each GL row (ceil(w/4)*4 bytes) to w bytes
                for (uint32_t y = 0; y < res.h; ++y) {
                    size_t off = (size_t)y * res.w;
                    if (off >= aCap) break;
                    size_t rn = res.w < aCap - off ? res.w : aCap - off;
                    std::memcpy(adst + off, asrc + (size_t)y * res.alphaRowGl, rn);
                }
            }
        }
    } else {
        size_t n = res.mainBytes < dstSize ? res.mainBytes : dstSize;
        std::memcpy(dst, res.ptr, n);
    }
    if (scaledDst && scaledSize > 0 && res.scaledBytes > 0) {
        size_t n = res.scaledBytes < scaledSize ? res.scaledBytes : scaledSize;
        std::memcpy(scaledDst, res.ptr + res.scaledOff, n);
    }
    for (size_t i = 0; i < targetDsts.size() && i < res.targets.size(); ++i) {
        uint8_t* td = targetDsts[i].data;
        const size_t cap = targetDsts[i].size;
        if (!td || !cap) continue;
        const TargetLayout& tl = res.targets[i];
        size_t dstOff = 0;  // a target's planes are tightly packed in the destination, in plane order
        for (int pi = 0; pi < tl.nPlanes && dstOff < cap; ++pi) {
            const TargetPlane& pl = tl.plane[pi];
            const uint8_t* src = res.ptr + pl.off;
            if (pl.rowGl == pl.rowDst) {
                size_t n = (size_t)pl.rowDst * pl.rows;
                if (n > cap - dstOff) n = cap - dstOff;
                std::memcpy(td + dstOff, src, n);
                dstOff += n;
                continue;
            }
            // the texel packing rounded the row up: trim each GL row back to rowDst bytes
            for (uint32_t y = 0; y < pl.rows && dstOff < cap; ++y) {
                size_t n = pl.rowDst < cap - dstOff ? pl.rowDst : cap - dstOff;
                std::memcpy(td + dstOff, src + (size_t)y * pl.rowGl, n);
                dstOff += n;
            }
        }
    }

    // Recycle the PBO (unmap) on the GL thread; fire-and-forget — the queue orders it before any
    // subsequent consume for this key (the caller only reuses a key after finish returns).
    GlThread* tp = &t;
    std::string k = key;
    tp->Post([tp, k] { tp->FinishUnmapOnThread(k); });
    return true;
}

bool Finish(const std::string& key, uint8_t* dst, size_t dstSize, uint8_t* scaledDst, size_t scaledSize, std::string& err) {
    return Finish(key, dst, dstSize, scaledDst, scaledSize, std::vector<TargetDst>(), err);
}

void ReleaseKey(const std::string& key) {
    GlThread* tp = &GlThread::Instance();
    if (!tp->InitOk()) return;
    std::string k = key;
    tp->Post([tp, k] { tp->ReleaseKeyOnThread(k); });
}

}  // namespace linuxgpu
}  // namespace osrcap

#endif  // __linux__
