// Linux GPU readback backend: import the Electron OSR dmabuf as an EGLImage/GL texture on a dedicated
// GL thread, convert (UYVY/UYVA/BGRA/RGBA) + optionally downscale with fragment shaders, and read back
// only the small converted result through a PBO. See LINUX_GPU_READBACK.md for the full design.
//
// All functions are thread-safe (they marshal onto the single GL thread internally) and may be called
// from any libuv worker. Init() is idempotent and cheap after the first call.
#pragma once

#if defined(__linux__)

#include <cstdint>
#include <string>
#include <vector>

#include "osr_readback.h"  // DmabufPlane

namespace osrcap {
namespace linuxgpu {

// Start the GL thread and create the headless EGL/GLES3 context (once). Returns true when the GPU path
// is usable (display + context + EGL_EXT_image_dma_buf_import + GL_OES_EGL_image + shaders all OK).
// Honors FS_LINUX_READBACK=cpu (forces false). Logs the chosen backend to stderr, once.
bool Init();

// Init() succeeded (regardless of later runtime demotion) — i.e. Finish/ReleaseKey may still find state.
bool Initialized();

// Init() succeeded AND the path has not been demoted by repeated per-frame import failures. Consume
// should only be attempted when this is true.
bool Available();

// "egl-gles3" when the GPU path is active, "cpu" when unavailable/forced/demoted, "none" before Init().
const char* BackendName();

// Phase 1: import the dmabuf, run the convert (+ dstW/dstH downscale) draws, kick the async PBO
// readback, and return once the GPU has finished READING the dmabuf — the caller may release the
// Electron shared texture immediately. One pending consume per key. On failure `importFailed` tells the
// caller the EGLImage import itself failed (candidate for CPU fallback / demotion) as opposed to a
// size/format precondition.
bool Consume(const std::vector<DmabufPlane>& planes, uint64_t modifier, uint32_t width, uint32_t height,
             int format, const std::string& key, uint32_t dstW, uint32_t dstH, std::string& err, bool& importFailed);

// Phase 2: wait for the readback DMA, map the PBO, and copy the converted bytes into `dst` (and the
// downscaled BGRA into `scaledDst` when requested at consume). The (large) memcpy runs on the CALLING
// thread — only the fence wait + map/unmap occupy the GL thread.
bool Finish(const std::string& key, uint8_t* dst, size_t dstSize, uint8_t* scaledDst, size_t scaledSize, std::string& err);

// Free a key's cached GL resources (FBO textures, PBO, any pending fence/mapping). No-op if unknown.
void ReleaseKey(const std::string& key);

}  // namespace linuxgpu
}  // namespace osrcap

#endif  // __linux__
