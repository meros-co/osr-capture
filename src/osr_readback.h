// Platform readback backends: given a shared-texture handle (Windows/macOS) or dmabuf planes (Linux),
// produce a tightly-packed BGRA/RGBA buffer (width * 4 * height bytes). Each is called from an N-API
// async worker (background thread), so they must be self-contained and thread-safe.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace osrcap {

// A per-consumer output produced from the same shared texture in the same GPU pass as the main readback:
// the source box-downscaled to w x h and packed as `format` (0 BGRA, 1 UYVY, 2 UYVA, 4 I420 planar
// Y/U/V, BT.601 limited range — what a video encoder wants). One render can feed outputs of several
// resolutions with one readback (render once at the largest, downscale the rest).
struct TargetSpec {
    uint32_t w = 0;
    uint32_t h = 0;
    int format = 0;
};
// Destination for one target's bytes in ReadbackFinish (caller-owned, at least TargetBytes(spec) long).
struct TargetDst {
    uint8_t* data = nullptr;
    size_t size = 0;
};
inline size_t TargetBytes(const TargetSpec& t) {
    size_t px = static_cast<size_t>(t.w) * t.h;
    if (t.format == 4) return px + 2 * (static_cast<size_t>(t.w / 2) * (t.h / 2));
    return t.format == 1 ? px * 2 : (t.format == 2 ? px * 3 : px * 4);
}
// Whether this platform's GPU backend produces TargetSpec outputs (Windows D3D11 today).
bool TargetsSupported();

// format: 0 = BGRA (raw), 1 = UYVY (opaque), 2 = UYVA (colour + alpha), 3 = RGBA (swizzle). All three
// platforms convert on the GPU — Windows HLSL compute, macOS Metal compute, Linux GLES3 shader — and each
// falls back to the CPU converter only when its GPU path is unavailable for the device or the frame.
#if defined(_WIN32) || defined(__APPLE__)
// Windows: `handle` is a HANDLE to the shared D3D11 texture.
// macOS:   `handle` is an IOSurface* (valid in this process).
bool ReadbackHandle(uintptr_t handle, uint32_t width, uint32_t height, int format, std::vector<uint8_t>& out, std::string& err);
#endif

#if defined(__APPLE__)
// One-time init of the macOS GPU (Metal) readback; true when the two-phase exports should be installed.
// Idempotent, thread-safe (called from every env's addon Init). Mirrors LinuxGpuReadbackInit below: with no
// Metal device the exports stay absent and FreeShow's `typeof readbackConsume === "function"` probe keeps it
// on the single-phase `readback`, which then uses the CPU IOSurface path.
bool MacGpuReadbackInit();
#endif

#if defined(_WIN32) || defined(__APPLE__)
// Two-phase readback (Windows/macOS), so the CALLER can release the Electron shared texture as soon as the GPU has
// consumed it — instead of pinning it for the whole (slow) PCIe read-back, which drains Electron's frame pool
// and stalls the main process. Consume: open the shared texture, GPU-convert into an internal buffer keyed by
// `key`, and return only once the GPU has finished reading the shared texture. Finish: copy that buffer to
// `out` (the slow part). A `key` may have at most one consume outstanding at a time.
// dstW/dstH > 0 also GPU-downscales the shared BGRA into a small buffer (server/stage) in the same pass;
// ReadbackFinish then fills `scaledDst` (scaledSize bytes) with that tightly-packed BGRA. Pass 0 / nullptr to skip.
bool ReadbackConsume(uintptr_t handle, uint32_t width, uint32_t height, int format, const std::string& key, uint32_t dstW, uint32_t dstH, std::string& err);
bool ReadbackFinish(const std::string& key, uint8_t* dst, size_t dstSize, uint8_t* scaledDst, size_t scaledSize, std::string& err);
// Same, with per-consumer targets (see TargetSpec). `targetDsts` must have one entry per target given to consume.
bool ReadbackConsume(uintptr_t handle, uint32_t width, uint32_t height, int format, const std::string& key, uint32_t dstW, uint32_t dstH, const std::vector<TargetSpec>& targets, std::string& err);
bool ReadbackFinish(const std::string& key, uint8_t* dst, size_t dstSize, uint8_t* scaledDst, size_t scaledSize, const std::vector<TargetDst>& targetDsts, std::string& err);
void ReadbackReleaseKey(const std::string& key);  // drop any pending consume for `key` (cleanup)

// SINGLE-DISPATCH readback (plan §11 fix #1 — collapse the two-phase Consume/Finish into ONE N-API async op).
// Runs the whole readback on one borrowed context in one Execute(): open + GPU-convert + WaitGpu + copy-out.
// `onGpuDone` is invoked EXACTLY ONCE, on the SAME worker thread, the instant the GPU has finished reading the
// shared texture (right after WaitGpu, BEFORE the slow PCIe copy-out) — so the caller releases the Electron
// shared texture as EARLY as the old two-phase Consume did (frame-pool-drain protection preserved) without a
// second dispatch+resume hop or the worker-JS-loop hop between the phases. `dst`/`scaledDst` are the caller's
// V8 output buffers (filled here, off the JS thread). No `key`/pending map: the context is held for the whole
// op and released at the end.
bool ReadbackOnce(uintptr_t handle, uint32_t width, uint32_t height, int format, uint32_t dstW, uint32_t dstH,
                  uint8_t* dst, size_t dstSize, uint8_t* scaledDst, size_t scaledSize,
                  const std::function<void()>& onGpuDone, std::string& err);

#if defined(_WIN32)
// Stage-0 harness for the global copy pool: hammer it with `concurrency` caller threads each doing
// `iterations` copies of `bytes` through the pool, verifying each copy is byte-exact. Returns false (and
// nonzero `mismatches`) on any corruption; hangs only if the pool deadlocks (that IS the test). `workers` =
// the pool's CALIBRATED active worker count after the run (plan §11 CV#1 knee-seeker — on cacheable test
// memory only its convergence matters, not the value); `ms` = wall time. No GPU/D3D — pure CPU validation.
bool CopyPoolSelfTest(uint32_t bytes, uint32_t concurrency, uint32_t iterations, uint32_t& workers, uint32_t& mismatches, double& ms);

// Telemetry (plan §11 CV#1 "observability, not knobs"): the copy pool's calibrated operating point
// (`active_n` base). Exposed so FreeShow can log it alongside [SEND-STATS] for send-starvation attribution.
unsigned CopyPoolActiveWorkers();
#endif  // _WIN32 (copy-pool harness/telemetry)

// Telemetry: the readback backend actually in use. Read-only — no setter.
//   Windows: "d3d11on12" (D3D12 READBACK heap, cached copy-out, copy-queue fence) or "d3d11" (classic WC
//            staging + WaitGpu; also the §5-ladder fallback and the FS_READBACK=d3d11 diagnostic selector).
//   macOS:   "metal" (IOSurface wrapped zero-copy as an MTLTexture, compute convert/downscale) or
//            "iosurface-cpu" (no Metal device / non-BGRA surface / unsupported width).
//   "none" before the first device init.
const char* ReadbackBackend();
#endif  // _WIN32 || __APPLE__

#if defined(__linux__)
struct DmabufPlane {
    int fd;
    uint32_t stride;
    uint64_t offset;
    uint64_t size;
};
// Single-phase readback. GPU (EGL dmabuf import + shader convert, any modifier) when available, else the
// mmap+CPU path (LINEAR only). `poolKey` (may be empty) keys the GPU path's cached FBO/PBO per output.
bool ReadbackDmabuf(const std::vector<DmabufPlane>& planes, uint64_t modifier, uint32_t width, uint32_t height, int format, const std::string& poolKey, std::vector<uint8_t>& out, std::string& err);

// One-time init of the Linux GPU (EGL/GLES3) readback; true when the two-phase exports should be
// installed. Idempotent, thread-safe (called from every env's addon Init). See LINUX_GPU_READBACK.md.
bool LinuxGpuReadbackInit();

// Two-phase readback (Linux flavour of the Windows contract above): Consume imports the dmabuf, runs the
// GPU convert (+ dstW/dstH downscale) and returns once the GPU has finished READING the dmabuf — the
// caller releases the Electron texture then. Finish waits the async PBO readback and copies out. A `key`
// may have at most one consume outstanding at a time. When a frame's GPU import fails, consume falls
// back to the CPU path internally (LINEAR only) so the contract holds either way.
bool ReadbackConsume(const std::vector<DmabufPlane>& planes, uint64_t modifier, uint32_t width, uint32_t height, int format, const std::string& key, uint32_t dstW, uint32_t dstH, std::string& err);
bool ReadbackFinish(const std::string& key, uint8_t* dst, size_t dstSize, uint8_t* scaledDst, size_t scaledSize, std::string& err);
void ReadbackReleaseKey(const std::string& key);  // drop pending state + cached GL resources for `key`

// Single-dispatch readback (parity with the Windows ReadbackOnce): consume + onGpuDone + finish.
bool ReadbackOnce(const std::vector<DmabufPlane>& planes, uint64_t modifier, uint32_t width, uint32_t height, int format, const std::string& key, uint32_t dstW, uint32_t dstH,
                  uint8_t* dst, size_t dstSize, uint8_t* scaledDst, size_t scaledSize,
                  const std::function<void()>& onGpuDone, std::string& err);

// Telemetry: "egl-gles3" (GPU path active), "cpu" (unavailable/forced/demoted), "none" (before init).
const char* ReadbackBackend();
#endif

}  // namespace osrcap
