// Linux backend: Electron passes the frame as dmabuf plane(s) ({fd, stride, offset, size}) plus a DRM
// format modifier. Two paths (LINUX_GPU_READBACK.md):
//   GPU (preferred): import the dmabuf as an EGLImage/GL texture, convert (+downscale) with fragment
//   shaders, and read back only the small result through a PBO — handles ANY modifier (LINEAR and
//   tiled), and exposes the same two-phase Consume/Finish contract as Windows so the caller can release
//   the Electron texture as soon as the GPU has consumed it. Implemented in readback_linux_gpu.cc.
//   CPU (fallback): for LINEAR (modifier 0 / DRM_FORMAT_MOD_INVALID) buffers only, mmap the fd, copy
//   rows by stride and convert with convert.cc. Used when EGL init fails (no display/extensions,
//   FS_LINUX_READBACK=cpu) or when a frame's EGLImage import fails at runtime (e.g. WSL's virtual GPU;
//   repeated failures demote the GPU path for good). Tiled buffers without GPU import still error.
#include "osr_readback.h"

#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <map>
#include <mutex>

#include "convert.h"
#include "readback_linux_gpu.h"

namespace osrcap {

namespace {
// DRM_FORMAT_MOD_LINEAR
constexpr uint64_t kModifierLinear = 0;
// DRM_FORMAT_MOD_INVALID is sometimes reported for implicit/linear layouts
constexpr uint64_t kModifierInvalid = 0x00ffffffffffffffULL;

// mmap the (LINEAR) dmabuf and copy it out as tightly-packed BGRA. The original CPU readback.
bool ReadLinearBgra(const std::vector<DmabufPlane>& planes, uint64_t modifier, uint32_t width, uint32_t height, std::vector<uint8_t>& out, std::string& err) {
    if (planes.empty()) {
        err = "no dmabuf planes";
        return false;
    }
    if (modifier != kModifierLinear && modifier != kModifierInvalid) {
        // Non-linear (tiled/compressed) layout: CPU-unmappable; only the GPU (EGL import) path can read it.
        err = "non-linear dmabuf modifier requires GPU import (EGL unavailable)";
        return false;
    }

    const DmabufPlane& plane = planes[0];  // single-plane BGRA/RGBA
    const size_t mapLength = static_cast<size_t>(plane.offset) + static_cast<size_t>(plane.stride) * height;

    void* mapped = mmap(nullptr, mapLength, PROT_READ, MAP_SHARED, plane.fd, 0);
    if (mapped == MAP_FAILED) {
        err = "mmap of dmabuf failed";
        return false;
    }

    const uint8_t* base = static_cast<const uint8_t*>(mapped) + plane.offset;
    const size_t rowBytes = static_cast<size_t>(width) * 4;
    out.resize(rowBytes * height);
    for (uint32_t y = 0; y < height; ++y) {
        std::memcpy(out.data() + static_cast<size_t>(y) * rowBytes, base + static_cast<size_t>(y) * plane.stride, rowBytes);
    }

    munmap(mapped, mapLength);
    return true;
}

// CPU-fallback pending frames for the two-phase contract: consume stores the converted bytes here,
// finish copies them out. Only used when the GPU path is unavailable / a frame's import failed.
struct CpuPending {
    std::vector<uint8_t> main;
    std::vector<uint8_t> scaled;
};
std::map<std::string, CpuPending> g_cpuPending;
std::mutex g_cpuPendingMutex;

bool CpuConsume(const std::vector<DmabufPlane>& planes, uint64_t modifier, uint32_t width, uint32_t height, int format, const std::string& key, uint32_t dstW, uint32_t dstH, std::string& err) {
    CpuPending p;
    std::vector<uint8_t> bgra;
    if (!ReadLinearBgra(planes, modifier, width, height, bgra, err)) return false;
    if (dstW > 0 && dstH > 0) DownscaleBgraRaw(bgra.data(), width, height, dstW, dstH, p.scaled);
    ConvertBgraInPlace(bgra, width, height, format);  // 1/2/3 convert in place; 0 stays BGRA
    p.main = std::move(bgra);
    std::lock_guard<std::mutex> lock(g_cpuPendingMutex);
    g_cpuPending[key] = std::move(p);
    return true;
}

std::atomic<uint64_t> g_anonKeySeq{0};
}  // namespace

bool LinuxGpuReadbackInit() { return linuxgpu::Init(); }

const char* ReadbackBackend() { return linuxgpu::BackendName(); }

// Single-phase readback (the `readback` export / probe): GPU consume+finish back-to-back when
// available (keyed `sp#<poolKey>` so the FBO/PBO cache is reused per output), else the CPU path.
bool ReadbackDmabuf(const std::vector<DmabufPlane>& planes, uint64_t modifier, uint32_t width, uint32_t height, int format, const std::string& poolKey, std::vector<uint8_t>& out, std::string& err) {
    if (linuxgpu::Available()) {
        const std::string key = poolKey.empty() ? ("sp#anon" + std::to_string(g_anonKeySeq.fetch_add(1))) : ("sp#" + poolKey);
        std::string gerr;
        bool importFailed = false;
        if (linuxgpu::Consume(planes, modifier, width, height, format, key, 0, 0, gerr, importFailed)) {
            size_t px = static_cast<size_t>(width) * height;
            out.resize(format == 1 ? px * 2 : (format == 2 ? px * 3 : px * 4));
            bool ok = linuxgpu::Finish(key, out.data(), out.size(), nullptr, 0, err);
            if (poolKey.empty()) linuxgpu::ReleaseKey(key);
            return ok;
        }
        // fall through to the CPU path (LINEAR only); keep the GPU error for a combined message
        std::string cerr2;
        std::vector<uint8_t> bgra;
        if (ReadLinearBgra(planes, modifier, width, height, bgra, cerr2)) {
            out = std::move(bgra);
            ConvertBgraInPlace(out, width, height, format);
            return true;
        }
        err = gerr + "; cpu fallback: " + cerr2;
        return false;
    }

    if (!ReadLinearBgra(planes, modifier, width, height, out, err)) return false;
    // format 1 = UYVY, 2 = UYVA, 3 = RGBA: convert the BGRA we just read; 0 leaves it as BGRA.
    ConvertBgraInPlace(out, width, height, format);
    return true;
}

// Two-phase readback (Linux flavour of the Windows contract, see osr_readback.h): consume returns once
// the dmabuf can be released; finish does the copy-out. GPU when possible; CPU fallback keeps the
// contract for LINEAR buffers so the JS caller never has to care which path ran.
bool ReadbackConsume(const std::vector<DmabufPlane>& planes, uint64_t modifier, uint32_t width, uint32_t height, int format, const std::string& key, uint32_t dstW, uint32_t dstH, std::string& err) {
    if (linuxgpu::Available()) {
        std::string gerr;
        bool importFailed = false;
        if (linuxgpu::Consume(planes, modifier, width, height, format, key, dstW, dstH, gerr, importFailed)) {
            // drop any stale CPU-fallback pending for this key (a consumed-but-never-finished frame from an
            // earlier import failure) so finish can't return old pixels
            std::lock_guard<std::mutex> lock(g_cpuPendingMutex);
            g_cpuPending.erase(key);
            return true;
        }
        std::string cerr2;
        if (CpuConsume(planes, modifier, width, height, format, key, dstW, dstH, cerr2)) return true;
        err = gerr + "; cpu fallback: " + cerr2;
        return false;
    }
    return CpuConsume(planes, modifier, width, height, format, key, dstW, dstH, err);
}

bool ReadbackFinish(const std::string& key, uint8_t* dst, size_t dstSize, uint8_t* scaledDst, size_t scaledSize, std::string& err) {
    // A CPU-fallback consume takes precedence (it only exists when the GPU consume didn't run).
    {
        std::lock_guard<std::mutex> lock(g_cpuPendingMutex);
        auto it = g_cpuPending.find(key);
        if (it != g_cpuPending.end()) {
            CpuPending p = std::move(it->second);
            g_cpuPending.erase(it);
            size_t n = p.main.size() < dstSize ? p.main.size() : dstSize;
            std::memcpy(dst, p.main.data(), n);
            if (scaledDst && scaledSize > 0 && !p.scaled.empty()) {
                size_t sn = p.scaled.size() < scaledSize ? p.scaled.size() : scaledSize;
                std::memcpy(scaledDst, p.scaled.data(), sn);
            }
            return true;
        }
    }
    if (linuxgpu::Initialized()) return linuxgpu::Finish(key, dst, dstSize, scaledDst, scaledSize, err);
    err = "no pending consume for key";
    return false;
}

void ReadbackReleaseKey(const std::string& key) {
    {
        std::lock_guard<std::mutex> lock(g_cpuPendingMutex);
        g_cpuPending.erase(key);
    }
    linuxgpu::ReleaseKey(key);
    linuxgpu::ReleaseKey("sp#" + key);    // the single-phase cache for this pool key
    linuxgpu::ReleaseKey(key + "#once");  // the readbackOnce cache for this pool key
}

// Single-dispatch readback (Linux): consume + fire onGpuDone (the caller's early texture release) +
// finish. Kept for parity with Windows readbackOnce; FreeShow prefers the two-phase pair when present.
bool ReadbackOnce(const std::vector<DmabufPlane>& planes, uint64_t modifier, uint32_t width, uint32_t height, int format, const std::string& key, uint32_t dstW, uint32_t dstH,
                  uint8_t* dst, size_t dstSize, uint8_t* scaledDst, size_t scaledSize,
                  const std::function<void()>& onGpuDone, std::string& err) {
    const std::string k = key.empty() ? ("once#anon" + std::to_string(g_anonKeySeq.fetch_add(1))) : (key + "#once");
    if (!ReadbackConsume(planes, modifier, width, height, format, k, dstW, dstH, err)) return false;
    onGpuDone();  // the dmabuf is consumed — the caller may release the Electron texture now
    bool ok = ReadbackFinish(k, dst, dstSize, scaledDst, scaledSize, err);
    if (key.empty()) linuxgpu::ReleaseKey(k);
    return ok;
}

}  // namespace osrcap

namespace osrcap {
// Per-consumer targets are not produced by the GLES path yet: the caller downscales/converts on the CPU.
bool TargetsSupported() { return false; }
}  // namespace osrcap
