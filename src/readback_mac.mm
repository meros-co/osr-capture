// macOS backend: Electron passes the shared texture as an IOSurface* (uintptr_t).
//
// GPU PATH (default whenever a Metal device exists): the IOSurface is wrapped ZERO-COPY as an MTLTexture
// (newTextureWithDescriptor:iosurface:plane:) and a Metal compute kernel does the BGRA->UYVY/UYVA convert,
// the BGRA->RGBA swizzle and the box-filter downscale on the GPU — the same work the Windows backend does in
// HLSL, with byte-identical BT.601 full-range integer math (see convert.cc / kConvertHLSL). Only the reduced
// result crosses to the CPU. On Apple Silicon the output MTLBuffer is StorageModeShared, i.e. the same
// physical memory the GPU wrote, so the "readback" is a plain cached memcpy with no bus transfer at all.
//
// This replaces the previous CPU path (IOSurfaceLock + full-frame row memcpy + ConvertBgraInPlace), which at
// 4K60 cost ~12ms/frame of pure CPU — 72-90% of the 16.7ms frame budget — and left no headroom for a second
// output. That path REMAINS as the fallback for machines with no Metal device, for a non-BGRA IOSurface, and
// for widths that are not a multiple of 4 (the packed 4-pixels-per-thread kernel writes 4-byte-aligned words,
// exactly like the Windows one). Hardware acceleration disabled is handled a level up: FreeShow never takes
// the shared-texture path at all in that case, so nothing here has to know about it.
//
// We do NOT own the IOSurface (Electron holds it until texture.release()), so we only read from it.
#include "osr_readback.h"

#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "convert.h"

namespace osrcap {

namespace {

// ---------------------------------------------------------------------------------------------------------
// Metal kernels. Ported 1:1 from the Windows HLSL (readback_win.cc kConvertHLSL / kDownscaleHLSL /
// kSwizzleHLSL) so all three platforms produce the same bytes for the same frame. toByte() recovers the exact
// source byte: a BGRA8Unorm texel reads back as byte/255.0, so byte/255.0*255.0 + 0.5 truncates to `byte`.
const char* kMetalSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

inline int toByte(float v) { return int(v * 255.0f + 0.5f); }

struct ConvParams { uint width; uint height; uint uyvySize; uint writeAlpha; };
struct DsParams   { uint srcW; uint srcH; uint dstW; uint dstH; };
struct SwParams   { uint width; uint height; };

// BGRA -> UYVY (+ optional full-res alpha plane). Each thread handles 4 horizontal pixels so every store to
// the output buffer is a 4-byte-aligned word (2 UYVY words + 1 packed-alpha word), matching the Windows
// RWByteAddressBuffer stores. Callers guarantee width % 4 == 0, so px+3 is always inside the row.
kernel void convertUyvy(texture2d<float, access::read> src [[texture(0)]],
                        device uint* dst [[buffer(0)]],
                        constant ConvParams& p [[buffer(1)]],
                        uint2 tid [[thread_position_in_grid]]) {
    uint px = tid.x * 4;
    uint y = tid.y;
    if (px >= p.width || y >= p.height) return;

    float4 c0 = src.read(uint2(px + 0, y));
    float4 c1 = src.read(uint2(px + 1, y));
    float4 c2 = src.read(uint2(px + 2, y));
    float4 c3 = src.read(uint2(px + 3, y));

    int r0 = toByte(c0.r), g0 = toByte(c0.g), b0 = toByte(c0.b);
    int r1 = toByte(c1.r), g1 = toByte(c1.g), b1 = toByte(c1.b);
    int r2 = toByte(c2.r), g2 = toByte(c2.g), b2 = toByte(c2.b);
    int r3 = toByte(c3.r), g3 = toByte(c3.g), b3 = toByte(c3.b);

    int y0 = (77 * r0 + 150 * g0 + 29 * b0) >> 8;
    int y1 = (77 * r1 + 150 * g1 + 29 * b1) >> 8;
    int y2 = (77 * r2 + 150 * g2 + 29 * b2) >> 8;
    int y3 = (77 * r3 + 150 * g3 + 29 * b3) >> 8;
    int u01 = clamp(((-43 * r0 - 85 * g0 + 128 * b0) >> 8) + 128, 0, 255);
    int v01 = clamp(((128 * r0 - 107 * g0 - 21 * b0) >> 8) + 128, 0, 255);
    int u23 = clamp(((-43 * r2 - 85 * g2 + 128 * b2) >> 8) + 128, 0, 255);
    int v23 = clamp(((128 * r2 - 107 * g2 - 21 * b2) >> 8) + 128, 0, 255);

    uint word0 = uint(u01) | (uint(y0) << 8) | (uint(v01) << 16) | (uint(y1) << 24);
    uint word1 = uint(u23) | (uint(y2) << 8) | (uint(v23) << 16) | (uint(y3) << 24);
    uint base = (y * p.width * 2 + tid.x * 8) >> 2;  // byte offset -> uint index
    dst[base] = word0;
    dst[base + 1] = word1;

    if (p.writeAlpha != 0) {
        uint aword = uint(toByte(c0.a)) | (uint(toByte(c1.a)) << 8) | (uint(toByte(c2.a)) << 16) | (uint(toByte(c3.a)) << 24);
        dst[(p.uyvySize + y * p.width + tid.x * 4) >> 2] = aword;
    }
}

// Box-filter downscale straight from the shared texture to a small tightly-packed BGRA buffer (server/stage
// previews). Same integer block bounds as the CPU downscaleBgra(), the Windows DSMain and the Linux GLES3
// pass. NOTE: like both of those GPU paths it accumulates in FLOAT and rounds (toByte), whereas the CPU
// downscaleBgra() accumulates in integers and truncates — so GPU and CPU previews can differ by ±1 per
// channel. That is the pre-existing convention on all three GPU backends, kept here deliberately so the
// three GPU paths agree with each other; the convert kernels above ARE byte-identical to the CPU converter.
kernel void downscaleBgra(texture2d<float, access::read> src [[texture(0)]],
                          device uint* dst [[buffer(0)]],
                          constant DsParams& p [[buffer(1)]],
                          uint2 tid [[thread_position_in_grid]]) {
    if (tid.x >= p.dstW || tid.y >= p.dstH) return;
    uint sx0 = tid.x * p.srcW / p.dstW;
    uint sx1 = (tid.x + 1) * p.srcW / p.dstW; if (sx1 <= sx0) sx1 = sx0 + 1;
    uint sy0 = tid.y * p.srcH / p.dstH;
    uint sy1 = (tid.y + 1) * p.srcH / p.dstH; if (sy1 <= sy0) sy1 = sy0 + 1;
    if (sx1 > p.srcW) sx1 = p.srcW;
    if (sy1 > p.srcH) sy1 = p.srcH;

    float4 acc = float4(0.0f);
    uint cnt = 0;
    for (uint y = sy0; y < sy1; ++y) {
        for (uint x = sx0; x < sx1; ++x) { acc += src.read(uint2(x, y)); ++cnt; }
    }
    acc /= float(cnt);
    uint word = uint(toByte(acc.b)) | (uint(toByte(acc.g)) << 8) | (uint(toByte(acc.r)) << 16) | (uint(toByte(acc.a)) << 24);
    dst[tid.y * p.dstW + tid.x] = word;  // B,G,R,A byte order
}

// BGRA -> RGBA channel swap (WebRTC's ImageData is RGBA). Same size as the source; the swizzle is free
// relative to the copy it rides along with.
kernel void swizzleRgba(texture2d<float, access::read> src [[texture(0)]],
                        device uint* dst [[buffer(0)]],
                        constant SwParams& p [[buffer(1)]],
                        uint2 tid [[thread_position_in_grid]]) {
    if (tid.x >= p.width || tid.y >= p.height) return;
    float4 c = src.read(uint2(tid.x, tid.y));
    uint word = uint(toByte(c.r)) | (uint(toByte(c.g)) << 8) | (uint(toByte(c.b)) << 16) | (uint(toByte(c.a)) << 24);
    dst[tid.y * p.width + tid.x] = word;  // R,G,B,A byte order
}
)MSL";

struct ConvParams {
    uint32_t width;
    uint32_t height;
    uint32_t uyvySize;
    uint32_t writeAlpha;
};
struct DsParams {
    uint32_t srcW;
    uint32_t srcH;
    uint32_t dstW;
    uint32_t dstH;
};
struct SwParams {
    uint32_t width;
    uint32_t height;
};

// ---------------------------------------------------------------------------------------------------------
// Device / pipeline singleton. Built once on first use; if anything fails we latch "no GPU" and every caller
// degrades to the CPU converter rather than erroring — same never-crash-always-degrade rule as the Windows
// §5 ladder.
struct MetalCtx {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> psoConvert = nil;
    id<MTLComputePipelineState> psoDownscale = nil;
    id<MTLComputePipelineState> psoSwizzle = nil;
    MTLStorageMode ioSurfaceStorage = MTLStorageModeManaged;
    bool ok = false;
};

const char* g_backend = "none";

MetalCtx& Ctx() {
    static MetalCtx ctx;
    static std::once_flag once;
    std::call_once(once, [] {
        @autoreleasepool {
            ctx.device = MTLCreateSystemDefaultDevice();
            if (!ctx.device) {
                g_backend = "iosurface-cpu";
                fprintf(stderr, "[READBACK] no Metal device -> CPU IOSurface path\n");
                fflush(stderr);
                return;
            }
            ctx.queue = [ctx.device newCommandQueue];
            NSError* err = nil;
            id<MTLLibrary> lib = [ctx.device newLibraryWithSource:[NSString stringWithUTF8String:kMetalSource]
                                                          options:nil
                                                            error:&err];
            if (!lib) {
                g_backend = "iosurface-cpu";
                fprintf(stderr, "[READBACK] Metal library compile failed (%s) -> CPU IOSurface path\n",
                        err ? [[err localizedDescription] UTF8String] : "unknown");
                fflush(stderr);
                return;
            }
            auto pso = [&](const char* name) -> id<MTLComputePipelineState> {
                id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
                if (!fn) return nil;
                NSError* e = nil;
                return [ctx.device newComputePipelineStateWithFunction:fn error:&e];
            };
            ctx.psoConvert = pso("convertUyvy");
            ctx.psoDownscale = pso("downscaleBgra");
            ctx.psoSwizzle = pso("swizzleRgba");
            if (!ctx.queue || !ctx.psoConvert || !ctx.psoDownscale || !ctx.psoSwizzle) {
                g_backend = "iosurface-cpu";
                fprintf(stderr, "[READBACK] Metal pipeline init failed -> CPU IOSurface path\n");
                fflush(stderr);
                return;
            }
            // An IOSurface-backed texture must use Shared storage on unified-memory (Apple Silicon) devices
            // and Managed on discrete/Intel ones.
            ctx.ioSurfaceStorage = ctx.device.hasUnifiedMemory ? MTLStorageModeShared : MTLStorageModeManaged;
            ctx.ok = true;
            g_backend = "metal";
            fprintf(stderr, "[READBACK] backend=metal (%s, %s memory)\n", [[ctx.device name] UTF8String],
                    ctx.device.hasUnifiedMemory ? "unified" : "discrete");
            fflush(stderr);
        }
    });
    return ctx;
}

// The GPU kernels write 4-byte words, so the packed 4-pixels-per-thread convert needs width % 4 == 0 (as on
// Windows). Every real output resolution satisfies this; anything else falls back to the CPU converter.
bool GpuUsable(IOSurfaceRef surface, uint32_t width, uint32_t height) {
    if (!surface || width == 0 || height == 0) return false;
    if (width % 4 != 0) return false;
    if (IOSurfaceGetPixelFormat(surface) != 'BGRA') return false;
    if (IOSurfaceGetWidth(surface) < width || IOSurfaceGetHeight(surface) < height) return false;
    return Ctx().ok;
}

size_t OutBytes(uint32_t w, uint32_t h, int format) {
    size_t px = static_cast<size_t>(w) * h;
    if (format == 1) return px * 2;      // UYVY
    if (format == 2) return px * 3;      // UYVY + alpha plane
    return px * 4;                       // BGRA (0) / RGBA (3)
}

id<MTLTexture> WrapSurface(IOSurfaceRef surface, uint32_t width, uint32_t height) {
    MetalCtx& ctx = Ctx();
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                    width:width
                                                                                   height:height
                                                                                mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = ctx.ioSurfaceStorage;
    return [ctx.device newTextureWithDescriptor:desc iosurface:surface plane:0];
}

// Reused GPU output buffers, keyed the way the caller keys its N-API buffer pool (FreeShow passes the output
// id, plus a slot suffix for pipelined captures). Avoids a per-frame MTLBuffer allocation. Guarded by g_mutex.
struct KeyBufs {
    id<MTLBuffer> main = nil;
    size_t mainSize = 0;
    id<MTLBuffer> scaled = nil;
    size_t scaledSize = 0;
    bool pending = false;  // a consume has run and its finish has not
    // Per-frame CPU fallback (a frame whose surface the GPU path can't take — see GpuUsable). Finish reads
    // these instead of the MTLBuffers, so the two-phase contract holds either way and a single odd frame
    // never fails the capture.
    bool cpuFallback = false;
    std::vector<uint8_t> cpuMain;
    std::vector<uint8_t> cpuScaled;
};

std::mutex g_mutex;
std::map<std::string, KeyBufs> g_keys;

id<MTLBuffer> EnsureBuffer(id<MTLBuffer> existing, size_t& existingSize, size_t needed) {
    if (existing && existingSize >= needed) return existing;
    existingSize = needed;
    // Shared storage: on Apple Silicon this is the very memory the GPU writes, so the copy-out is a cached
    // memcpy with no bus transfer. On discrete GPUs it is host memory the GPU writes over PCIe — still far
    // cheaper than reading back a full 4K BGRA frame and converting on the CPU.
    return [Ctx().device newBufferWithLength:needed options:MTLResourceStorageModeShared];
}

// Run the GPU convert (+ optional downscale) for one frame and BLOCK until the GPU has finished reading the
// IOSurface. On return the caller may release the Electron shared texture; the results live in the returned
// MTLBuffers. This is the whole "consume" phase — see the two-phase contract in osr_readback.h.
bool RunGpu(IOSurfaceRef surface, uint32_t width, uint32_t height, int format, uint32_t dstW, uint32_t dstH,
            id<MTLBuffer> mainBuf, id<MTLBuffer> scaledBuf, std::string& err) {
    MetalCtx& ctx = Ctx();
    @autoreleasepool {
        id<MTLTexture> tex = WrapSurface(surface, width, height);
        if (!tex) {
            err = "failed to wrap IOSurface as MTLTexture";
            return false;
        }

        id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
        if (!cb) {
            err = "failed to create Metal command buffer";
            return false;
        }

        if (format == 1 || format == 2) {
            ConvParams p{width, height, static_cast<uint32_t>(width) * 2 * height, format == 2 ? 1u : 0u};
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ctx.psoConvert];
            [enc setTexture:tex atIndex:0];
            [enc setBuffer:mainBuf offset:0 atIndex:0];
            [enc setBytes:&p length:sizeof(p) atIndex:1];
            [enc dispatchThreads:MTLSizeMake((width + 3) / 4, height, 1)
                threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc endEncoding];
        } else if (format == 3) {
            SwParams p{width, height};
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ctx.psoSwizzle];
            [enc setTexture:tex atIndex:0];
            [enc setBuffer:mainBuf offset:0 atIndex:0];
            [enc setBytes:&p length:sizeof(p) atIndex:1];
            [enc dispatchThreads:MTLSizeMake(width, height, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc endEncoding];
        } else {
            // format 0: tightly-packed BGRA. A blit strips the IOSurface's row padding on the GPU, so the CPU
            // never walks the frame row by row.
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            [blit copyFromTexture:tex
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(width, height, 1)
                         toBuffer:mainBuf
                destinationOffset:0
           destinationBytesPerRow:static_cast<NSUInteger>(width) * 4
         destinationBytesPerImage:static_cast<NSUInteger>(width) * 4 * height];
            [blit endEncoding];
        }

        if (scaledBuf && dstW > 0 && dstH > 0) {
            DsParams p{width, height, dstW, dstH};
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ctx.psoDownscale];
            [enc setTexture:tex atIndex:0];
            [enc setBuffer:scaledBuf offset:0 atIndex:0];
            [enc setBytes:&p length:sizeof(p) atIndex:1];
            [enc dispatchThreads:MTLSizeMake(dstW, dstH, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc endEncoding];
        }

        [cb commit];
        [cb waitUntilCompleted];  // GPU is done READING the IOSurface -> caller may release the texture
        if (cb.error) {
            err = std::string("Metal command buffer failed: ") + [[cb.error localizedDescription] UTF8String];
            return false;
        }
    }
    return true;
}

// The original CPU path, kept as the fallback: lock the IOSurface, copy rows (stripping padding), convert.
bool ReadbackCpu(IOSurfaceRef surface, uint32_t width, uint32_t height, int format, std::vector<uint8_t>& out, std::string& err) {
    IOReturn lockResult = IOSurfaceLock(surface, kIOSurfaceLockReadOnly, nullptr);
    if (lockResult != kIOReturnSuccess) {
        err = "IOSurfaceLock failed";
        return false;
    }
    uint8_t* base = static_cast<uint8_t*>(IOSurfaceGetBaseAddress(surface));
    const size_t rowPitch = IOSurfaceGetBytesPerRow(surface);
    if (!base) {
        IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, nullptr);
        err = "IOSurfaceGetBaseAddress returned null";
        return false;
    }
    const size_t rowBytes = static_cast<size_t>(width) * 4;
    out.resize(rowBytes * height);
    for (uint32_t y = 0; y < height; ++y) {
        std::memcpy(out.data() + static_cast<size_t>(y) * rowBytes, base + static_cast<size_t>(y) * rowPitch, rowBytes);
    }
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, nullptr);
    ConvertBgraInPlace(out, width, height, format);
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------
// Single-shot readback into a caller vector (the `readback` export / main-thread fallback path).
bool ReadbackHandle(uintptr_t handle, uint32_t width, uint32_t height, int format, std::vector<uint8_t>& out, std::string& err) {
    IOSurfaceRef surface = reinterpret_cast<IOSurfaceRef>(handle);
    if (!surface) {
        err = "null IOSurface handle";
        return false;
    }
    if (!GpuUsable(surface, width, height)) return ReadbackCpu(surface, width, height, format, out, err);

    const size_t outSize = OutBytes(width, height, format);
    @autoreleasepool {
        size_t sz = 0;
        id<MTLBuffer> buf = EnsureBuffer(nil, sz, outSize);
        if (!buf) {
            err = "failed to allocate Metal output buffer";
            return false;
        }
        if (!RunGpu(surface, width, height, format, 0, 0, buf, nil, err)) return false;
        out.resize(outSize);
        std::memcpy(out.data(), [buf contents], outSize);
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------
// Two-phase readback. Consume converts on the GPU and returns once the GPU has read the shared texture, so
// FreeShow can hand the Electron texture straight back to the compositor's frame pool; Finish copies the
// (already small) result out. Identical contract to the Windows implementation, so the JS side needs no
// platform branch — its `typeof osr.readbackConsume === "function"` probe simply now succeeds on macOS.
bool ReadbackConsume(uintptr_t handle, uint32_t width, uint32_t height, int format, const std::string& key, uint32_t dstW, uint32_t dstH, std::string& err) {
    IOSurfaceRef surface = reinterpret_cast<IOSurfaceRef>(handle);
    if (!surface) {
        err = "null IOSurface handle";
        return false;
    }
    const bool wantScaled = dstW > 0 && dstH > 0;

    // Frame the GPU path can't take (non-BGRA surface, width not a multiple of 4): convert on the CPU now and
    // stash the bytes for Finish. Rare-to-never in practice, but it keeps the contract total.
    if (!GpuUsable(surface, width, height)) {
        std::vector<uint8_t> bgra;
        if (!ReadbackCpu(surface, width, height, 0, bgra, err)) return false;  // format 0 = leave as BGRA
        std::lock_guard<std::mutex> lock(g_mutex);
        KeyBufs& k = g_keys[key];
        if (wantScaled) DownscaleBgraRaw(bgra.data(), width, height, dstW, dstH, k.cpuScaled);
        ConvertBgraInPlace(bgra, width, height, format);
        k.cpuMain.swap(bgra);
        k.cpuFallback = true;
        k.pending = true;
        return true;
    }

    const size_t mainSize = OutBytes(width, height, format);
    const size_t scaledSize = wantScaled ? static_cast<size_t>(dstW) * dstH * 4 : 0;

    id<MTLBuffer> mainBuf = nil;
    id<MTLBuffer> scaledBuf = nil;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        KeyBufs& k = g_keys[key];
        k.main = EnsureBuffer(k.main, k.mainSize, mainSize);
        mainBuf = k.main;
        if (wantScaled) {
            k.scaled = EnsureBuffer(k.scaled, k.scaledSize, scaledSize);
            scaledBuf = k.scaled;
        }
        if (!mainBuf || (wantScaled && !scaledBuf)) {
            err = "failed to allocate Metal output buffer";
            return false;
        }
    }

    if (!RunGpu(surface, width, height, format, dstW, dstH, mainBuf, scaledBuf, err)) return false;

    std::lock_guard<std::mutex> lock(g_mutex);
    KeyBufs& k = g_keys[key];
    k.cpuFallback = false;
    k.pending = true;
    return true;
}

bool ReadbackFinish(const std::string& key, uint8_t* dst, size_t dstSize, uint8_t* scaledDst, size_t scaledSize, std::string& err) {
    id<MTLBuffer> mainBuf = nil;
    id<MTLBuffer> scaledBuf = nil;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_keys.find(key);
        if (it == g_keys.end() || !it->second.pending) {
            err = "no pending consume for key";
            return false;
        }
        KeyBufs& k = it->second;
        k.pending = false;

        if (k.cpuFallback) {
            if (k.cpuMain.size() < dstSize || (scaledDst && k.cpuScaled.size() < scaledSize)) {
                err = "pending CPU buffer smaller than requested copy";
                return false;
            }
            if (dst && dstSize) std::memcpy(dst, k.cpuMain.data(), dstSize);
            if (scaledDst && scaledSize) std::memcpy(scaledDst, k.cpuScaled.data(), scaledSize);
            return true;
        }

        if (k.mainSize < dstSize || (scaledDst && k.scaledSize < scaledSize)) {
            err = "pending buffer smaller than requested copy";
            return false;
        }
        mainBuf = k.main;
        scaledBuf = k.scaled;
    }
    // Copy outside the lock: on unified memory this is a cached memcpy of the already-reduced result, but it
    // must not serialize other outputs' consumes behind it.
    if (dst && dstSize) std::memcpy(dst, [mainBuf contents], dstSize);
    if (scaledDst && scaledSize) std::memcpy(scaledDst, [scaledBuf contents], scaledSize);
    return true;
}

void ReadbackReleaseKey(const std::string& key) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_keys.erase(key);
}

// ---------------------------------------------------------------------------------------------------------
// Single-dispatch readback: the whole convert + wait + copy-out in one async op, firing `onGpuDone` the
// instant the GPU is finished with the shared texture (before the copy-out) so the caller's early-release
// behaviour matches the two-phase path without a second N-API hop.
bool ReadbackOnce(uintptr_t handle, uint32_t width, uint32_t height, int format, uint32_t dstW, uint32_t dstH,
                  uint8_t* dst, size_t dstSize, uint8_t* scaledDst, size_t scaledSize,
                  const std::function<void()>& onGpuDone, std::string& err) {
    IOSurfaceRef surface = reinterpret_cast<IOSurfaceRef>(handle);
    if (!surface) {
        err = "null IOSurface handle";
        return false;
    }
    // Same total-contract fallback as ReadbackConsume: a frame the GPU path can't take is converted on the
    // CPU here rather than failing the capture.
    if (!GpuUsable(surface, width, height)) {
        std::vector<uint8_t> bgra;
        if (!ReadbackCpu(surface, width, height, 0, bgra, err)) return false;
        std::vector<uint8_t> small;
        if (scaledDst && scaledSize && dstW > 0 && dstH > 0) DownscaleBgraRaw(bgra.data(), width, height, dstW, dstH, small);
        ConvertBgraInPlace(bgra, width, height, format);
        if (onGpuDone) onGpuDone();
        if (dst && dstSize && bgra.size() >= dstSize) std::memcpy(dst, bgra.data(), dstSize);
        if (scaledDst && scaledSize && small.size() >= scaledSize) std::memcpy(scaledDst, small.data(), scaledSize);
        return true;
    }

    @autoreleasepool {
        size_t ms = 0, ss = 0;
        id<MTLBuffer> mainBuf = EnsureBuffer(nil, ms, OutBytes(width, height, format));
        id<MTLBuffer> scaledBuf = (dstW > 0 && dstH > 0) ? EnsureBuffer(nil, ss, static_cast<size_t>(dstW) * dstH * 4) : nil;
        if (!mainBuf) {
            err = "failed to allocate Metal output buffer";
            return false;
        }
        if (!RunGpu(surface, width, height, format, dstW, dstH, mainBuf, scaledBuf, err)) return false;
        if (onGpuDone) onGpuDone();
        if (dst && dstSize) std::memcpy(dst, [mainBuf contents], dstSize);
        if (scaledDst && scaledSize && scaledBuf) std::memcpy(scaledDst, [scaledBuf contents], scaledSize);
    }
    return true;
}

// One-time Metal init. True when the two-phase/once exports should be installed: with no Metal device they
// stay absent, FreeShow's `typeof readbackConsume === "function"` probe fails, and it keeps to single-phase
// `readback` — which then takes the CPU IOSurface path. Idempotent and thread-safe (Ctx() is call_once).
bool MacGpuReadbackInit() { return Ctx().ok; }

// Telemetry: which path is actually running ("metal" | "iosurface-cpu" | "none" before first use). Surfaces in
// FreeShow's [SEND-STATS] `rb=` field, which previously could never resolve on macOS.
const char* ReadbackBackend() {
    Ctx();
    return g_backend;
}

}  // namespace osrcap

namespace osrcap {
// Per-consumer targets are not produced by the Metal path yet: the caller downscales/converts on the CPU.
bool TargetsSupported() { return false; }
bool ReadbackConsume(uintptr_t handle, uint32_t width, uint32_t height, int format, const std::string& key, uint32_t dstW, uint32_t dstH, const std::vector<TargetSpec>& targets, std::string& err) {
    if (!targets.empty()) { err = "per-consumer targets are not supported by the macOS backend"; return false; }
    return ReadbackConsume(handle, width, height, format, key, dstW, dstH, err);
}
bool ReadbackFinish(const std::string& key, uint8_t* dst, size_t dstSize, uint8_t* scaledDst, size_t scaledSize, const std::vector<TargetDst>& targetDsts, std::string& err) {
    if (!targetDsts.empty()) { err = "per-consumer targets are not supported by the macOS backend"; return false; }
    return ReadbackFinish(key, dst, dstSize, scaledDst, scaledSize, err);
}
}  // namespace osrcap
