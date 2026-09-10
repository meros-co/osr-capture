// Native BGRA -> UYVY / UYVA colour conversion (BT.601 full range, integer/fixed-point).
//
// NDI's wire format is UYVY (4:2:2). Handing the SDK BGRA makes it do this conversion internally, which is
// pathologically slow at 4K. Doing it here in native code is an order of magnitude faster than a JS loop, so
// callers can pre-convert and send UYVY/UYVA directly. The raw functions are also called by the macOS/Linux
// readback backends (see convert.h) so those platforms return UYVY/UYVA too, not just Windows.
//
// convertBgraToUyvy(bgra: Buffer, width, height) => Buffer  // width*2*height (opaque)
// convertBgraToUyva(bgra: Buffer, width, height) => Buffer  // width*2*height UYVY + width*height alpha
#include "convert.h"

#include <napi.h>

#include <cstring>

namespace osrcap {

namespace {

inline uint8_t clamp8(int v) {
    return v < 0 ? 0 : (v > 255 ? 255 : static_cast<uint8_t>(v));
}

// coefficients scaled by 256; full-range BT.601
inline uint8_t lumaY(int r, int g, int b) {
    return static_cast<uint8_t>((77 * r + 150 * g + 29 * b) >> 8);
}
inline uint8_t chromaU(int r, int g, int b) {
    return clamp8((((-43 * r - 85 * g + 128 * b) >> 8) + 128));
}
inline uint8_t chromaV(int r, int g, int b) {
    return clamp8((((128 * r - 107 * g - 21 * b) >> 8) + 128));
}

}  // namespace

void ConvertBgraToUyvyRaw(const uint8_t* bgra, uint32_t width, uint32_t height, std::vector<uint8_t>& out) {
    out.resize(static_cast<size_t>(width) * 2 * height);
    uint8_t* dst = out.data();
    for (uint32_t y = 0; y < height; ++y) {
        const uint8_t* s = bgra + static_cast<size_t>(y) * width * 4;
        uint8_t* d = dst + static_cast<size_t>(y) * width * 2;
        for (uint32_t x = 0; x < width; x += 2) {
            const int b0 = s[0], g0 = s[1], r0 = s[2];
            const int b1 = s[4], g1 = s[5], r1 = s[6];
            d[0] = chromaU(r0, g0, b0);  // U
            d[1] = lumaY(r0, g0, b0);    // Y0
            d[2] = chromaV(r0, g0, b0);  // V
            d[3] = lumaY(r1, g1, b1);    // Y1
            s += 8;
            d += 4;
        }
    }
}

void ConvertBgraToUyvaRaw(const uint8_t* bgra, uint32_t width, uint32_t height, std::vector<uint8_t>& out) {
    const size_t uyvySize = static_cast<size_t>(width) * 2 * height;
    out.resize(uyvySize + static_cast<size_t>(width) * height);
    uint8_t* dst = out.data();
    uint8_t* alpha = dst + uyvySize;
    for (uint32_t y = 0; y < height; ++y) {
        const uint8_t* s = bgra + static_cast<size_t>(y) * width * 4;
        uint8_t* d = dst + static_cast<size_t>(y) * width * 2;
        uint8_t* a = alpha + static_cast<size_t>(y) * width;
        for (uint32_t x = 0; x < width; x += 2) {
            const int b0 = s[0], g0 = s[1], r0 = s[2], a0 = s[3];
            const int b1 = s[4], g1 = s[5], r1 = s[6], a1 = s[7];
            d[0] = chromaU(r0, g0, b0);  // U
            d[1] = lumaY(r0, g0, b0);    // Y0
            d[2] = chromaV(r0, g0, b0);  // V
            d[3] = lumaY(r1, g1, b1);    // Y1
            a[0] = static_cast<uint8_t>(a0);
            a[1] = static_cast<uint8_t>(a1);
            s += 8;
            d += 4;
            a += 2;
        }
    }
}

bool ConvertBgraInPlace(std::vector<uint8_t>& buf, uint32_t width, uint32_t height, int format) {
    // format 3 = RGBA: swap the R and B channels in place (for WebRTC's ImageData, which is RGBA). Same size.
    if (format == 3) {
        uint8_t* p = buf.data();
        const size_t n = static_cast<size_t>(width) * height;
        for (size_t i = 0; i < n; ++i) std::swap(p[i * 4], p[i * 4 + 2]);
        return true;
    }
    // format 4 = planar I420 (Y, then U, then V at half resolution), BT.601 limited range — the same
    // coefficients as the GPU kernels, so a CPU-fallback frame matches a GPU one.
    if (format == 4) {
        const uint32_t cw = width / 2, ch = height / 2;
        std::vector<uint8_t> out(static_cast<size_t>(width) * height + 2u * static_cast<size_t>(cw) * ch);
        const uint8_t* src = buf.data();
        uint8_t* yp = out.data();
        uint8_t* up = yp + static_cast<size_t>(width) * height;
        uint8_t* vp = up + static_cast<size_t>(cw) * ch;
        for (uint32_t y = 0; y < height; ++y) {
            const uint8_t* row = src + static_cast<size_t>(y) * width * 4;
            uint8_t* yrow = yp + static_cast<size_t>(y) * width;
            for (uint32_t x = 0; x < width; ++x) {
                const int b = row[x * 4], g = row[x * 4 + 1], r = row[x * 4 + 2];
                int luma = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
                yrow[x] = static_cast<uint8_t>(luma < 16 ? 16 : (luma > 235 ? 235 : luma));
            }
        }
        for (uint32_t cy = 0; cy < ch; ++cy) {
            for (uint32_t cx = 0; cx < cw; ++cx) {
                int rs = 0, gs = 0, bs = 0;
                for (uint32_t dy = 0; dy < 2; ++dy) {
                    const uint8_t* row = src + (static_cast<size_t>(cy) * 2 + dy) * width * 4;
                    for (uint32_t dx = 0; dx < 2; ++dx) {
                        const uint32_t x = cx * 2 + dx;
                        bs += row[x * 4];
                        gs += row[x * 4 + 1];
                        rs += row[x * 4 + 2];
                    }
                }
                const int r = (rs + 2) / 4, g = (gs + 2) / 4, b = (bs + 2) / 4;
                int u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
                int v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
                up[static_cast<size_t>(cy) * cw + cx] = static_cast<uint8_t>(u < 16 ? 16 : (u > 240 ? 240 : u));
                vp[static_cast<size_t>(cy) * cw + cx] = static_cast<uint8_t>(v < 16 ? 16 : (v > 240 ? 240 : v));
            }
        }
        buf.swap(out);
        return true;
    }
    if (format != 1 && format != 2) return false;
    std::vector<uint8_t> converted;
    if (format == 2) {
        ConvertBgraToUyvaRaw(buf.data(), width, height, converted);
    } else {
        ConvertBgraToUyvyRaw(buf.data(), width, height, converted);
    }
    buf.swap(converted);
    return true;
}

Napi::Value ConvertBgraToUyvy(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Buffer<uint8_t> src = info[0].As<Napi::Buffer<uint8_t>>();
    uint32_t width = info[1].As<Napi::Number>().Uint32Value();
    uint32_t height = info[2].As<Napi::Number>().Uint32Value();

    std::vector<uint8_t> out;
    ConvertBgraToUyvyRaw(src.Data(), width, height, out);
    return Napi::Buffer<uint8_t>::Copy(env, out.data(), out.size());
}

Napi::Value ConvertBgraToUyva(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Buffer<uint8_t> src = info[0].As<Napi::Buffer<uint8_t>>();
    uint32_t width = info[1].As<Napi::Number>().Uint32Value();
    uint32_t height = info[2].As<Napi::Number>().Uint32Value();

    std::vector<uint8_t> out;
    ConvertBgraToUyvaRaw(src.Data(), width, height, out);
    return Napi::Buffer<uint8_t>::Copy(env, out.data(), out.size());
}

// Native box-filter downscale of tightly-packed BGRA (srcW*srcH*4) to dstW*dstH*4. Averages each destination
// pixel's source block — reads the whole source once (memory-bound, ~10-30ms at 4K), which is an order of
// magnitude cheaper than a 4K nativeImage.resize on the JS main thread (~200ms). Used so server/stage/preview
// consumers get a small image without pinning the main thread.
void DownscaleBgraRaw(const uint8_t* s, uint32_t srcW, uint32_t srcH, uint32_t dstW, uint32_t dstH, std::vector<uint8_t>& out) {
    if (dstW < 1) dstW = 1;
    if (dstH < 1) dstH = 1;
    out.resize(static_cast<size_t>(dstW) * dstH * 4);
    uint8_t* d = out.data();

    for (uint32_t dy = 0; dy < dstH; ++dy) {
        uint32_t sy0 = static_cast<uint32_t>(static_cast<uint64_t>(dy) * srcH / dstH);
        uint32_t sy1 = static_cast<uint32_t>(static_cast<uint64_t>(dy + 1) * srcH / dstH);
        if (sy1 <= sy0) sy1 = sy0 + 1;
        if (sy1 > srcH) sy1 = srcH;
        for (uint32_t dx = 0; dx < dstW; ++dx) {
            uint32_t sx0 = static_cast<uint32_t>(static_cast<uint64_t>(dx) * srcW / dstW);
            uint32_t sx1 = static_cast<uint32_t>(static_cast<uint64_t>(dx + 1) * srcW / dstW);
            if (sx1 <= sx0) sx1 = sx0 + 1;
            if (sx1 > srcW) sx1 = srcW;
            uint32_t b = 0, g = 0, r = 0, a = 0, cnt = 0;
            for (uint32_t sy = sy0; sy < sy1; ++sy) {
                const uint8_t* row = s + (static_cast<size_t>(sy) * srcW + sx0) * 4;
                for (uint32_t sx = sx0; sx < sx1; ++sx) {
                    b += row[0];
                    g += row[1];
                    r += row[2];
                    a += row[3];
                    row += 4;
                    ++cnt;
                }
            }
            uint8_t* o = d + (static_cast<size_t>(dy) * dstW + dx) * 4;
            o[0] = static_cast<uint8_t>(b / cnt);
            o[1] = static_cast<uint8_t>(g / cnt);
            o[2] = static_cast<uint8_t>(r / cnt);
            o[3] = static_cast<uint8_t>(a / cnt);
        }
    }
}

Napi::Value DownscaleBgra(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Buffer<uint8_t> src = info[0].As<Napi::Buffer<uint8_t>>();
    uint32_t srcW = info[1].As<Napi::Number>().Uint32Value();
    uint32_t srcH = info[2].As<Napi::Number>().Uint32Value();
    uint32_t dstW = info[3].As<Napi::Number>().Uint32Value();
    uint32_t dstH = info[4].As<Napi::Number>().Uint32Value();

    std::vector<uint8_t> out;
    DownscaleBgraRaw(src.Data(), srcW, srcH, dstW, dstH, out);
    return Napi::Buffer<uint8_t>::Copy(env, out.data(), out.size());
}

// planar I420 for the RTMP encoder, for the rare frame no GPU pass produced one for
Napi::Value ConvertBgraToI420(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Buffer<uint8_t> src = info[0].As<Napi::Buffer<uint8_t>>();
    uint32_t width = info[1].As<Napi::Number>().Uint32Value();
    uint32_t height = info[2].As<Napi::Number>().Uint32Value();

    std::vector<uint8_t> buf(src.Data(), src.Data() + src.ByteLength());
    if (!ConvertBgraInPlace(buf, width, height, 4)) {
        Napi::Error::New(env, "convertBgraToI420 failed").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    return Napi::Buffer<uint8_t>::Copy(env, buf.data(), buf.size());
}

void RegisterConvert(Napi::Env env, Napi::Object exports) {
    exports.Set("convertBgraToI420", Napi::Function::New(env, ConvertBgraToI420));
    exports.Set("convertBgraToUyvy", Napi::Function::New(env, ConvertBgraToUyvy));
    exports.Set("convertBgraToUyva", Napi::Function::New(env, ConvertBgraToUyva));
    exports.Set("downscaleBgra", Napi::Function::New(env, DownscaleBgra));
}

}  // namespace osrcap
