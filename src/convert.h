// Native BGRA -> UYVY / UYVA colour conversion (BT.601 full range, integer/fixed-point).
// Shared by the N-API exports (convert.cc) and the per-platform readback backends so that macOS and
// Linux — which currently read the shared texture back as BGRA on the CPU — can still hand callers the
// reduced NDI/SDI wire format (UYVY/UYVA) instead of BGRA, skipping the sender SDK's slow internal
// conversion. On Windows the same conversion is done on the GPU (compute shader) during readback.
#pragma once

#include <cstdint>
#include <vector>

namespace osrcap {

// Bytes a converted frame occupies, so a caller can size a destination before converting.
// format: 0/3 = BGRA/RGBA (w*h*4), 1 = UYVY (w*2*h), 2 = UYVA (w*3*h), 4 = I420 (w*h + 2*(w/2)*(h/2)).
size_t ConvertedSize(uint32_t width, uint32_t height, int format);

// Convert straight into caller memory. `dst` must hold ConvertedSize(width, height, format) bytes and
// must not overlap `src` unless format is 3 (which is a channel swap and may be done in place).
// Returns false for a format that is not a conversion, leaving `dst` untouched.
bool ConvertBgraInto(const uint8_t* src, uint32_t width, uint32_t height, int format, uint8_t* dst);

// Box-filter downscale straight into caller memory (dstW*dstH*4 bytes).
void DownscaleBgraInto(const uint8_t* src, uint32_t srcW, uint32_t srcH, uint32_t dstW, uint32_t dstH, uint8_t* dst);

// Convert tightly-packed BGRA (width*height*4) to packed UYVY 4:2:2 (width*2*height).
void ConvertBgraToUyvyRaw(const uint8_t* bgra, uint32_t width, uint32_t height, std::vector<uint8_t>& out);

// Convert to UYVA: a UYVY plane (width*2*height) immediately followed by a full-res alpha plane
// (width*height), i.e. width*3*height bytes total.
void ConvertBgraToUyvaRaw(const uint8_t* bgra, uint32_t width, uint32_t height, std::vector<uint8_t>& out);

// In-place helper for readback backends: `buf` holds BGRA on entry. For format 1 (UYVY) / 2 (UYVA) it is
// replaced with the converted bytes and true is returned; for any other format it is left untouched and
// false is returned (caller keeps the BGRA).
bool ConvertBgraInPlace(std::vector<uint8_t>& buf, uint32_t width, uint32_t height, int format);

// Box-filter downscale of tightly-packed BGRA (srcW*srcH*4) into `out` (dstW*dstH*4). Shared by the
// N-API downscaleBgra export and the Linux CPU-fallback consume (which mirrors the GPU downscale).
void DownscaleBgraRaw(const uint8_t* src, uint32_t srcW, uint32_t srcH, uint32_t dstW, uint32_t dstH, std::vector<uint8_t>& out);

}  // namespace osrcap
