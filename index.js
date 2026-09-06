// osr-capture: read back an Electron OSR shared GPU texture to a CPU buffer off the main thread.
// See index.d.ts for the full typed API. Summary:
//   readback(source, width, height, format = 0, poolKey?) => Promise<Buffer>
//     format 0 = BGRA (tightly-packed), 1 = UYVY (opaque), 2 = UYVA (colour + full-res alpha), 3 = RGBA.
//     All platforms convert on the GPU: Windows a D3D11 compute shader, macOS a Metal compute kernel on the
//     IOSurface wrapped zero-copy as an MTLTexture, Linux EGL/GLES3 when that path initialized (see
//     LINUX_GPU_READBACK.md). Each falls back to the CPU converter only when its GPU path is unavailable.
//   readbackConsume(source, w, h, format, poolKey, dstW?, dstH?) / readbackFinish(poolKey, w, h, format, dstW?, dstH?, dst?)
//     dst: an optional caller-owned Buffer (>= frame bytes, untouched until the promise settles) that receives
//     the frame directly, so the caller can hand it to its consumers without copying it again.
//     targets: [{ width, height, format }] per-consumer outputs produced in the SAME pass (box-downscale +
//     convert) when `targetsSupported` (Windows D3D11); finish takes the same list plus optional caller
//     buffers (targetDsts) and resolves { main, scaled?, targets: Buffer[] }.
//     Two-phase readback (Windows always; macOS when a Metal device exists; Linux when the GPU path is
//     active — absent otherwise): release the shared texture right after the GPU consume, then copy out.
//     dstW/dstH also GPU-downscale a small BGRA in the same pass -> readbackFinish resolves { main, scaled }.
//   releasePool(poolKey)                                  // free a poolKey's reused output buffers
//   convertBgraToUyvy / convertBgraToUyva(bgra, w, h)     // CPU fallback converters (sync)
//   downscaleBgra(bgra, srcW, srcH, dstW, dstH)           // CPU box-filter downscale (sync)
module.exports = require("./build/Release/osr_readback.node")
