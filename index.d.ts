declare module "osr-capture" {
    interface DmabufPlane {
        fd: number
        stride: number
        offset: number
        size: number
    }

    interface DmabufInfo {
        planes: DmabufPlane[]
        modifier: number | bigint
    }

    /**
     * Pixel format for {@link readback}:
     * - `0` BGRA — tightly-packed `width * 4 * height`.
     * - `1` UYVY — packed 4:2:2 `width * 2 * height` (opaque; e.g. NDI/SDI without alpha).
     * - `2` UYVA — UYVY plane `width * 2 * height` + full-res alpha plane `width * height` (NDI with alpha).
     * - `3` RGBA — tightly-packed `width * 4 * height` (BGRA with R/B swapped; e.g. WebRTC's ImageData).
     *
     * The conversion/swizzle is done on the GPU during readback on every platform. Windows: a D3D11 compute
     * shader. macOS: a Metal compute kernel reading the IOSurface wrapped zero-copy as an MTLTexture, falling
     * back to a CPU convert of the locked surface when there is no Metal device (or the surface is not BGRA /
     * the width is not a multiple of 4). Linux: EGL dmabuf import + GLES3 fragment shaders when the headless
     * EGL path initialized, falling back to a CPU convert of the mmap'd LINEAR dmabuf otherwise. All paths
     * produce identical bytes (BT.601 full range for 1/2, channel swap for 3).
     */
    export type ReadbackFormat = 0 | 1 | 2 | 3

    /**
     * Read back an Electron OSR shared texture into a CPU buffer, off the main thread.
     *
     * @param source Windows/macOS: the 8-byte Buffer from `texture.textureInfo.sharedTextureHandle`.
     *               Linux: `{ planes: texture.textureInfo.planes, modifier: texture.textureInfo.modifier }`.
     * @param width  `texture.textureInfo.codedSize.width`.
     * @param height `texture.textureInfo.codedSize.height`.
     * @param format Output pixel format (default `0` BGRA). See {@link ReadbackFormat}.
     * @param poolKey Optional stable key (e.g. the output id). When given, the addon reuses a small pool of
     *               output buffers for that key instead of allocating one per call — avoids a per-frame
     *               ~16MB alloc/zero-fill on the main thread. Safe only with a single readback in flight per
     *               key at a time. Call {@link releasePool} when the key is no longer used.
     * @returns A Promise resolving to the converted buffer (size depends on `format`).
     */
    export function readback(source: Buffer | DmabufInfo, width: number, height: number, format?: ReadbackFormat, poolKey?: string): Promise<Buffer>

    /** Release the reused buffer pool for a `poolKey` (see {@link readback}). No-op if unknown. */
    export function releasePool(poolKey: string): void

    /**
     * Two-phase readback (Windows always; macOS when a Metal device exists; Linux when the GPU readback path
     * initialized — the export is ABSENT otherwise, so callers can feature-detect with
     * `typeof readbackConsume === "function"`). Splits {@link readback} so the caller can release the Electron
     * shared texture as soon as the GPU has consumed it, instead of pinning it for the whole read-back —
     * which otherwise drains Electron's compositor frame pool.
     *
     * `readbackConsume` opens the texture, GPU-converts it into an internal buffer keyed by `poolKey`, and
     * resolves once the GPU has finished reading the shared texture (so it can be released). {@link readbackFinish}
     * then does the slow copy-out. A `poolKey` may have at most one consume outstanding at a time.
     *
     * `dstWidth`/`dstHeight` > 0 also GPU-downscales the source to a small tightly-packed BGRA buffer in the
     * SAME pass (for a preview/server/stage consumer); {@link readbackFinish} then resolves `{ main, scaled }`.
     */
    export function readbackConsume(source: Buffer | DmabufInfo, width: number, height: number, format: ReadbackFormat, poolKey: string, dstWidth?: number, dstHeight?: number, targets?: TargetSpec[]): Promise<void>

    /** A per-consumer output of the same pass: the source box-downscaled to width x height, packed as `format`. */
    export type TargetSpec = { width: number; height: number; format: ReadbackFormat }
    /** Whether this platform's GPU backend produces {@link TargetSpec} outputs (Windows today). Else derive them on the CPU. */
    export const targetsSupported: boolean

    /**
     * Phase 2 of the two-phase readback (see {@link readbackConsume}). Copies the consumed frame out. Resolves the converted
     * `Buffer` when no downscale was requested, or `{ main, scaled }` when `dstWidth`/`dstHeight` were given to
     * {@link readbackConsume} (`main` = the `format` buffer, `scaled` = `dstWidth * dstHeight * 4` BGRA).
     */
    export function readbackFinish(poolKey: string, width: number, height: number, format?: ReadbackFormat, dstWidth?: number, dstHeight?: number, dst?: Buffer, targets?: TargetSpec[], targetDsts?: Buffer[]): Promise<Buffer | { main: Buffer; scaled?: Buffer; targets?: Buffer[] }>

    /**
     * SINGLE-DISPATCH readback (Windows / Linux-GPU) — the collapse of {@link readbackConsume} + {@link readbackFinish}
     * into ONE N-API async op (plan §11 fix #1). Runs open + GPU-convert + GPU-wait + copy-out in one worker
     * dispatch and calls `onRelease` EXACTLY ONCE, the instant the GPU has consumed the shared texture (before
     * the slow copy-out) — so the caller releases the Electron texture just as early as two-phase, without the
     * second dispatch/resume hop or the JS-loop hop between phases.
     *
     * `dstWidth`/`dstHeight` > 0 also GPU-downscales in the same pass; the promise then resolves `{ main, scaled }`,
     * else just the `format` `Buffer`. `poolKey` selects the reused output-buffer pool (as {@link readbackFinish}).
     */
    export function readbackOnce(source: Buffer | DmabufInfo, width: number, height: number, format: ReadbackFormat, poolKey: string, dstWidth: number, dstHeight: number, onRelease: () => void): Promise<Buffer | { main: Buffer; scaled: Buffer }>

    /** CPU BGRA -> UYVY (4:2:2, `width * 2 * height`). Synchronous fallback converter. */
    export function convertBgraToUyvy(bgra: Buffer, width: number, height: number): Buffer

    /** CPU BGRA -> UYVA (UYVY plane + full-res alpha plane, `width * 3 * height`). Synchronous fallback. */
    export function convertBgraToUyva(bgra: Buffer, width: number, height: number): Buffer

    /**
     * CPU box-filter downscale of tightly-packed BGRA (`srcWidth * srcHeight * 4`) to `dstWidth * dstHeight * 4`.
     * An order of magnitude cheaper than a 4K `nativeImage.resize` on the JS main thread. Synchronous.
     */
    export function downscaleBgra(bgra: Buffer, srcWidth: number, srcHeight: number, dstWidth: number, dstHeight: number): Buffer
}
