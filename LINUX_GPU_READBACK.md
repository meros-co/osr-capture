# Linux GPU readback (EGL dmabuf import) — design note

Status: IMPLEMENTED (compiles everywhere; runtime GPU validation requires real hardware — see §8).
Companion docs: `HANDOFF_TESTING.md` (per-platform handle formats, probe), FreeShow
`READBACK_REWORK_PLAN.md` (the Windows readback architecture this mirrors).

## 1. Problem

The original Linux backend (`ReadbackDmabuf` in `src/readback_linux.cc`) mmap'd the Electron OSR dmabuf
and did a single-threaded CPU copy of the full ~33MB 4K BGRA frame out of GPU memory, then a CPU
BGRA→UYVY/UYVA convert. Measured on a 2018 NVIDIA laptop (bare metal): **consume = 88–244ms per 4K
frame** (Windows equivalent: ~5–12ms). Reading uncached VRAM over the bus on one CPU thread is
inherently 10–30x too slow; it backpressures Electron's compositor frame pool, dropping paints /
preview / NDI to single-digit fps. It also only handled LINEAR / INVALID modifiers — tiled dmabufs
errored out entirely.

## 2. Fix — match the Windows architecture

Do the convert (and downscale) **on the GPU**, and read back only the small converted result
(~16MB UYVY instead of ~33MB BGRA; a few hundred KB for the scaled preview), via an async PBO DMA
instead of a CPU walk of uncached memory.

Per-frame pipeline (all on a dedicated GL thread, §4):

1. **Import**: `eglCreateImageKHR(EGL_LINUX_DMA_BUF_EXT)` with the frame's plane fd/offset/pitch and
   DRM format modifier (`EGL_EXT_image_dma_buf_import_modifiers`), fourcc `ARGB8888` (retry
   `XRGB8888`; ABGR is deliberately not tried — it would swap channels). This is the driver's own tiling decode — **tiled/compressed
   modifiers now work**, closing the old TODO. Bind with `glEGLImageTargetTexture2DOES`.
2. **Convert draw(s)** into cached FBO textures (fragment shaders, GLES 3.0, `texelFetch`):
   - format 1 UYVY: one `w/2 × h` RGBA8 target; each texel = (U, Y0, V, Y1) from source pixels
     `2x, 2x+1` — chroma from the even pixel, BT.601 full range with the exact integer coefficients
     of `convert.cc` (77/150/29, −43/−85/128, 128/−107/−21, /256) so GPU and CPU paths agree.
   - format 2 UYVA: the UYVY draw + an alpha-pack draw (`ceil(w/4) × h` RGBA8, 4 alphas per texel).
   - format 0 BGRA / 3 RGBA: swizzle copy draw (`w × h`).
   - dstW/dstH: an extra box-filter downscale draw to `dstW × dstH` BGRA (same-pass, like Windows).
3. **fenceDraw** (`glFenceSync`) — marks "GPU has finished *reading the dmabuf*".
4. **`glReadPixels` into a per-key PBO** (`GL_PIXEL_PACK_BUFFER`, `GL_STREAM_READ`) at plane offsets —
   asynchronous DMA of the FBO results, does not touch the dmabuf.
5. **fenceRead** + `glFlush`; then client-wait **fenceDraw** only, destroy the source texture +
   EGLImage and return → the caller releases the Electron shared texture immediately. The readback
   DMA is still in flight.
   **fd ownership**: per `EGL_EXT_image_dma_buf_import`, a *successful* import transfers ownership
   of the plane fds to EGL (which closes them at its discretion); a failed import does not. Electron
   also owns its fds (closed by `texture.release()`), so the import passes freshly **dup'd** fds —
   EGL closes the dups on success, we close them on failure, and Electron's fds are never touched.
   (Importing Electron's fds directly was a per-frame double close that corrupted the fd table and
   wedged Chromium's OSR frame pool after a few frames: paints stopped entirely.)
6. Finish (later, second phase): client-wait **fenceRead**, `glMapBufferRange` the PBO, hand the
   mapped pointer back to the calling libuv worker which memcpy's into the caller's pooled V8 buffer
   (UYVA alpha rows trimmed from the GL row pitch when `w % 4 != 0`), then an unmap job recycles the
   PBO.

Orientation/stride: source stride is handled by the EGL import attribs (`PLANE0_PITCH`); output rows
are tightly packed (RGBA8 targets, `GL_PACK_ALIGNMENT 1`). Row order is identity by construction:
shaders fetch source row `gl_FragCoord.y` with `texelFetch` (texel row 0 = first row in dmabuf
memory) and `glReadPixels` writes framebuffer row 0 first — memory-row-in == memory-row-out, same as
the old CPU memcpy. If a driver ever imports with inverted orientation, `FS_LINUX_READBACK_FLIP=1`
flips the fetch row in every shader (remote-validation insurance).

## 3. EGL context strategy (headless, one-time init)

Display ladder (first that initializes wins), covering both Mesa and the NVIDIA proprietary driver:

1. `EGL_MESA_platform_surfaceless` (`eglGetPlatformDisplayEXT(EGL_PLATFORM_SURFACELESS_MESA)`) — Mesa.
2. `EGL_EXT_platform_device` (`eglQueryDevicesEXT` → per-device display) — the NVIDIA headless path.
3. `eglGetDisplay(EGL_DEFAULT_DISPLAY)` — last resort.

Then: `eglBindAPI(GLES)`, GLES **3.0** context (needed for `texelFetch`, dynamic loops, `glFenceSync`,
`gl_VertexID` fullscreen triangle — no VBO), `EGL_KHR_no_config_context` when present else an RGBA8
pbuffer-capable config, `eglMakeCurrent` with `EGL_NO_SURFACE` (`EGL_KHR_surfaceless_context`) else a
1x1 pbuffer. gbm was deliberately skipped: surfaceless covers Mesa, platform-device covers NVIDIA, and
it would add a libgbm build/runtime dep for no extra coverage. Required extensions:
`EGL_EXT_image_dma_buf_import` + `GL_OES_EGL_image`; `..._modifiers` is optional (without it only
implicit-modifier imports are attempted).

## 4. Thread model — one dedicated GL thread with a request queue

GL contexts are thread-bound, and readbacks are dispatched from **multiple** libuv async workers.
Chosen design: a single GL thread owns the EGL context and services a FIFO of jobs (consume /
finish-map / unmap / release-key); callers block on a future. Why not per-worker shared contexts:
libuv's pool is anonymous (any of N threads), share-group + per-thread MakeCurrent is the classic
driver-bug minefield, and the GPU serializes the actual work anyway. The GL thread's own occupancy per
4K frame is small — issue draws + wait fenceDraw (~1–3ms GPU) — because the two *slow* stages overlap:
the PBO DMA runs asynchronously between consume and finish, and the 16MB copy-out from the *mapped*
PBO (cached system memory) runs on the calling libuv worker, not the GL thread (map/unmap are GL-thread
jobs; the memcpy between them is not). Multiple concurrent 4K outputs therefore pipeline: key A's DMA
and copy-out overlap key B's draws. All per-key GL state (FBO textures, PBO, pending fences) lives in
a GL-thread-only map — no locks around GL objects at all.

## 5. consume / finish mapping (public interface unchanged)

Linux now exports the same two-phase API as Windows — `readbackConsume` / `readbackFinish` (and
`readbackOnce`, kept dormant-but-working like on Windows) — **only when the GPU path initialized**, so
FreeShow's existing `typeof osr.readbackConsume === "function"` checks (ndiWorker two-phase preference,
OutputLifecycle `hasGpuDownscale` off-main gating) light up the fast path on Linux with **zero JS
changes**, and fall back to single-phase `readback` exactly as today when it didn't.

- `readbackConsume` = steps 1–5 above: returns once the dmabuf is consumed (fenceDraw) → early
  texture release, frame pool freed fast. dstW/dstH GPU-downscale supported (enables off-main mixed
  outputs on Linux).
- `readbackFinish` = step 6: wait fenceRead, map, copy out (`{ main, scaled }` when downscaled).
- `readback` (single-phase) routes through the same GPU consume+finish internally (key `sp#<poolKey>`),
  so the main-path/probe callers get the speedup and tiled support too.
- `releasePool(key)` additionally frees the key's GL resources (FBOs/PBO) on the GL thread.

## 6. Fallback ladder

- Init-time: no EGL display / no GLES3 / missing dma_buf_import or GL_OES_EGL_image →
  `readbackConsume`/`readbackFinish`/`readbackOnce` are **not exported**; `readback` uses the old
  mmap+CPU path (LINEAR/INVALID only; tiled errors as before). One clear stderr log states the active
  backend and why: `[osr-capture] linux readback backend: ...`. `_readbackBackend()` (now also
  exported on Linux) reports `egl-gles3` / `cpu`.
- Runtime: if an EGLImage import fails (e.g. WSL's virtual GPU), consume falls back **per-frame** to
  the CPU path for LINEAR buffers (result stashed in a CPU pending map; finish copies it out — the
  two-phase contract holds, JS never notices); after 3 consecutive import failures the GPU path is
  demoted for good (logged once). Tiled + no GPU import still errors cleanly.
- `FS_LINUX_READBACK=cpu` forces the CPU backend (A/B diagnostic); `FS_LINUX_READBACK_FLIP=1` flips
  rows (§2).

## 7. Build

- `binding.gyp` (Linux block only): adds `src/readback_linux_gpu.cc` and `-lEGL -lGLESv2`.
- apt build deps (WSL/CI): `libegl1-mesa-dev libgles2-mesa-dev` (runtime: `libegl1 libgles2`,
  present on any GPU-composited desktop). DRM fourcc/modifier constants are defined locally — no
  libdrm-dev needed. Build: `npx node-gyp rebuild` with the usual Electron ABI flags
  (see HANDOFF_TESTING.md).

## 8. Validation reality + the one thing most likely to go wrong

WSL's virtual GPU generally lacks EGL dmabuf import — there the code must compile, init-detect, log,
and fall back to CPU cleanly. Real validation is the bare-metal NVIDIA laptop: expect consume to drop
from ~88–244ms to single-digit ms, paints/done/sentReal to reach full rate, and tiled modifiers to
import. Most likely failure: **headless EGL init on the NVIDIA proprietary driver** (platform-device
display needs the NVIDIA EGL ICD + `libnvidia-egl-*` bits; if `eglQueryDevicesEXT` yields only a
software device, we land on CPU fallback) — second candidate: **import orientation/fourcc quirks**
(mitigations: fourcc retry ladder, `FS_LINUX_READBACK_FLIP`).
