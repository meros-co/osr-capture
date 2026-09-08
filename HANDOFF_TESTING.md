# osr-capture — macOS / Linux validation hand-off

You are validating a native Node addon on **macOS** or **Linux**. Windows is already implemented and
validated; macOS and Linux backends are written but have **never been compiled or run**. Your job: build
the addon for the target Electron ABI, run the standalone probe, and confirm it reads the shared GPU
texture back into a correct CPU image — then report results (and fix platform-specific issues if you can).

## What this addon does (context)

Electron offscreen rendering with `webPreferences.offscreen: { useSharedTexture: true }` delivers each frame
as a **GPU texture handle** in the `paint` event (`event.texture`) instead of a CPU bitmap — no CPU copy is
made, and `event.texture.textureInfo.image`/the paint `NativeImage` is empty. `osr-capture` opens that GPU
texture and copies it into a CPU **BGRA** buffer inside an N-API async worker (a background thread), so the
GPU→CPU readback never blocks the JS main thread. It exists so an app (FreeShow) can capture multiple
high-res output windows for NDI/streaming/SDI without the main-thread `toBitmap()` bottleneck.

API: `readback(source, width, height, format = 0) => Promise<Buffer>`. `source` differs by platform (below).
`format` selects the output pixel layout: `0` BGRA (`w*4*h`), `1` UYVY 4:2:2 (`w*2*h`, opaque), `2` UYVA
(`w*2*h` UYVY + `w*h` alpha). Also exported: synchronous CPU fallbacks `convertBgraToUyvy(buf, w, h)` and
`convertBgraToUyva(buf, w, h)`.

### Colour-convert (the `format` arg) — cross-platform status

NDI/SDI want UYVY, not BGRA; handing their SDKs BGRA makes them convert internally (very slow at 4K). So the
addon can emit UYVY/UYVA directly:

- **Windows**: converts on the **GPU** (a D3D11 compute shader) during readback — see `kConvertHLSL` /
  `ReadbackConvert` in `readback_win.cc`. This also shrinks the GPU→CPU copy (UYVY is half of BGRA).
- **Linux**: converts on the **GPU** too when the headless EGL/GLES3 path initializes (EGL dmabuf import +
  fragment-shader convert + async PBO readback — design + validation notes in `LINUX_GPU_READBACK.md`;
  this also adds the two-phase `readbackConsume`/`readbackFinish` exports on Linux and makes tiled
  modifiers work). Falls back to the mmap+`ConvertBgraInPlace` **CPU** path (LINEAR only) when
  EGL/dma_buf_import is unavailable — check `_readbackBackend()` or the
  `[osr-capture] linux readback backend:` stderr line to see which path is active.
- **macOS**: the readback is still BGRA, then `ConvertBgraInPlace` (in `convert.cc`, shared code)
  does a **CPU** BGRA→UYVY/UYVA convert (BT.601 full range). Correct and still avoids the sender SDK's
  convert, but not as fast as a GPU convert. **TODO (future optimization):** a Metal compute shader on
  the `IOSurface`; keep `ConvertBgraInPlace` as the fallback for formats/paths a GPU path doesn't cover.

**Validate the format arg too:** run the probe once per format (0/1/2) and confirm the returned buffer length
is `w*4*h` / `w*2*h` / `w*3*h` respectively, and (for 1/2) that a UYVY→RGB view of the buffer still shows the
test card in the right colours (a 601/709 mismatch would show as slightly off saturation, not garbage).

## Per-platform handle format (authoritative, from Electron `shell/browser/osr/osr_paint_event.h`)

```
#if IS_WIN || IS_MAC
  uintptr_t shared_texture_handle;   // JS: textureInfo.sharedTextureHandle = 8-byte Buffer
                                     //   Windows: a HANDLE to a shared D3D11 texture
                                     //   macOS:   an IOSurface* (pointer valid in THIS process)
#elif IS_LINUX
  std::vector<...> planes;           // JS: textureInfo.planes = [{ fd, stride, offset, size }, ...]
  uint64_t modifier;                 // JS: textureInfo.modifier  (DRM format modifier)
  bool supports_zero_copy_webgpu_import;
#endif
```

- **macOS**: the addon casts the 8-byte handle to `IOSurfaceRef` and does `IOSurfaceLock` → copy → `Unlock`.
- **Linux**: the addon imports the planes + modifier as an EGLImage (GPU path — any modifier, incl.
  tiled); the CPU fallback `mmap`s plane 0's `fd` and copies rows by `stride`, which is **only correct
  for LINEAR buffers** (`modifier == 0`/INVALID). See `LINUX_GPU_READBACK.md`.

## Prerequisites

- A machine with a **real GPU and an interactive desktop session** (shared textures require GPU
  compositing; headless/SSH/software-GL sessions won't produce them).
- Node + a C/C++ toolchain:
  - **macOS**: Xcode command-line tools (`xcode-select --install`).
  - **Linux**: `build-essential`, `python3`, and kernel dmabuf support (standard on desktop distros).
- The Electron version you are validating against. FreeShow currently uses **37.10.3** — use that unless
  told otherwise. The addon MUST be built for the SAME Electron version you run the probe with.

## Steps

1. **Get the addon** (this repo): `git clone https://github.com/schplay/osr-capture && cd osr-capture`

2. **Install the build-time dep** (node-addon-api), skipping the default node-ABI build:
   ```
   npm install --ignore-scripts
   ```

3. **Build for Electron's ABI** (set ELECTRON_VER; set ARCH to `arm64` on Apple Silicon, else `x64`):
   ```
   npx node-gyp rebuild --target=37.10.3 --dist-url=https://electronjs.org/headers --arch=arm64
   ```
   Success = `build/Release/osr_readback.node` is produced with no errors.

4. **Run the probe** with the matching Electron:
   ```
   npx electron@37.10.3 test/probe.js
   ```

## Success criteria

- Terminal logs `[probe] readback N/s WxH ...` continuously with **no** `readback error:` lines.
- `<tmpdir>/osr-test.png` (path is printed) shows the **animated test card** — a moving rounded square that
  cycles colours plus the text "osr-capture test", on a dark background. Correct = recognizable and the
  right colours. Wrong = black, garbage/noise, sheared/striped, or red/blue swapped.
- Note the readback rate and, if you can, try a larger window (edit `width/height` in `test/probe.js`).

Report: platform + OS version + GPU, Electron version + arch, the first `[probe] textureInfo …` log lines,
the readback rate, whether the PNG is correct (attach it), and any error text.

## Likely issues & fixes per platform

**macOS**
- Build error about missing frameworks → confirm `binding.gyp` links `-framework IOSurface` and
  `-framework CoreFoundation` (it does under `OS=='mac'`).
- `readback error: null IOSurface handle` or garbage → the 8-byte handle isn't being interpreted as an
  `IOSurface*`. Check the probe's `sharedTextureHandle isBuffer/len/type` log; if it isn't an 8-byte
  Buffer, adjust extraction in `src/osr_readback.cc`. The pointer is only valid in-process, which is fine
  here (the addon runs in the same process).
- Colours swapped (red↔blue) → the surface is RGBA, not BGRA; check `info.pixelFormat`. If it's `rgba`,
  either swizzle in `readback_mac.mm` or have the caller use `nativeImage.createFromBuffer`/adjust.
- Apple Silicon: make sure you built with `--arch=arm64` and ran an arm64 Electron.

**Linux**
- The probe log `[probe] planes: … modifier: N`. **If `modifier` is not `0`** (and not the linear/invalid
  sentinel), the addon returns `readback error: non-linear dmabuf modifier requires GPU import` — this is
  expected and means the tiled path is needed: implement `eglCreateImageKHR(EGL_LINUX_DMA_BUF_EXT)` →
  bind to a GL texture → FBO → `glReadPixels` in `readback_linux.cc` (this is the documented TODO).
  Report the exact `modifier` value + whether `supportsZeroCopyWebgpuImport` is set.
- `readback error: mmap of dmabuf failed` with `EBADF` → the plane `fd` was closed before the async worker
  ran. Fix: in `src/osr_readback.cc` (Linux branch, main thread, synchronously) `dup()` each `fd` right
  when the call arrives and pass the dup to the worker; `close()` it after `munmap`. (Electron may hand a
  short-lived fd.)
- Garbage despite `modifier == 0` → check `stride`/`offset` usage against the probe's `planes` values;
  ensure `mmap` length covers `offset + stride*height`.
- Wayland vs X11 shouldn't matter for the readback itself, but confirm GPU compositing is active.

## Where this plugs into FreeShow (for reference, not required to test here)

FreeShow calls it from `src/electron/output/helpers/OutputLifecycle.ts` (`attachOsrSharedTexture`):
per paint it throttles, calls `readback(source, w, h)` off-thread, wraps the buffer via
`nativeImage.createFromBitmap`, and always calls `texture.release()`. `source` is the raw
`sharedTextureHandle` on Win/Mac and `{ planes, modifier }` on Linux. If the addon can't load or read back,
FreeShow falls back to CPU-mode offscreen capture, so nothing breaks while this is being validated.
