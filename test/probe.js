// Standalone validation harness for the osr-capture addon.
//
// Creates an offscreen (useSharedTexture) BrowserWindow rendering animated content, then on each paint:
//   1) logs the shape of textureInfo once (so we can confirm per-platform field types),
//   2) reads the shared texture back to a CPU buffer via the addon (off the main thread),
//   3) saves the latest frame to <tmpdir>/osr-test.png and logs the readback rate,
//   4) releases the texture (required, or the compositor frame pool drains).
//
// Run with an Electron whose version MATCHES the ABI the addon was built for, from the addon repo root:
//   npx electron@<version> test/probe.js
//
// Needs a real GPU + interactive desktop session (shared textures require GPU compositing; a headless/
// software session will not produce them).

const { app, BrowserWindow, nativeImage } = require("electron")
const path = require("path")
const fs = require("fs")
const os = require("os")

const osr = require("..") // loads ./index.js -> ./build/Release/osr_readback.node

const OUT_PNG = path.join(os.tmpdir(), "osr-test.png")

app.whenReady().then(() => {
    const win = new BrowserWindow({
        width: 1280,
        height: 720,
        show: false,
        webPreferences: { offscreen: { useSharedTexture: true } }
    })
    win.webContents.setFrameRate(30)

    // moving, multi-color content so paints keep firing and correctness is obvious in the PNG
    const html = `<body style="margin:0;background:#101010;overflow:hidden">
      <div id="b" style="position:absolute;top:120px;width:220px;height:220px;border-radius:20px"></div>
      <div style="position:absolute;left:20px;top:20px;color:#fff;font:48px sans-serif">osr-capture test</div>
      <script>
        let x=0,h=0;const b=document.getElementById('b');
        setInterval(()=>{x=(x+9)%1040;h=(h+3)%360;b.style.left=x+'px';b.style.background='hsl('+h+',85%,55%)';},33);
      </script></body>`
    win.loadURL("data:text/html;charset=utf-8," + encodeURIComponent(html))

    let logged = false
    let busy = false
    let frames = 0
    let lastLog = Date.now()
    let lastSave = 0

    win.webContents.on("paint", async (event) => {
        const tex = event.texture
        const info = tex && tex.textureInfo
        if (!info) {
            tex && tex.release && tex.release()
            return
        }

        if (!logged) {
            logged = true
            console.log("[probe] platform:", process.platform)
            console.log("[probe] textureInfo keys:", Object.keys(info))
            console.log("[probe] pixelFormat:", info.pixelFormat, "codedSize:", JSON.stringify(info.codedSize))
            const h = info.sharedTextureHandle
            if (h !== undefined) console.log("[probe] sharedTextureHandle: isBuffer=", Buffer.isBuffer(h), "len=", h && h.length, "type=", typeof h)
            if (info.planes !== undefined) console.log("[probe] planes:", JSON.stringify(info.planes), "modifier:", String(info.modifier), "modifierType:", typeof info.modifier)
        }

        if (busy) {
            tex.release()
            return
        }
        busy = true
        try {
            const { width, height } = info.codedSize
            const source = process.platform === "linux" ? { planes: info.planes, modifier: info.modifier } : info.sharedTextureHandle
            const buf = await osr.readback(source, width, height)
            frames++

            const now = Date.now()
            if (now - lastSave > 1500) {
                lastSave = now
                // BGRA buffer -> NativeImage -> PNG (createFromBitmap expects BGRA on all platforms)
                fs.writeFileSync(OUT_PNG, nativeImage.createFromBitmap(buf, { width, height }).toPNG())
            }
            if (now - lastLog > 1000) {
                console.log(`[probe] readback ${frames}/s ${width}x${height}  bufLen=${buf.length}  -> ${OUT_PNG}`)
                frames = 0
                lastLog = now
            }
        } catch (err) {
            console.error("[probe] readback error:", err)
        } finally {
            busy = false
            tex.release()
        }
    })

    console.log("[probe] running. Open", OUT_PNG, "after a few seconds; it should show the animated test card.")
})
