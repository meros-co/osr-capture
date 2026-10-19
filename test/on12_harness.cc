// on12_harness.cc — Stage-2 sub-step A: standalone D3D11On12 cached-readback harness (plan §10/§11).
//
// PURPOSE (READBACK_REWORK_PLAN.md, "Stage 2 sub-step A"): prove, IN ISOLATION from FreeShow/Electron,
// that the D3D11On12 path — existing D3D11 consume code UNCHANGED on an 11-on-12 device, convert output
// in a wrapped D3D12 UAV buffer, CopyBufferRegion on the D3D12 COPY queue into a D3D12_HEAP_TYPE_READBACK
// (WRITE_BACK/CPU-cached) buffer, completion via a SINGLE-DEVICE copy-queue fence
// (SetEventOnCompletion + WaitForSingleObject — NOT the abandoned cross-device ID3D11Fence that TDR'd) —
// is correct (no tears / stale frames), TDR-safe, and moves copy-out from WC-class (~10-11 ms/4K frame)
// to cached-memcpy class. Gates (a)-(d) below are stdout counters; ALL must pass before sub-step B
// touches readback_win.cc's live consume path.
//
// CODE SHARING (no fork): this file #includes ../src/readback_win.cc — one translation unit — so the
// convert that runs here is the EXACT ReadbackContext::DispatchConvert + kConvertHLSL shipping in the
// addon. The 11On12 device and a wrapped D3D12 buffer are injected into the context's public members
// (EnsureDevice/EnsureOutBuffer early-return on injected state), so sub-step B is a drop-in. The "d3d11"
// baseline variant calls the addon's public ReadbackConsume/ReadbackFinish — the live path verbatim,
// including the self-calibrating CopyPool and the WaitGpu spin — for a same-machine A/B.
//
// BUILD (same toolchain as the addon; builds the addon AND this exe):
//   npx node-gyp rebuild --target=37.10.3 --dist-url=https://electronjs.org/headers --arch=x64
//   -> build/Release/on12_harness.exe
// The exe needs NO Electron/node at runtime (pure Win32 + D3D). It needs a GPU only to RUN.
//
// RUN (defaults exercise the full gate matrix — paste ALL stdout back):
//   build\Release\on12_harness.exe
//   options: --secs=45 --rate-secs=10 --report=5 --width=3840 --height=2160 --alt-width=2560
//            --alt-height=1440 --format=1|2 --variant=all|on12|d3d11 --mutex=all|1|0 --fps=60
//            --streams=3 --no-resize --no-stopstart --no-validate --skip-soak --skip-rate --quick
//            --debug-layer --live=d3d11|on12 (sub-step B: backend for the LIVE ReadbackConsume/Finish
//            arm — default pins FS_READBACK=d3d11 so "d3d11" stays the WC baseline; --live=on12 lets the
//            §5 ladder engage, soaking the folded-in 11On12 consume path under the full gate matrix)
//
// GATES (all measured — no target constants, per plan §11 Principle):
//  (a) [GATE-A] mismatchFrames MUST be 0 over the soak (3 streams x 4K60, keyed-mutex AND no-mutex,
//      mid-run resize + producer stop/start). Validation: exact CPU port of the convert shader; rows>=1
//      against 16 precomputed frameIndex%16 patterns, row 0 against the FULL frameIndex (stale-frame
//      detector, with a stale-by-k probe on mismatch). A startup probe verifies the GPU float->unorm
//      round-trip is byte-exact so the CPU model is a valid oracle.
//  (b) [GATE-B] deviceRemoved / watchdogTimeouts (2s fence waits) / acquireFails / consume errors MUST
//      all be 0 — the TDR/hang class the abandoned fence build taught us to gate on.
//  (c) [GATE-C] per-frame timing split: open+convert vs copy-queue vs copy-out vs total, on12 vs d3d11.
//      The HYPOTHESIS under measurement: copy-out drops from WC-class to cached-memcpy class.
//  (c2)[GATE-C2] sustained isolated consume rate (unthrottled) at 1/2/3 streams + consumer-thread CPU
//      cores, on12 vs d3d11 (the WaitGpu spin burn should collapse to an event block).
//  (d) [GATE-D] clean teardown per phase: wrapped resources released before the final Flush, fences
//      drained, all devices released, shared NT handles closed; process exits 0.
//
// SCOPE (plan): correctness + copy/GPU timing ONLY. True rtt / unique-fps are judged in FreeShow after
// sub-step B. This harness does not touch FreeShow, N-API surfaces, or the live consume path.

#ifndef NOMINMAX
#define NOMINMAX
#endif

// The code under test — the whole addon Windows backend in this TU (ReadbackContext, kConvertHLSL,
// CopyPool, public osrcap API). Do NOT fork any of it here.
#include "../src/readback_win.cc"

#include <d3d11on12.h>
#include <d3d12.h>
#include <dxgi1_4.h>

#include <array>
#include <cinttypes>
#include <cstdarg>
#include <functional>

using namespace osrcap;  // brings ReadbackContext (unnamed ns inside osrcap) into scope in this TU

namespace harness {

// ---------------------------------------------------------------------------------------------------
// small utils
// ---------------------------------------------------------------------------------------------------
static double NowMs() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static double ThreadCpuSeconds() {  // current thread, kernel+user
    FILETIME c, e, k, u;
    if (!GetThreadTimes(GetCurrentThread(), &c, &e, &k, &u)) return 0;
    auto f = [](FILETIME t) { return (((uint64_t)t.dwHighDateTime << 32) | t.dwLowDateTime) * 1e-7; };
    return f(k) + f(u);
}

static double ProcessCpuSeconds() {
    FILETIME c, e, k, u;
    if (!GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)) return 0;
    auto f = [](FILETIME t) { return (((uint64_t)t.dwHighDateTime << 32) | t.dwLowDateTime) * 1e-7; };
    return f(k) + f(u);
}

static std::string Fmt(const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return buf;
}

static inline int Clamp8(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }

// ---------------------------------------------------------------------------------------------------
// Deterministic pattern + exact CPU model of the convert shader (the checksum oracle)
// ---------------------------------------------------------------------------------------------------
// MUST match kProducerHLSL's HashU bit-for-bit (uint32 arithmetic).
static inline uint32_t HashPix(uint32_t f, uint32_t x, uint32_t y) {
    uint32_t h = f * 0x9E3779B1u;
    h ^= x * 0x85EBCA6Bu;
    h ^= y * 0xC2B2AE35u;
    h ^= h >> 16;
    h *= 0x7FEB352Du;
    h ^= h >> 15;
    h *= 0x846CA68Bu;
    h ^= h >> 16;
    return h;
}

// BGRA bytes of pixel (x,y) of pattern `sel` (the producer writes B=h, G=h>>8, R=h>>16, A=h>>24).
static inline void PixelBgra(uint32_t sel, uint32_t x, uint32_t y, uint8_t out[4]) {
    const uint32_t h = HashPix(sel, x, y);
    out[0] = (uint8_t)(h & 0xff);
    out[1] = (uint8_t)((h >> 8) & 0xff);
    out[2] = (uint8_t)((h >> 16) & 0xff);
    out[3] = (uint8_t)((h >> 24) & 0xff);
}

// Exact integer port of kConvertHLSL (readback_win.cc) for one output row. UYVY rows are independent
// (chroma pairs are horizontal only), so per-row computation is valid. alphaRow may be null (format 1).
static void ExpectedRow(uint32_t sel, uint32_t y, uint32_t w, uint8_t* uyvyRow, uint8_t* alphaRow) {
    for (uint32_t gx = 0; gx * 4 < w; ++gx) {
        const uint32_t px = gx * 4;
        int r[4], g[4], b[4], a[4], yy[4];
        for (int i = 0; i < 4; ++i) {
            if (px + i < w) {
                uint8_t p[4];
                PixelBgra(sel, px + i, y, p);
                b[i] = p[0]; g[i] = p[1]; r[i] = p[2]; a[i] = p[3];
            } else {
                b[i] = g[i] = r[i] = a[i] = 0;  // Texture2D.Load OOB returns 0
            }
            yy[i] = (77 * r[i] + 150 * g[i] + 29 * b[i]) >> 8;  // shader has no clamp here (max is 255)
        }
        const int u01 = Clamp8(((-43 * r[0] - 85 * g[0] + 128 * b[0]) >> 8) + 128);
        const int v01 = Clamp8(((128 * r[0] - 107 * g[0] - 21 * b[0]) >> 8) + 128);
        const int u23 = Clamp8(((-43 * r[2] - 85 * g[2] + 128 * b[2]) >> 8) + 128);
        const int v23 = Clamp8(((128 * r[2] - 107 * g[2] - 21 * b[2]) >> 8) + 128);
        uint8_t* o = uyvyRow + (size_t)gx * 8;
        o[0] = (uint8_t)u01; o[1] = (uint8_t)yy[0]; o[2] = (uint8_t)v01; o[3] = (uint8_t)yy[1];
        o[4] = (uint8_t)u23; o[5] = (uint8_t)yy[2]; o[6] = (uint8_t)v23; o[7] = (uint8_t)yy[3];
        if (alphaRow) {
            uint8_t* ao = alphaRow + (size_t)gx * 4;
            ao[0] = (uint8_t)a[0]; ao[1] = (uint8_t)a[1]; ao[2] = (uint8_t)a[2]; ao[3] = (uint8_t)a[3];
        }
    }
}

// Precomputed expected outputs for the 16 frameIndex%16 patterns of one (w,h,format). Rows >= 1 of a
// frame are pattern (frameIndex & 15); row 0 is pattern (full frameIndex) and is computed per frame
// (cheap: one row) so ANY stale frame is caught exactly, not just mod-16 aliases.
struct ExpectedSet {
    uint32_t w = 0, h = 0;
    int format = 1;
    size_t uyvySize = 0, total = 0;
    std::array<std::vector<uint8_t>, 16> mods;

    void Build(uint32_t W, uint32_t H, int fmt) {
        w = W; h = H; format = fmt;
        uyvySize = (size_t)w * 2 * h;
        total = fmt == 2 ? uyvySize + (size_t)w * h : uyvySize;
        const double t0 = NowMs();
        std::vector<std::thread> ths;
        for (int m = 0; m < 16; ++m) {
            ths.emplace_back([this, m] {
                auto& buf = mods[m];
                buf.resize(total);
                for (uint32_t y = 0; y < h; ++y)
                    ExpectedRow((uint32_t)m, y, w, buf.data() + (size_t)y * w * 2,
                                format == 2 ? buf.data() + uyvySize + (size_t)y * w : nullptr);
            });
        }
        for (auto& t : ths) t.join();
        printf("[EXPECTED] built %ux%u fmt%d x16 patterns in %.1fs (%.0f MB)\n", w, h, format,
               (NowMs() - t0) / 1000.0, 16.0 * total / 1e6);
        fflush(stdout);
    }
};

struct ExpectedCache {
    std::vector<std::unique_ptr<ExpectedSet>> sets;
    const ExpectedSet* Get(uint32_t w, uint32_t h) const {
        for (auto& s : sets)
            if (s->w == w && s->h == h) return s.get();
        return nullptr;
    }
    void Build(uint32_t w, uint32_t h, int fmt) {
        if (Get(w, h)) return;
        sets.push_back(std::make_unique<ExpectedSet>());
        sets.back()->Build(w, h, fmt);
    }
};

// Validate one consumed frame. Returns true if byte-exact. On row-0 mismatch, probes whether the frame
// is a STALE older frame (the silent-wrong-frame failure class the plan's texture-LRU note warns about).
static bool ValidateFrame(const uint8_t* got, const ExpectedSet& es, uint32_t frame, std::string& detail, int& staleBy) {
    staleBy = 0;
    const uint32_t w = es.w, h = es.h;
    const size_t rowB = (size_t)w * 2;
    bool ok = true;
    size_t firstOff = (size_t)-1;

    // rows 1..h-1 vs the precomputed mod-16 pattern
    const auto& exp = es.mods[frame & 15];
    if (h > 1 && memcmp(got + rowB, exp.data() + rowB, rowB * (h - 1)) != 0) {
        ok = false;
        for (size_t i = rowB; i < es.uyvySize; ++i)
            if (got[i] != exp[i]) { firstOff = i; break; }
    }
    if (es.format == 2 && ok) {
        const size_t aOff = es.uyvySize + w;  // alpha rows 1..h-1
        if (h > 1 && memcmp(got + aOff, exp.data() + aOff, (size_t)w * (h - 1)) != 0) {
            ok = false;
            firstOff = aOff;
        }
    }

    // row 0 vs the FULL frame index (exact staleness detector)
    std::vector<uint8_t> r0(rowB), a0(es.format == 2 ? w : 0);
    ExpectedRow(frame, 0, w, r0.data(), es.format == 2 ? a0.data() : nullptr);
    if (memcmp(got, r0.data(), rowB) != 0) {
        ok = false;
        if (firstOff == (size_t)-1)
            for (size_t i = 0; i < rowB; ++i)
                if (got[i] != r0[i]) { firstOff = i; break; }
        for (int k = 1; k <= 32 && k <= (int)frame; ++k) {  // stale-by-k probe
            std::vector<uint8_t> rk(rowB);
            ExpectedRow(frame - k, 0, w, rk.data(), nullptr);
            if (memcmp(got, rk.data(), rowB) == 0) { staleBy = k; break; }
        }
    } else if (es.format == 2 && memcmp(got + es.uyvySize, a0.data(), w) != 0) {
        ok = false;
        if (firstOff == (size_t)-1) firstOff = es.uyvySize;
    }

    if (!ok)
        detail = Fmt("frame=%u %ux%u firstDiffOff=%zu got=0x%02x staleBy=%d", frame, w, h,
                     firstOff == (size_t)-1 ? 0 : firstOff,
                     firstOff == (size_t)-1 ? 0 : got[firstOff], staleBy);
    return ok;
}

// ---------------------------------------------------------------------------------------------------
// Producer — plain D3D11, stands in for Electron's GPU process. Renders hash(frameIndex,x,y) into a
// ring of shared textures (NT handle, keyed-mutex and no-mutex variants), like Electron OSR paints.
// ---------------------------------------------------------------------------------------------------
static const char* kProducerHLSL = R"HLSL(
cbuffer PParams : register(b0) { uint gFrameFull; uint gFrameMod; uint gPad0; uint gPad1; };
struct VSOut { float4 pos : SV_Position; };
VSOut VSMain(uint vid : SV_VertexID) {
    VSOut o;
    float2 p = float2((vid << 1) & 2, vid & 2);
    o.pos = float4(p * float2(2, -2) + float2(-1, 1), 0, 1);
    return o;
}
uint HashU(uint f, uint x, uint y) {
    uint h = f * 0x9E3779B1u;
    h ^= x * 0x85EBCA6Bu;
    h ^= y * 0xC2B2AE35u;
    h ^= h >> 16; h *= 0x7FEB352Du;
    h ^= h >> 15; h *= 0x846CA68Bu;
    h ^= h >> 16;
    return h;
}
float4 PSMain(float4 pos : SV_Position) : SV_Target {
    uint x = (uint)pos.x;
    uint y = (uint)pos.y;
    uint h = HashU(y == 0 ? gFrameFull : gFrameMod, x, y);
    // B8G8R8A8 target: .r stores R etc.  B=h, G=h>>8, R=h>>16, A=h>>24 (matches PixelBgra).
    return float4((h >> 16) & 0xff, (h >> 8) & 0xff, h & 0xff, (h >> 24) & 0xff) / 255.0;
}
)HLSL";

struct ProdParams {
    uint32_t frameFull, frameMod, pad0, pad1;
};

static const int kSlots = 3;  // shared-texture ring depth per stream (like Electron's frame pool)

struct Producer {
    ComPtr<ID3D11Device> dev;
    ComPtr<ID3D11DeviceContext> ctx;
    ComPtr<ID3D11VertexShader> vs;
    ComPtr<ID3D11PixelShader> ps;
    ComPtr<ID3D11Buffer> cb;
    ComPtr<ID3D11Query> evQuery;
    struct Slot {
        ComPtr<ID3D11Texture2D> tex;
        ComPtr<ID3D11RenderTargetView> rtv;
        ComPtr<IDXGIKeyedMutex> km;
        HANDLE handle = nullptr;
    };
    std::vector<Slot> slots;
    uint32_t w = 0, h = 0;
    bool useMutex = false;

    bool Init(std::string& err) {
        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
        D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0;
        if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags, nullptr, 0,
                                     D3D11_SDK_VERSION, &dev, &fl, &ctx))) {
            err = "producer D3D11CreateDevice failed";
            return false;
        }
        ComPtr<ID3DBlob> vsb, psb, errb;
        if (FAILED(D3DCompile(kProducerHLSL, strlen(kProducerHLSL), "producer.hlsl", nullptr, nullptr,
                              "VSMain", "vs_5_0", 0, 0, &vsb, &errb)) ||
            FAILED(D3DCompile(kProducerHLSL, strlen(kProducerHLSL), "producer.hlsl", nullptr, nullptr,
                              "PSMain", "ps_5_0", 0, 0, &psb, &errb))) {
            err = std::string("producer shader compile failed: ") +
                  (errb ? (const char*)errb->GetBufferPointer() : "?");
            return false;
        }
        if (FAILED(dev->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &vs)) ||
            FAILED(dev->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), nullptr, &ps))) {
            err = "producer shader create failed";
            return false;
        }
        D3D11_BUFFER_DESC bd = {};
        bd.ByteWidth = sizeof(ProdParams);
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        if (FAILED(dev->CreateBuffer(&bd, nullptr, &cb))) { err = "producer cb failed"; return false; }
        D3D11_QUERY_DESC qd = {};
        qd.Query = D3D11_QUERY_EVENT;
        if (FAILED(dev->CreateQuery(&qd, &evQuery))) { err = "producer query failed"; return false; }
        return true;
    }

    LUID AdapterLuid() {
        ComPtr<IDXGIDevice> dxgiDev;
        ComPtr<IDXGIAdapter> ad;
        DXGI_ADAPTER_DESC desc = {};
        if (SUCCEEDED(dev.As(&dxgiDev)) && SUCCEEDED(dxgiDev->GetAdapter(&ad))) ad->GetDesc(&desc);
        return desc.AdapterLuid;
    }

    bool CreateSlots(uint32_t W, uint32_t H, bool mutexVariant, std::string& err) {
        w = W; h = H; useMutex = mutexVariant;
        slots.assign(kSlots, Slot{});
        for (int i = 0; i < kSlots; ++i) {
            D3D11_TEXTURE2D_DESC d = {};
            d.Width = w; d.Height = h; d.MipLevels = 1; d.ArraySize = 1;
            d.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            d.SampleDesc.Count = 1;
            d.Usage = D3D11_USAGE_DEFAULT;
            d.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
            d.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE |
                          (mutexVariant ? D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX : D3D11_RESOURCE_MISC_SHARED);
            if (FAILED(dev->CreateTexture2D(&d, nullptr, &slots[i].tex))) { err = "shared tex create failed"; return false; }
            if (FAILED(dev->CreateRenderTargetView(slots[i].tex.Get(), nullptr, &slots[i].rtv))) { err = "rtv failed"; return false; }
            if (mutexVariant && FAILED(slots[i].tex.As(&slots[i].km))) { err = "producer keyed mutex query failed"; return false; }
            ComPtr<IDXGIResource1> res1;
            if (FAILED(slots[i].tex.As(&res1)) ||
                FAILED(res1->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                                                nullptr, &slots[i].handle))) {
                err = "CreateSharedHandle failed";
                return false;
            }
        }
        return true;
    }

    void DestroySlots() {
        for (auto& s : slots) {
            if (s.handle) CloseHandle(s.handle);
            s.handle = nullptr;
            s.km.Reset();
            s.rtv.Reset();
            s.tex.Reset();
        }
        slots.clear();
    }

    bool Render(int slot, uint32_t frameFull, std::string& err) {
        Slot& s = slots[slot];
        ProdParams p{frameFull, frameFull & 15u, 0, 0};
        ctx->UpdateSubresource(cb.Get(), 0, nullptr, &p, 0, 0);
        if (s.km && s.km->AcquireSync(0, 5000) != S_OK) { err = "producer AcquireSync failed"; return false; }
        D3D11_VIEWPORT vp{0, 0, (float)w, (float)h, 0, 1};
        ctx->RSSetViewports(1, &vp);
        ID3D11RenderTargetView* rtvs[] = {s.rtv.Get()};
        ctx->OMSetRenderTargets(1, rtvs, nullptr);
        ctx->IASetInputLayout(nullptr);
        ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx->VSSetShader(vs.Get(), nullptr, 0);
        ctx->PSSetShader(ps.Get(), nullptr, 0);
        ID3D11Buffer* cbs[] = {cb.Get()};
        ctx->PSSetConstantBuffers(0, 1, cbs);
        ctx->Draw(3, 0);
        ID3D11RenderTargetView* nullRtv[] = {nullptr};
        ctx->OMSetRenderTargets(1, nullRtv, nullptr);
        if (s.km) {
            s.km->ReleaseSync(0);  // keyed mutex orders the consumer's GPU reads after this render
        } else {
            // no-mutex variant: CPU-complete before publishing (Electron orders paints itself)
            ctx->End(evQuery.Get());
            ctx->Flush();
            while (ctx->GetData(evQuery.Get(), nullptr, 0, 0) != S_OK) std::this_thread::yield();
        }
        return true;
    }

    // Startup oracle probe: verify the GPU's float->unorm store of n/255 recovers byte n exactly, so the
    // CPU model (PixelBgra/ExpectedRow) is a valid checksum oracle on this GPU/driver. If this fails,
    // gate (a) cannot be judged here — that itself is a reportable finding.
    bool SelfProbe(std::string& err) {
        const uint32_t frame = 300;  // frameFull=300, frameMod=12 -> row0 differs from other rows
        if (!Render(0, frame, err)) return false;
        D3D11_TEXTURE2D_DESC d = {};
        slots[0].tex->GetDesc(&d);
        d.Usage = D3D11_USAGE_STAGING;
        d.BindFlags = 0;
        d.MiscFlags = 0;
        d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ComPtr<ID3D11Texture2D> st;
        if (FAILED(dev->CreateTexture2D(&d, nullptr, &st))) { err = "probe staging failed"; return false; }
        if (slots[0].km && slots[0].km->AcquireSync(0, 5000) != S_OK) { err = "probe acquire failed"; return false; }
        ctx->CopyResource(st.Get(), slots[0].tex.Get());
        if (slots[0].km) slots[0].km->ReleaseSync(0);
        D3D11_MAPPED_SUBRESOURCE map = {};
        if (FAILED(ctx->Map(st.Get(), 0, D3D11_MAP_READ, 0, &map))) { err = "probe map failed"; return false; }
        bool ok = true;
        for (uint32_t y = 0; y < h && ok; ++y) {
            const uint8_t* row = (const uint8_t*)map.pData + (size_t)y * map.RowPitch;
            const uint32_t sel = y == 0 ? frame : (frame & 15);
            for (uint32_t x = 0; x < w; ++x) {
                uint8_t e[4];
                PixelBgra(sel, x, y, e);
                if (memcmp(row + (size_t)x * 4, e, 4) != 0) {
                    err = Fmt("float->unorm NOT byte-exact at (%u,%u): got %02x%02x%02x%02x want %02x%02x%02x%02x",
                              x, y, row[x * 4], row[x * 4 + 1], row[x * 4 + 2], row[x * 4 + 3], e[0], e[1], e[2], e[3]);
                    ok = false;
                    break;
                }
            }
        }
        ctx->Unmap(st.Get(), 0);
        return ok;
    }
};

// ---------------------------------------------------------------------------------------------------
// On12 consumer — THE PATH UNDER TEST
// ---------------------------------------------------------------------------------------------------
struct FrameTimings {
    double schedMs = 0;    // open shared + AcquireSync + DispatchConvert (existing code) + Flush
    double convMs = 0;     // GPU convert completion (direct-queue fence)
    double copyQMs = 0;    // D3D12 COPY-queue DMA into the READBACK heap (copy-queue fence)
    double copyOutMs = 0;  // CPU memcpy from the persistently-mapped cached readback buffer
    double totalMs = 0;
    size_t bytes = 0;
};

struct On12Consumer {
    ComPtr<ID3D12Device> d12;
    ComPtr<ID3D12CommandQueue> qDirect, qCopy;
    ComPtr<ID3D12Fence> fence;
    UINT64 fenceVal = 0;
    HANDLE evt = nullptr;
    ComPtr<ID3D12CommandAllocator> copyAlloc;
    ComPtr<ID3D12GraphicsCommandList> copyList;
    ComPtr<ID3D11Device> d11;
    ComPtr<ID3D11DeviceContext> d11ctx;
    ComPtr<ID3D11Device1> d11_1;
    ComPtr<ID3D11On12Device> on12;
    ComPtr<ID3D12Resource> convBuf12;  // DEFAULT-heap UAV buffer: the convert output, D3D12-owned
    ComPtr<ID3D12Resource> rbBuf;      // READBACK heap (WRITE_BACK / CPU-cached) — the whole point
    void* rbPtr = nullptr;
    size_t curTotal = 0;
    ReadbackContext ctx;  // the addon's context — device/context/outBuf INJECTED, code unchanged
    std::vector<uint8_t> cpuBuf;
    // gate B counters
    uint64_t watchdogTimeouts = 0, deviceRemoved = 0, acquireFails = 0, hrFails = 0, configMismatch = 0;
    bool fatal = false;

    bool Init(LUID luid, bool debugLayer, std::string& err) {
        if (debugLayer) {
            ComPtr<ID3D12Debug> dbg;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) dbg->EnableDebugLayer();
        }
        ComPtr<IDXGIFactory1> fac;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&fac)))) { err = "CreateDXGIFactory1 failed"; return false; }
        ComPtr<IDXGIAdapter1> adapter;
        for (UINT i = 0; fac->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 d = {};
            adapter->GetDesc1(&d);
            if (d.AdapterLuid.LowPart == luid.LowPart && d.AdapterLuid.HighPart == luid.HighPart) break;
            adapter.Reset();
        }
        HRESULT hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d12));
        if (FAILED(hr)) { err = Fmt("D3D12CreateDevice failed hr=0x%08lx", hr); return false; }
        D3D12_COMMAND_QUEUE_DESC qd = {};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (FAILED(d12->CreateCommandQueue(&qd, IID_PPV_ARGS(&qDirect)))) { err = "direct queue failed"; return false; }
        qd.Type = D3D12_COMMAND_LIST_TYPE_COPY;
        if (FAILED(d12->CreateCommandQueue(&qd, IID_PPV_ARGS(&qCopy)))) { err = "copy queue failed"; return false; }
        if (FAILED(d12->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) { err = "fence failed"; return false; }
        evt = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!evt) { err = "event failed"; return false; }
        if (FAILED(d12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&copyAlloc))) ||
            FAILED(d12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, copyAlloc.Get(), nullptr,
                                          IID_PPV_ARGS(&copyList)))) {
            err = "copy list failed";
            return false;
        }
        copyList->Close();  // created open; keep the per-frame Reset pattern uniform
        IUnknown* queues[] = {qDirect.Get()};
        hr = D3D11On12CreateDevice(d12.Get(), D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0, queues, 1, 0,
                                   &d11, &d11ctx, nullptr);
        if (FAILED(hr)) { err = Fmt("D3D11On12CreateDevice failed hr=0x%08lx", hr); return false; }
        if (FAILED(d11.As(&d11_1))) { err = "ID3D11Device1 query failed"; return false; }
        if (FAILED(d11.As(&on12))) { err = "ID3D11On12Device query failed"; return false; }
        // Inject the 11-on-12 device into the addon's context: EnsureDevice() early-returns on device1,
        // so every line of the consume code (EnsureShader, DispatchConvert, kConvertHLSL) runs UNCHANGED.
        ctx.device = d11;
        ctx.device1 = d11_1;
        ctx.context = d11ctx;
        return true;
    }

    // (Re)create the wrapped convert-output buffer + readback buffer for `total` bytes and inject them
    // into the addon context (EnsureOutBuffer early-returns on outBuf && outBufSize == total).
    bool EnsureSize(size_t total, std::string& err) {
        if (curTotal == total) return true;
        ctx.outUav.Reset();
        ctx.outBuf.Reset();
        ctx.outBufSize = 0;
        convBuf12.Reset();
        if (rbPtr) { rbBuf->Unmap(0, nullptr); rbPtr = nullptr; }
        rbBuf.Reset();
        curTotal = 0;

        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC bd = {};
        bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bd.Width = total;
        bd.Height = 1;
        bd.DepthOrArraySize = 1;
        bd.MipLevels = 1;
        bd.Format = DXGI_FORMAT_UNKNOWN;
        bd.SampleDesc.Count = 1;
        bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        bd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        HRESULT hr = d12->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd,
                                                  D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&convBuf12));
        if (FAILED(hr)) { err = Fmt("convBuf12 create failed hr=0x%08lx", hr); return false; }

        // Wrap the D3D12 buffer into D3D11 so the UNCHANGED compute dispatch writes straight into it.
        // Buffers have simultaneous-access state semantics (promote from / decay to COMMON at every
        // ExecuteCommandLists), so COMMON in/out lets the COPY queue promote to COPY_SOURCE implicitly.
        D3D11_RESOURCE_FLAGS f11 = {};
        f11.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        f11.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        ComPtr<ID3D11Buffer> buf11;
        hr = on12->CreateWrappedResource(convBuf12.Get(), &f11, D3D12_RESOURCE_STATE_COMMON,
                                         D3D12_RESOURCE_STATE_COMMON, IID_PPV_ARGS(&buf11));
        if (FAILED(hr)) { err = Fmt("CreateWrappedResource(buffer) failed hr=0x%08lx", hr); return false; }

        // Same UAV desc as the addon's EnsureOutBuffer — RAW R32_TYPELESS over the whole buffer.
        D3D11_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.Format = DXGI_FORMAT_R32_TYPELESS;
        ud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        ud.Buffer.FirstElement = 0;
        ud.Buffer.NumElements = (UINT)(total / 4);
        ud.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
        ComPtr<ID3D11UnorderedAccessView> uav;
        {
            // UAV creation needs the resource acquired on the 11-on-12 device
            ID3D11Resource* res[] = {buf11.Get()};
            on12->AcquireWrappedResources(res, 1);
            hr = d11->CreateUnorderedAccessView(buf11.Get(), &ud, &uav);
            on12->ReleaseWrappedResources(res, 1);
        }
        if (FAILED(hr)) { err = Fmt("UAV on wrapped buffer failed hr=0x%08lx", hr); return false; }

        hp.Type = D3D12_HEAP_TYPE_READBACK;
        bd.Flags = D3D12_RESOURCE_FLAG_NONE;
        hr = d12->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd,
                                          D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rbBuf));
        if (FAILED(hr)) { err = Fmt("readback buffer create failed hr=0x%08lx", hr); return false; }
        if (FAILED(rbBuf->Map(0, nullptr, &rbPtr))) { err = "readback Map failed"; return false; }  // persistent

        ctx.outBuf = buf11;
        ctx.outUav = uav;
        ctx.outBufSize = total;
        cpuBuf.resize(total);
        curTotal = total;
        return true;
    }

    bool WaitFence(UINT64 v, DWORD ms) {
        if (fence->GetCompletedValue() >= v) return true;
        fence->SetEventOnCompletion(v, evt);
        if (WaitForSingleObject(evt, ms) == WAIT_OBJECT_0) return true;
        ++watchdogTimeouts;
        if (d12->GetDeviceRemovedReason() != S_OK) ++deviceRemoved;
        fatal = true;  // never busy-retry a hung GPU — that's the TDR class gate (b) exists for
        return false;
    }

    bool ConsumeFrame(uint64_t handle, uint32_t w, uint32_t h, int format, bool expectMutex,
                      FrameTimings& t, std::string& err) {
        const size_t uyvy = (size_t)w * 2 * h;
        const size_t total = format == 2 ? uyvy + (size_t)w * h : uyvy;
        if (!EnsureSize(total, err)) return false;

        const double t0 = NowMs();
        ComPtr<ID3D11Texture2D> shared;
        HRESULT hr = d11_1->OpenSharedResource1((HANDLE)(uintptr_t)handle, IID_PPV_ARGS(&shared));
        if (FAILED(hr) || !shared) {
            ++hrFails;
            err = Fmt("OpenSharedResource1 failed hr=0x%08lx", hr);
            return false;
        }
        ComPtr<IDXGIKeyedMutex> km;
        const bool haveKm = SUCCEEDED(shared.As(&km)) && km;
        if (haveKm != expectMutex) ++configMismatch;
        if (haveKm && km->AcquireSync(0, 1000) != S_OK) {
            ++acquireFails;
            err = "AcquireSync failed/timed out";
            return false;
        }

        ID3D11Resource* wrapped[] = {ctx.outBuf.Get()};
        on12->AcquireWrappedResources(wrapped, 1);
        const bool convOk = ctx.DispatchConvert(shared.Get(), w, h, format == 2, err);  // UNCHANGED addon code
        on12->ReleaseWrappedResources(wrapped, 1);
        d11ctx->Flush();  // submit the 11-on-12 work to the direct queue NOW
        const double t1 = NowMs();
        if (!convOk) {
            if (haveKm) km->ReleaseSync(0);
            return false;
        }

        // Single-device fences: direct-queue signal orders the COPY queue after the convert; the
        // copy-queue signal is the completion the CPU blocks on (event wait — no spin, no cross-device
        // fence, no TDR bait).
        const UINT64 vConv = ++fenceVal;
        qDirect->Signal(fence.Get(), vConv);
        qCopy->Wait(fence.Get(), vConv);
        copyAlloc->Reset();
        copyList->Reset(copyAlloc.Get(), nullptr);
        copyList->CopyBufferRegion(rbBuf.Get(), 0, convBuf12.Get(), 0, total);
        copyList->Close();
        ID3D12CommandList* lists[] = {copyList.Get()};
        qCopy->ExecuteCommandLists(1, lists);
        const UINT64 vCopy = ++fenceVal;
        qCopy->Signal(fence.Get(), vCopy);

        if (!WaitFence(vConv, 2000)) {
            if (haveKm) km->ReleaseSync(0);
            err = "watchdog: convert fence timeout";
            return false;
        }
        const double t2 = NowMs();
        if (haveKm) km->ReleaseSync(0);  // GPU reads of the shared texture are done at vConv
        if (!WaitFence(vCopy, 2000)) {
            err = "watchdog: copy fence timeout";
            return false;
        }
        const double t3 = NowMs();
        memcpy(cpuBuf.data(), rbPtr, total);  // THE measured copy-out: cached readback heap -> CPU buffer
        const double t4 = NowMs();

        t.schedMs = t1 - t0;
        t.convMs = t2 - t1;
        t.copyQMs = t3 - t2;
        t.copyOutMs = t4 - t3;
        t.totalMs = t4 - t0;
        t.bytes = total;
        return true;
    }

    // Gate (d): wrapped resources released (they are, per frame), fences drained, THEN release the
    // device chain. Returns false if the drain hangs or the device died.
    bool Teardown() {
        bool clean = true;
        if (fence && fenceVal > 0 && !fatal) clean = WaitFence(fenceVal, 2000);
        ctx.outUav.Reset();
        ctx.outBuf.Reset();
        ctx.outBufSize = 0;
        ctx.shader.Reset();
        ctx.paramsCb.Reset();
        if (rbPtr && rbBuf) { rbBuf->Unmap(0, nullptr); rbPtr = nullptr; }
        rbBuf.Reset();
        convBuf12.Reset();
        if (d11ctx) {
            d11ctx->ClearState();
            d11ctx->Flush();
        }
        if (d12 && d12->GetDeviceRemovedReason() != S_OK) { ++deviceRemoved; clean = false; }
        ctx.device.Reset();
        ctx.device1.Reset();
        ctx.context.Reset();
        on12.Reset();
        d11_1.Reset();
        d11ctx.Reset();
        d11.Reset();
        copyList.Reset();
        copyAlloc.Reset();
        fence.Reset();
        qCopy.Reset();
        qDirect.Reset();
        d12.Reset();
        if (evt) { CloseHandle(evt); evt = nullptr; }
        return clean;
    }
};

// ---------------------------------------------------------------------------------------------------
// D3D11 baseline consumer — the LIVE addon path verbatim (ReadbackConsume + ReadbackFinish), for the
// same-machine A/B: WaitGpu spin + WC staging copy through the self-calibrating CopyPool.
// ---------------------------------------------------------------------------------------------------
struct D3D11Consumer {
    std::string key;
    std::vector<uint8_t> cpuBuf;
    uint64_t hrFails = 0;
    bool fatal = false;

    bool ConsumeFrame(uint64_t handle, uint32_t w, uint32_t h, int format, FrameTimings& t, std::string& err) {
        const size_t uyvy = (size_t)w * 2 * h;
        const size_t total = format == 2 ? uyvy + (size_t)w * h : uyvy;
        cpuBuf.resize(total);
        const double t0 = NowMs();
        if (!osrcap::ReadbackConsume((uintptr_t)handle, w, h, format, key, 0, 0, err)) { ++hrFails; return false; }
        const double t1 = NowMs();
        if (!osrcap::ReadbackFinish(key, cpuBuf.data(), total, nullptr, 0, err)) { ++hrFails; return false; }
        const double t2 = NowMs();
        t.schedMs = t1 - t0;   // open + convert + WaitGpu (incl. the busy-spin)
        t.convMs = 0;
        t.copyQMs = 0;
        t.copyOutMs = t2 - t1;  // the WC staging copy-out (CopyPool)
        t.totalMs = t2 - t0;
        t.bytes = total;
        return true;
    }
};

// ---------------------------------------------------------------------------------------------------
// Stream: producer thread + consumer thread + slot handshake, with per-window stats
// ---------------------------------------------------------------------------------------------------
struct Msg {
    int slot;
    uint32_t frame, w, h;
    uint64_t handle;
};

struct Stats {
    std::mutex m;
    // window (reset each report)
    uint64_t wFrames = 0;
    double wSched = 0, wConv = 0, wCopyQ = 0, wCopyOut = 0, wValidate = 0, wTotal = 0, wMax = 0;
    uint64_t wBytes = 0;
    // phase totals
    uint64_t frames = 0, framesTimed = 0, framesValidated = 0, mismatches = 0, staleDetected = 0;
    double sSched = 0, sConv = 0, sCopyQ = 0, sCopyOut = 0, sValidate = 0, sTotal = 0, maxTotal = 0;
    uint64_t sBytes = 0;
    uint64_t producerSkips = 0, prodErrors = 0, consErrors = 0, behinds = 0, drainTimeouts = 0;
    uint64_t resizes = 0, stopstarts = 0;
    double producerCpuSec = 0, consumerCpuSec = 0;
    std::string firstMismatch, firstConsError;
};

struct StreamCfg {
    bool on12 = true;
    bool useMutex = true;
    int format = 1;
    uint32_t w = 3840, h = 2160, altW = 2560, altH = 1440;
    double fps = 60;     // 0 = unthrottled
    double secs = 45;
    bool doResize = true, doStopStart = true, validate = true;
    bool debugLayer = false;
    int warmupFrames = 5;  // excluded from timing means (first-frame shader compile etc.)
};

struct Stream {
    int index = 0;
    StreamCfg cfg;
    const ExpectedCache* expected = nullptr;
    LUID luid = {};

    std::mutex m;
    std::condition_variable cv;
    std::deque<Msg> ready;
    std::deque<int> freeSlots;
    std::atomic<bool> stop{false};
    std::atomic<bool> fatal{false};

    Producer prod;
    std::unique_ptr<On12Consumer> c12;
    std::unique_ptr<D3D11Consumer> c11;
    Stats stats;
    std::thread prodTh, consTh;

    bool Init(std::string& err) {
        if (!prod.Init(err)) return false;
        luid = prod.AdapterLuid();
        if (cfg.on12) {
            c12 = std::make_unique<On12Consumer>();
            if (!c12->Init(luid, cfg.debugLayer, err)) return false;
        } else {
            c11 = std::make_unique<D3D11Consumer>();
            c11->key = Fmt("harness#%d", index);
        }
        if (!prod.CreateSlots(cfg.w, cfg.h, cfg.useMutex, err)) return false;
        for (int i = 0; i < kSlots; ++i) freeSlots.push_back(i);
        return true;
    }

    void Start() {
        prodTh = std::thread([this] { ProducerLoop(); });
        consTh = std::thread([this] { ConsumerLoop(); });
    }

    void Stop() {
        stop = true;
        cv.notify_all();
        if (prodTh.joinable()) prodTh.join();
        if (consTh.joinable()) consTh.join();
    }

    bool DrainSlots(int timeoutMs) {
        std::unique_lock<std::mutex> lk(m);
        return cv.wait_for(lk, std::chrono::milliseconds(timeoutMs),
                           [&] { return stop.load() || (freeSlots.size() == (size_t)kSlots && ready.empty()); });
    }

    void ProducerLoop() {
        const double cpu0 = ThreadCpuSeconds();
        std::string err;
        uint32_t frame = 1;
        const double interval = cfg.fps > 0 ? 1000.0 / cfg.fps : 0;
        const double start = NowMs();
        double next = start;
        bool didResize = false, didStop = false, didResizeBack = false;
        uint32_t curW = cfg.w, curH = cfg.h;

        auto recreate = [&](uint32_t nw, uint32_t nh, bool pause, uint64_t& counter) {
            if (!DrainSlots(5000)) {
                std::lock_guard<std::mutex> g(stats.m);
                ++stats.drainTimeouts;
                return;
            }
            prod.DestroySlots();
            if (pause) std::this_thread::sleep_for(std::chrono::milliseconds(300));
            std::string e2;
            if (!prod.CreateSlots(nw, nh, cfg.useMutex, e2)) {
                std::lock_guard<std::mutex> g(stats.m);
                ++stats.prodErrors;
                if (stats.firstConsError.empty()) stats.firstConsError = "recreate: " + e2;
                stop = true;
                cv.notify_all();
                return;
            }
            curW = nw;
            curH = nh;
            {
                std::lock_guard<std::mutex> lk(m);
                freeSlots.clear();
                for (int i = 0; i < kSlots; ++i) freeSlots.push_back(i);
            }
            cv.notify_all();
            std::lock_guard<std::mutex> g(stats.m);
            ++counter;
        };

        while (!stop) {
            const double el = (NowMs() - start) / 1000.0;
            if (cfg.doResize && !didResize && el >= cfg.secs / 3) {
                didResize = true;
                recreate(cfg.altW, cfg.altH, false, stats.resizes);
                continue;
            }
            if (cfg.doStopStart && !didStop && el >= cfg.secs / 2) {
                didStop = true;
                recreate(curW, curH, true, stats.stopstarts);
                continue;
            }
            if (cfg.doResize && !didResizeBack && el >= 2 * cfg.secs / 3) {
                didResizeBack = true;
                recreate(cfg.w, cfg.h, false, stats.resizes);
                continue;
            }

            int slot = -1;
            if (interval > 0) {
                const double now = NowMs();
                if (now < next) std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(next - now));
                next += interval;
                if (NowMs() > next + interval) {  // fell behind: resync, don't burst
                    next = NowMs() + interval;
                    std::lock_guard<std::mutex> g(stats.m);
                    ++stats.behinds;
                }
                std::lock_guard<std::mutex> lk(m);
                if (!freeSlots.empty()) {
                    slot = freeSlots.front();
                    freeSlots.pop_front();
                } else {
                    std::lock_guard<std::mutex> g(stats.m);
                    ++stats.producerSkips;  // consumer can't keep up with the paced rate
                    continue;
                }
            } else {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [&] { return stop.load() || !freeSlots.empty(); });
                if (stop) break;
                slot = freeSlots.front();
                freeSlots.pop_front();
            }

            if (!prod.Render(slot, frame, err)) {
                std::lock_guard<std::mutex> g(stats.m);
                ++stats.prodErrors;
                if (stats.firstConsError.empty()) stats.firstConsError = "render: " + err;
                std::lock_guard<std::mutex> lk(m);
                freeSlots.push_back(slot);
                continue;
            }
            {
                std::lock_guard<std::mutex> lk(m);
                ready.push_back(Msg{slot, frame, curW, curH, (uint64_t)(uintptr_t)prod.slots[slot].handle});
            }
            cv.notify_all();
            ++frame;
        }
        std::lock_guard<std::mutex> g(stats.m);
        stats.producerCpuSec = ThreadCpuSeconds() - cpu0;
    }

    void ConsumerLoop() {
        const double cpu0 = ThreadCpuSeconds();
        int warmupLeft = cfg.warmupFrames;
        for (;;) {
            Msg msg;
            {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [&] { return stop.load() || !ready.empty(); });
                if (ready.empty()) {
                    if (stop) break;
                    continue;
                }
                msg = ready.front();
                ready.pop_front();
            }

            FrameTimings ft;
            std::string err;
            bool ok = cfg.on12 ? c12->ConsumeFrame(msg.handle, msg.w, msg.h, cfg.format, cfg.useMutex, ft, err)
                               : c11->ConsumeFrame(msg.handle, msg.w, msg.h, cfg.format, ft, err);

            double vMs = 0;
            bool valOk = true;
            int staleBy = 0;
            std::string detail;
            if (ok && cfg.validate) {
                const ExpectedSet* es = expected->Get(msg.w, msg.h);
                const double v0 = NowMs();
                const uint8_t* got = cfg.on12 ? c12->cpuBuf.data() : c11->cpuBuf.data();
                valOk = ValidateFrame(got, *es, msg.frame, detail, staleBy);
                vMs = NowMs() - v0;
            }

            {
                std::lock_guard<std::mutex> g(stats.m);
                if (ok) {
                    ++stats.frames;
                    if (warmupLeft > 0) {
                        --warmupLeft;
                    } else {
                        ++stats.framesTimed;
                        ++stats.wFrames;
                        stats.wSched += ft.schedMs; stats.sSched += ft.schedMs;
                        stats.wConv += ft.convMs; stats.sConv += ft.convMs;
                        stats.wCopyQ += ft.copyQMs; stats.sCopyQ += ft.copyQMs;
                        stats.wCopyOut += ft.copyOutMs; stats.sCopyOut += ft.copyOutMs;
                        stats.wValidate += vMs; stats.sValidate += vMs;
                        stats.wTotal += ft.totalMs; stats.sTotal += ft.totalMs;
                        stats.wBytes += ft.bytes; stats.sBytes += ft.bytes;
                        if (ft.totalMs > stats.wMax) stats.wMax = ft.totalMs;
                        if (ft.totalMs > stats.maxTotal) stats.maxTotal = ft.totalMs;
                    }
                    if (cfg.validate) {
                        ++stats.framesValidated;
                        if (!valOk) {
                            ++stats.mismatches;
                            if (staleBy) ++stats.staleDetected;
                            if (stats.firstMismatch.empty()) stats.firstMismatch = detail;
                        }
                    }
                } else {
                    ++stats.consErrors;
                    if (stats.firstConsError.empty()) stats.firstConsError = err;
                }
            }

            {
                std::lock_guard<std::mutex> lk(m);
                freeSlots.push_back(msg.slot);
            }
            cv.notify_all();

            if ((c12 && c12->fatal) || (c11 && c11->fatal)) {
                fatal = true;
                stop = true;
                cv.notify_all();
                break;
            }
        }
        std::lock_guard<std::mutex> g(stats.m);
        stats.consumerCpuSec = ThreadCpuSeconds() - cpu0;
    }
};

// ---------------------------------------------------------------------------------------------------
// Phase runner + gates
// ---------------------------------------------------------------------------------------------------
struct Gates {
    uint64_t mismatchFrames = 0, framesValidated = 0, staleDetected = 0, resizes = 0, stopstarts = 0;
    uint64_t deviceRemoved = 0, watchdog = 0, acquireFails = 0, hrFails = 0, consErrors = 0, prodErrors = 0;
    uint64_t drainTimeouts = 0, configMismatch = 0, fatals = 0;
    double on12CopyOutMs = 0, on12SchedMs = 0, on12ConvMs = 0, on12CopyQMs = 0, on12TotalMs = 0;
    uint64_t on12Frames = 0, on12Bytes = 0;
    double d11ConsumeMs = 0, d11CopyOutMs = 0, d11TotalMs = 0;
    uint64_t d11Frames = 0, d11Bytes = 0;
    struct RateCell { double fpsPerStream = 0, consumerCores = 0, validateCores = 0; bool have = false; };
    RateCell rate[2][3];  // [on12=0/d3d11=1][streams-1]
    bool teardownClean = true;
    bool ranOn12 = false, ranD3d11 = false;
    std::vector<std::string> failNotes;
};

struct PhaseCfg {
    bool on12;
    bool useMutex;
    int streams;
    double fps;   // 0 = unthrottled
    double secs;
    bool resize, stopstart;
    const char* label;
};

struct Options {
    uint32_t w = 3840, h = 2160, altW = 2560, altH = 1440;
    int format = 1;
    int streams = 3;
    double fps = 60, secs = 45, rateSecs = 10, report = 5;
    bool resize = true, stopstart = true, validate = true;
    bool runOn12 = true, runD3d11 = true;
    bool mutexOn = true, mutexOff = true;
    bool skipSoak = false, skipRate = false;
    bool debugLayer = false;
    // Sub-step B: which backend the LIVE addon path (the "d3d11" variant = ReadbackConsume/ReadbackFinish
    // verbatim) runs on. false (default) pins FS_READBACK=d3d11 so that arm stays the classic WC baseline;
    // --live=on12 leaves the §5 ladder free, so the live arm exercises the folded-in 11On12 consume path
    // (EnsureOn12OutBuffer, wrapped DispatchConvert, copy-queue DMA, cached copy-out) under the FULL gate
    // matrix — resize, stop/start, mutex variants, byte-exact GATE-A validation.
    bool liveOn12 = false;
};

static bool RunPhase(const PhaseCfg& pc, const Options& opt, const ExpectedCache& cache, Gates& g) {
    printf("\n[PHASE] %s variant=%s mutex=%d streams=%d fps=%g secs=%g resize=%d stopstart=%d\n",
           pc.label, pc.on12 ? "on12" : "d3d11", pc.useMutex ? 1 : 0, pc.streams, pc.fps, pc.secs,
           pc.resize ? 1 : 0, pc.stopstart ? 1 : 0);
    fflush(stdout);

    std::vector<std::unique_ptr<Stream>> streams;
    for (int i = 0; i < pc.streams; ++i) {
        auto s = std::make_unique<Stream>();
        s->index = i;
        s->cfg.on12 = pc.on12;
        s->cfg.useMutex = pc.useMutex;
        s->cfg.format = opt.format;
        s->cfg.w = opt.w; s->cfg.h = opt.h; s->cfg.altW = opt.altW; s->cfg.altH = opt.altH;
        s->cfg.fps = pc.fps;
        s->cfg.secs = pc.secs;
        s->cfg.doResize = pc.resize;
        s->cfg.doStopStart = pc.stopstart;
        s->cfg.validate = opt.validate;
        s->cfg.debugLayer = opt.debugLayer;
        s->expected = &cache;
        std::string err;
        if (!s->Init(err)) {
            printf("[FAIL] stream %d init: %s\n", i, err.c_str());
            g.failNotes.push_back(Fmt("%s s%d init: %s", pc.label, i, err.c_str()));
            for (auto& st : streams) st->Stop();
            return false;
        }
        streams.push_back(std::move(s));
    }

    const double procCpu0 = ProcessCpuSeconds();
    const double t0 = NowMs();
    for (auto& s : streams) s->Start();

    // reporter
    double lastReport = t0;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const double now = NowMs();
        bool anyFatal = false;
        for (auto& s : streams) anyFatal = anyFatal || s->fatal.load();
        const bool timeUp = (now - t0) / 1000.0 >= pc.secs;
        if (now - lastReport >= opt.report * 1000.0 || timeUp || anyFatal) {
            lastReport = now;
            for (auto& s : streams) {
                std::lock_guard<std::mutex> lk(s->stats.m);
                Stats& st = s->stats;
                const double dt = opt.report;
                const uint64_t n = st.wFrames ? st.wFrames : 1;
                if (pc.on12)
                    printf("[W] on12 m%d s%d t=%.0fs fps=%.1f sched=%.2f convGpu=%.2f copyQ=%.2f copyOut=%.2f(%.0fMB/s) val=%.2f tot=%.2f max=%.1f mism=%" PRIu64 " skip=%" PRIu64 "\n",
                           pc.useMutex ? 1 : 0, s->index, (now - t0) / 1000.0, st.wFrames / dt,
                           st.wSched / n, st.wConv / n, st.wCopyQ / n, st.wCopyOut / n,
                           st.wCopyOut > 0 ? st.wBytes / (st.wCopyOut / 1000.0) / 1e6 : 0,
                           st.wValidate / n, st.wTotal / n, st.wMax, st.mismatches, st.producerSkips);
                else
                    printf("[W] d3d11 m%d s%d t=%.0fs fps=%.1f consume=%.2f copyOutWC=%.2f(%.0fMB/s) val=%.2f tot=%.2f max=%.1f mism=%" PRIu64 " skip=%" PRIu64 "\n",
                           pc.useMutex ? 1 : 0, s->index, (now - t0) / 1000.0, st.wFrames / dt,
                           st.wSched / n, st.wCopyOut / n,
                           st.wCopyOut > 0 ? st.wBytes / (st.wCopyOut / 1000.0) / 1e6 : 0,
                           st.wValidate / n, st.wTotal / n, st.wMax, st.mismatches, st.producerSkips);
                st.wFrames = 0;
                st.wSched = st.wConv = st.wCopyQ = st.wCopyOut = st.wValidate = st.wTotal = st.wMax = 0;
                st.wBytes = 0;
            }
            fflush(stdout);
        }
        if (timeUp || anyFatal) break;
    }

    for (auto& s : streams) s->Stop();
    const double wallSec = (NowMs() - t0) / 1000.0;
    const double procCpu = ProcessCpuSeconds() - procCpu0;

    // aggregate
    double sumFps = 0, consCpu = 0, prodCpu = 0, valCpuSec = 0;
    bool phaseOk = true;
    for (auto& s : streams) {
        Stats& st = s->stats;  // threads joined — no lock needed
        sumFps += st.frames / wallSec;
        consCpu += st.consumerCpuSec;
        prodCpu += st.producerCpuSec;
        valCpuSec += st.sValidate / 1000.0;
        g.mismatchFrames += st.mismatches;
        g.framesValidated += st.framesValidated;
        g.staleDetected += st.staleDetected;
        g.resizes += st.resizes;
        g.stopstarts += st.stopstarts;
        g.consErrors += st.consErrors;
        g.prodErrors += st.prodErrors;
        g.drainTimeouts += st.drainTimeouts;
        if (st.mismatches && !st.firstMismatch.empty())
            g.failNotes.push_back(Fmt("%s %s m%d s%d MISMATCH: %s", pc.label, pc.on12 ? "on12" : "d3d11",
                                      pc.useMutex ? 1 : 0, s->index, st.firstMismatch.c_str()));
        if ((st.consErrors || st.prodErrors) && !st.firstConsError.empty())
            g.failNotes.push_back(Fmt("%s %s m%d s%d ERROR: %s", pc.label, pc.on12 ? "on12" : "d3d11",
                                      pc.useMutex ? 1 : 0, s->index, st.firstConsError.c_str()));
        if (s->fatal) { ++g.fatals; phaseOk = false; }
        if (pc.on12) {
            On12Consumer* c = s->c12.get();
            g.watchdog += c->watchdogTimeouts;
            g.deviceRemoved += c->deviceRemoved;
            g.acquireFails += c->acquireFails;
            g.hrFails += c->hrFails;
            g.configMismatch += c->configMismatch;
            g.on12SchedMs += st.sSched; g.on12ConvMs += st.sConv; g.on12CopyQMs += st.sCopyQ;
            g.on12CopyOutMs += st.sCopyOut; g.on12TotalMs += st.sTotal;
            g.on12Frames += st.framesTimed; g.on12Bytes += st.sBytes;
            g.ranOn12 = true;
            if (!c->Teardown()) { g.teardownClean = false; phaseOk = false; }
        } else {
            g.hrFails += s->c11->hrFails;
            g.d11ConsumeMs += st.sSched; g.d11CopyOutMs += st.sCopyOut; g.d11TotalMs += st.sTotal;
            g.d11Frames += st.framesTimed; g.d11Bytes += st.sBytes;
            g.ranD3d11 = true;
        }
        s->prod.DestroySlots();
    }

    printf("[PHASE-END] %s %s m%d streams=%d wall=%.1fs fps/stream=%.1f consumerCPU=%.2f cores (validate~%.2f) producerCPU=%.2f processCPU=%.2f\n",
           pc.label, pc.on12 ? "on12" : "d3d11", pc.useMutex ? 1 : 0, pc.streams, wallSec,
           sumFps / pc.streams, consCpu / wallSec, valCpuSec / wallSec, prodCpu / wallSec, procCpu / wallSec);
    fflush(stdout);

    if (pc.fps == 0) {  // rate phase -> gate C2 cell
        auto& cell = g.rate[pc.on12 ? 0 : 1][pc.streams - 1];
        cell.fpsPerStream = sumFps / pc.streams;
        cell.consumerCores = consCpu / wallSec;
        cell.validateCores = valCpuSec / wallSec;
        cell.have = true;
    }
    return phaseOk;
}

}  // namespace harness

// ---------------------------------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------------------------------
int main(int argc, char** argv) {
    using namespace harness;
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&](const char* k) -> const char* {
            const size_t n = strlen(k);
            return a.compare(0, n, k) == 0 && a[n] == '=' ? a.c_str() + n + 1 : nullptr;
        };
        const char* v;
        if ((v = val("--width"))) opt.w = (uint32_t)atoi(v) & ~3u;
        else if ((v = val("--height"))) opt.h = (uint32_t)atoi(v);
        else if ((v = val("--alt-width"))) opt.altW = (uint32_t)atoi(v) & ~3u;
        else if ((v = val("--alt-height"))) opt.altH = (uint32_t)atoi(v);
        else if ((v = val("--format"))) opt.format = atoi(v) == 2 ? 2 : 1;
        else if ((v = val("--streams"))) opt.streams = atoi(v) < 1 ? 1 : (atoi(v) > 8 ? 8 : atoi(v));
        else if ((v = val("--fps"))) opt.fps = atof(v);
        else if ((v = val("--secs"))) opt.secs = atof(v);
        else if ((v = val("--rate-secs"))) opt.rateSecs = atof(v);
        else if ((v = val("--report"))) opt.report = atof(v);
        else if ((v = val("--variant"))) { opt.runOn12 = strcmp(v, "d3d11") != 0; opt.runD3d11 = strcmp(v, "on12") != 0; }
        else if ((v = val("--mutex"))) { opt.mutexOn = strcmp(v, "0") != 0; opt.mutexOff = strcmp(v, "1") != 0; }
        else if (a == "--no-resize") opt.resize = false;
        else if (a == "--no-stopstart") opt.stopstart = false;
        else if (a == "--no-validate") opt.validate = false;
        else if (a == "--skip-soak") opt.skipSoak = true;
        else if (a == "--skip-rate") opt.skipRate = true;
        else if (a == "--quick") { opt.secs = 8; opt.rateSecs = 4; opt.report = 2; }
        else if (a == "--debug-layer") opt.debugLayer = true;
        else if ((v = val("--live"))) opt.liveOn12 = strcmp(v, "on12") == 0;
        else { printf("unknown arg: %s (see file header for options)\n", a.c_str()); return 2; }
    }

    // Sub-step B: readback_win.cc's live EnsureDevice now prefers the 11On12 backend itself (§5 ladder). By
    // default pin the live path to the classic backend so the "d3d11" arm stays the WC-staging BASELINE for
    // the A/B (same binary diagnostic selector FreeShow honors). With --live=on12 the pin is skipped and the
    // "d3d11"-labelled arm becomes the live LADDER path — run `--variant=d3d11 --live=on12` to soak-validate
    // the folded-in 11On12 consume (the [READBACK] backend= stderr line says which engaged). The injected
    // on12 arm never goes through EnsureDevice and is unaffected either way.
    if (!opt.liveOn12) _putenv_s("FS_READBACK", "d3d11");

    timeBeginPeriod(1);
    printf("[CFG] %ux%u alt=%ux%u fmt=%d streams=%d fps=%g soakSecs=%g rateSecs=%g report=%gs validate=%d variants=%s%s mutex=%s%s liveBackend=%s\n",
           opt.w, opt.h, opt.altW, opt.altH, opt.format, opt.streams, opt.fps, opt.secs, opt.rateSecs,
           opt.report, opt.validate ? 1 : 0, opt.runOn12 ? "on12 " : "", opt.runD3d11 ? "d3d11" : "",
           opt.mutexOn ? "1 " : "", opt.mutexOff ? "0" : "", opt.liveOn12 ? "on12-ladder" : "d3d11-pinned");
    fflush(stdout);

    // Oracle probe (see Producer::SelfProbe). Without a byte-exact round trip gate (a) has no oracle.
    {
        Producer p;
        std::string err;
        if (!p.Init(err) || !p.CreateSlots(64, 64, true, err) || !p.SelfProbe(err)) {
            printf("[PROBE] FAILED: %s\n[RESULT] ABORT — cannot validate on this GPU/driver (finding: report this line)\n", err.c_str());
            return 2;
        }
        p.DestroySlots();
        printf("[PROBE] producer float->unorm round-trip byte-exact: PASS\n");
        fflush(stdout);
    }

    ExpectedCache cache;
    if (opt.validate) {
        cache.Build(opt.w, opt.h, opt.format);
        if (opt.resize) cache.Build(opt.altW, opt.altH, opt.format);
    }

    std::vector<PhaseCfg> phases;
    if (!opt.skipSoak) {
        for (int variant = 0; variant < 2; ++variant) {
            const bool on12 = variant == 0;
            if (on12 && !opt.runOn12) continue;
            if (!on12 && !opt.runD3d11) continue;
            for (int mu = 1; mu >= 0; --mu) {
                if (mu && !opt.mutexOn) continue;
                if (!mu && !opt.mutexOff) continue;
                phases.push_back({on12, mu == 1, opt.streams, opt.fps, opt.secs, opt.resize, opt.stopstart, "soak"});
            }
        }
    }
    if (!opt.skipRate) {
        const bool rateMutex = opt.mutexOn;  // rate phases use the Electron-realistic keyed-mutex mode
        for (int variant = 0; variant < 2; ++variant) {
            const bool on12 = variant == 0;
            if (on12 && !opt.runOn12) continue;
            if (!on12 && !opt.runD3d11) continue;
            for (int n = 1; n <= 3; ++n)
                phases.push_back({on12, rateMutex, n, 0, opt.rateSecs, false, false, "rate"});
        }
    }

    Gates g;
    bool allOk = true;
    for (auto& pc : phases) allOk = RunPhase(pc, opt, cache, g) && allOk;

    // ---- pasteable gate summary ----
    printf("\n==== GATE SUMMARY (paste this whole block) ====\n");
    printf("[GATE-A] mismatchFrames=%" PRIu64 " framesValidated=%" PRIu64 " staleDetected=%" PRIu64
           " resizes=%" PRIu64 " stopStarts=%" PRIu64 "  -> %s (criterion: mismatchFrames == 0)\n",
           g.mismatchFrames, g.framesValidated, g.staleDetected, g.resizes, g.stopstarts,
           g.mismatchFrames == 0 && g.framesValidated > 0 ? "PASS" : "FAIL");
    const bool gateB = g.deviceRemoved == 0 && g.watchdog == 0 && g.acquireFails == 0 && g.consErrors == 0 &&
                       g.prodErrors == 0 && g.hrFails == 0 && g.drainTimeouts == 0 && g.configMismatch == 0 &&
                       g.fatals == 0;
    printf("[GATE-B] deviceRemoved=%" PRIu64 " watchdogTimeouts=%" PRIu64 " acquireFails=%" PRIu64
           " openFails=%" PRIu64 " consumeErrors=%" PRIu64 " producerErrors=%" PRIu64 " drainTimeouts=%" PRIu64
           " configMismatch=%" PRIu64 " fatals=%" PRIu64 "  -> %s (criterion: all 0)\n",
           g.deviceRemoved, g.watchdog, g.acquireFails, g.hrFails, g.consErrors, g.prodErrors,
           g.drainTimeouts, g.configMismatch, g.fatals, gateB ? "PASS" : "FAIL");
    if (g.on12Frames > 0)
        printf("[GATE-C] on12 (n=%" PRIu64 "): open+convert(sched)=%.2fms convGpu=%.2fms copyQueue=%.2fms copyOut=%.2fms (%.0f MB/s) total=%.2fms\n",
               g.on12Frames, g.on12SchedMs / g.on12Frames, g.on12ConvMs / g.on12Frames,
               g.on12CopyQMs / g.on12Frames, g.on12CopyOutMs / g.on12Frames,
               g.on12CopyOutMs > 0 ? g.on12Bytes / (g.on12CopyOutMs / 1000.0) / 1e6 : 0,
               g.on12TotalMs / g.on12Frames);
    if (g.d11Frames > 0)
        printf("[GATE-C] d3d11 (n=%" PRIu64 "): open+convert+WaitGpu(consume)=%.2fms copyOutWC=%.2fms (%.0f MB/s) total=%.2fms\n",
               g.d11Frames, g.d11ConsumeMs / g.d11Frames, g.d11CopyOutMs / g.d11Frames,
               g.d11CopyOutMs > 0 ? g.d11Bytes / (g.d11CopyOutMs / 1000.0) / 1e6 : 0,
               g.d11TotalMs / g.d11Frames);
    if (g.on12Frames > 0 && g.d11Frames > 0)
        printf("[GATE-C] copy-out WC->cached ratio: %.1fx (hypothesis: cached-memcpy class; measured, no target)\n",
               (g.d11CopyOutMs / g.d11Frames) / ((g.on12CopyOutMs / g.on12Frames) > 0 ? g.on12CopyOutMs / g.on12Frames : 1));
    for (int variant = 0; variant < 2; ++variant) {
        const char* name = variant == 0 ? "on12" : "d3d11";
        if (!g.rate[variant][0].have && !g.rate[variant][1].have && !g.rate[variant][2].have) continue;
        printf("[GATE-C2] %s isolated consume rate (unthrottled, validate on):", name);
        for (int n = 0; n < 3; ++n)
            if (g.rate[variant][n].have)
                printf(" %dstream=%.1ffps/stream (consCPU=%.2f cores, validate~%.2f)", n + 1,
                       g.rate[variant][n].fpsPerStream, g.rate[variant][n].consumerCores,
                       g.rate[variant][n].validateCores);
        printf("\n");
    }
    printf("[GATE-D] teardown clean=%s (wrapped released before final Flush, fences drained, devices freed, NT handles closed)\n",
           g.teardownClean ? "YES" : "NO");
    for (auto& n : g.failNotes) printf("[NOTE] %s\n", n.c_str());

    const bool pass = allOk && g.mismatchFrames == 0 && g.framesValidated > 0 && gateB && g.teardownClean;
    printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
    fflush(stdout);
    timeEndPeriod(1);
    return pass ? 0 : 1;
}
