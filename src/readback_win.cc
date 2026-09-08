// Windows backend: open Electron's shared D3D11 texture, optionally convert BGRA -> UYVY/UYVA on the GPU
// with a compute shader, and read the result back to a CPU buffer.
//
// format: 0 = BGRA (raw copy), 1 = UYVY (opaque, width*2*height), 2 = UYVA (UYVY plane + width*height alpha).
// For UYVY/UYVA the conversion runs on the GPU (no CPU colour-conversion traffic) and the smaller converted
// buffer is what gets read back. Each pool entry owns an independent device/context so outputs run in
// parallel; the immediate context is only ever used by one readback at a time.
#include "osr_readback.h"

#include <windows.h>
#include <d3d11_1.h>
#include <d3d11on12.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

#include <smmintrin.h>  // SSE4.1: _mm_stream_load_si128 (MOVNTDQA) for fast reads of WC GPU-mapped memory

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {
// Copy FROM GPU-mapped memory with MOVNTDQA streaming loads. `src` must be 16-byte aligned.
inline void StreamCopyFromWC(void* dst, const void* src, size_t bytes) {
    auto* s = static_cast<const __m128i*>(src);
    auto* d = static_cast<__m128i*>(dst);
    size_t blocks = bytes / 16;
    for (size_t i = 0; i < blocks; ++i) {
        __m128i v = _mm_stream_load_si128(s + i);
        _mm_storeu_si128(d + i, v);
    }
    _mm_sfence();
    size_t done = blocks * 16;
    if (done < bytes) std::memcpy(static_cast<char*>(dst) + done, static_cast<const char*>(src) + done, bytes - done);
}

// GPU-mapped readback staging is LATENCY-bound (BAR/VRAM over PCIe, ~400 MB/s single-threaded), so a 4K frame
// is split across threads to overlap the stalls. A persistent, process-global pool of workers pulls fixed-SIZE
// chunks from a shared FIFO queue. This CAPS total concurrent copy threads regardless of how many readbacks
// are in flight — previously each readback spawned its own 8 threads (~48 under load) that oversubscribed the
// CPU and made readbacks finish in CLUSTERS (the delivery convoy). The shared queue also interleaves chunks
// from concurrent frames, spreading their completions. The pool is process-lifetime (intentionally leaked — no
// static-dtor join/terminate, OS reclaims at exit) and copies only into C++ vectors (never V8 buffers), so
// there is no per-env teardown/lifetime hazard.
//
// SIZING (plan §11 CV#1): the worker count is MEASURED at runtime, never guessed. hardware_concurrency() is
// used ONLY as a STRUCTURAL ceiling on how many workers are PARKED; the operating point `active_n` (how many
// may dequeue) self-calibrates to the throughput KNEE of the real chunk stream: WC/BAR copies saturate the
// link at some worker count, past which extra workers add zero copy throughput and pure CPU burn (stolen from
// the SpeedHQ send). Every batch (one Run() call) yields a bytes/wall-time sample; a duty-cycled A/B collects
// equal sample counts from `active_n` and `active_n ± 1` in interleaved sub-blocks (so load drift cancels)
// and moves `active_n` by the SIGN of the median difference — no percent thresholds, no tuned margins; the
// only hysteresis is integer replication on the climb (see Decide). The duty cycle IS the steady state (the
// pool re-probes both neighbours forever, tracking knee movement with GPU load/thermals); "settled" means the
// BASE stops drifting — a ±1 dither during probe windows is by construction, not oscillation. Workers run at
// THREAD_PRIORITY_BELOW_NORMAL [STRUCTURAL — an OS mechanism, not a value]: WC copies are memory-stall-bound
// so idle-core cost is ~nil, and under contention the OS scheduler (itself a measuring, adapting system)
// awards the CPU to the compress.
class CopyPool {
public:
    static CopyPool& Get() {
        static CopyPool* inst = new CopyPool();  // leaked on purpose: process-lifetime pool
        return *inst;
    }

    // Split [src, bytes) into chunks, run them across the pool, and BLOCK until all complete (so src/dst stay
    // valid for the whole copy). `remaining` is set before any worker can observe a job, so it's race-free.
    // Each completed batch feeds one bytes/wall-time sample to the calibrator (the CV#1 MEASURED signal).
    void Run(void* dst, const void* src, size_t bytes) {
        if (parked_ == 0 || bytes < kInlineFloor) {
            StreamCopyFromWC(dst, src, bytes);
            return;
        }
        const size_t nchunks = (bytes + kChunk - 1) / kChunk;
        auto batch = std::make_shared<Batch>();
        batch->remaining = static_cast<int>(nchunks);
        const uint64_t epoch = epoch_.load();  // tag the batch with the calibration arm it starts under
        const auto t0 = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (size_t off = 0; off < bytes; off += kChunk) {
                const size_t len = (bytes - off < kChunk) ? (bytes - off) : kChunk;
                queue_.push_back(Job{static_cast<char*>(dst) + off, static_cast<const char*>(src) + off, len, batch});
            }
        }
        cv_.notify_all();
        {
            std::unique_lock<std::mutex> lk(batch->mu);
            batch->done.wait(lk, [&] { return batch->remaining == 0; });
        }
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (sec > 0) OnBatchSample(static_cast<double>(bytes) / sec, epoch);
    }

    // Telemetry ("observability, not knobs"): the calibrated BASE operating point. During probe windows the
    // instantaneous eligible count dithers ±1 around this by construction; the base is the settled value.
    unsigned ActiveWorkers() const { return base_.load(); }

private:
    static constexpr size_t kChunk = 1u << 20;        // 1 MiB — a fixed SIZE (universal plateau), NOT a thread count
    static constexpr size_t kInlineFloor = 1u << 16;  // tiny copies run inline on the caller
    // Samples per A/B arm [UNIVERSAL — a statistics property, identical on every machine]: enough for a stable
    // median (odd, so the median is a real sample), few enough that a full A/B cycle spans about a second of
    // 4K video (~120 batches/s), keeping knee-tracking at window granularity.
    static constexpr size_t kArmSamples = 15;
    // Arm alternation block [UNIVERSAL — paired-comparison structure, not a tuned value]: the two arms are
    // measured in interleaved sub-blocks (A,B,A,B,...) rather than one long window each, so slow drift in the
    // offered load (concurrent outputs starting/stopping, GPU load) biases BOTH arms equally and cancels in
    // the median comparison. Any block giving >=2 blocks per arm works (plateau); kArmSamples/3 gives 3.
    static constexpr size_t kBlock = kArmSamples / 3;

    struct Batch {
        int remaining = 0;  // guarded by mu
        std::mutex mu;
        std::condition_variable done;
    };
    struct Job {
        char* dst;
        const char* src;
        size_t len;
        std::shared_ptr<Batch> batch;
    };

    CopyPool() {
        const unsigned hc = std::thread::hardware_concurrency();
        parked_ = hc > 0 ? hc : 1;  // STRUCTURAL ceiling: no more CPU-bound threads than logical cores can help
        logStats_ = std::getenv("FS_CAP_STATS") != nullptr;  // telemetry gate (verification), not a tuning knob
        lastLog_ = std::chrono::steady_clock::now();
        // Park `parked_` workers; only the first `active_` are eligible to dequeue. Start at 1 and climb.
        for (unsigned i = 0; i < parked_; ++i) std::thread([this, i] { WorkerLoop(i); }).detach();
    }

    void WorkerLoop(unsigned index) {
        // BELOW_NORMAL delegates the instantaneous copy-vs-compress CPU split to the OS scheduler: on idle
        // cores a stall-bound copy thread loses ~nothing, under contention the compress wins the core.
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return !queue_.empty() && index < active_.load(); });
                job = std::move(queue_.front());
                queue_.pop_front();
            }
            StreamCopyFromWC(job.dst, job.src, job.len);
            std::lock_guard<std::mutex> lk(job.batch->mu);
            if (--job.batch->remaining == 0) job.batch->done.notify_one();
        }
    }

    // Calibrator: collect kArmSamples per arm — base and base±1 in INTERLEAVED kBlock sub-blocks (paired
    // comparison, see kBlock) — then move the base by the sign of the median difference. Samples whose batch
    // spanned an arm switch (epoch mismatch) belong to neither arm and are discarded — equal-count windows
    // must be clean.
    void OnBatchSample(double bytesPerSec, uint64_t epoch) {
        std::lock_guard<std::mutex> lk(calibMu_);
        if (parked_ < 2) return;  // one core: nothing to calibrate, active_n stays 1
        if (epoch != epoch_.load()) return;
        std::vector<double>& arm = onProbeArm_ ? probeSamples_ : baseSamples_;
        if (arm.size() < kArmSamples) {
            arm.push_back(bytesPerSec);
            ++blockFill_;
        }
        if (baseSamples_.size() >= kArmSamples && probeSamples_.size() >= kArmSamples) {
            Decide();
            return;
        }
        if (blockFill_ >= kBlock || arm.size() >= kArmSamples) {
            blockFill_ = 0;
            std::vector<double>& other = onProbeArm_ ? baseSamples_ : probeSamples_;
            if (other.size() < kArmSamples) {  // else keep filling this arm — the other is already complete
                onProbeArm_ = !onProbeArm_;
                const unsigned b = base_.load();
                SetActive(onProbeArm_ ? static_cast<unsigned>(static_cast<int>(b) + probeDir_) : b);
            }
        }
    }

    void Decide() {  // calibMu_ held; both arms have kArmSamples samples
        const double mBase = Median(baseSamples_);
        const double mProbe = Median(probeSamples_);
        const unsigned oldBase = base_.load();
        unsigned newBase = oldBase;
        // Sign-of-median-difference decisions (universal rules, no tuned margins):
        //  up-probe:  a climb requires the strict gain to REPLICATE across 2 consecutive up-probe cycles —
        //             the same integer-replication hysteresis CV#2 uses, applied because on the plateau past
        //             the knee the sign of a single A/B is pure noise (an unbiased random walk that never
        //             settles). Adding CPU burn bears the burden of replicated proof.
        //  down-probe: drop the top worker whenever removing it did NOT reduce median throughput — single
        //             cycle, because the objective is max throughput for MINIMUM CPU: below the knee the
        //             drop reliably loses the A/B (real throughput loss), so a noise-drop at the knee is
        //             rare and the next replicated up-probe gain undoes it.
        if (probeDir_ > 0) {
            if (mProbe > mBase) {
                if (++upWins_ >= 2) {
                    newBase = oldBase + 1;
                    upWins_ = 0;
                }
            } else {
                upWins_ = 0;
            }
        } else {
            if (mProbe >= mBase) {
                newBase = oldBase - 1;
                upWins_ = 0;  // the top worker was surplus — stale half-climb evidence dies with it
            }
        }
        if (logStats_) {
            const auto now = std::chrono::steady_clock::now();
            if (now - lastLog_ >= std::chrono::seconds(1)) {  // rate-limit: decisions run ~4/s at 4K60
                lastLog_ = now;
                fprintf(stderr, "[COPY-POOL] active_n=%u (was %u; probe %+d: base %.0f vs probe %.0f MB/s -> %s) parked=%u\n",
                        newBase, oldBase, probeDir_, mBase / 1e6, mProbe / 1e6,
                        newBase > oldBase ? "climb" : (newBase < oldBase ? "drop" : "hold"), parked_);
                fflush(stderr);
            }
        }
        base_.store(newBase);
        baseSamples_.clear();
        probeSamples_.clear();
        onProbeArm_ = false;
        blockFill_ = 0;
        // Duty cycle: alternate the probe direction so BOTH neighbours are re-tested forever (knee tracking).
        // At the structural bounds only the valid direction is probed.
        probeDir_ = -probeDir_;
        if (probeDir_ > 0 && newBase >= parked_) probeDir_ = -1;
        if (probeDir_ < 0 && newBase <= 1) probeDir_ = +1;
        SetActive(newBase);
    }

    void SetActive(unsigned n) {  // calibMu_ held (single writer); lock order calibMu_ -> mu_ (never reversed)
        {
            std::lock_guard<std::mutex> lk(mu_);  // publish under the queue lock so no parked waiter misses it
            active_.store(n);
        }
        epoch_.fetch_add(1);  // in-flight batches tagged with the old epoch are discarded on completion
        cv_.notify_all();
    }

    static double Median(const std::vector<double>& v) {
        std::vector<double> tmp(v);
        const size_t mid = tmp.size() / 2;
        std::nth_element(tmp.begin(), tmp.begin() + mid, tmp.end());
        return tmp[mid];
    }

    unsigned parked_ = 0;               // STRUCTURAL: spawned workers (= hardware_concurrency ceiling)
    std::atomic<unsigned> active_{1};   // instantaneous eligible-to-dequeue count (dithers during probes)
    std::atomic<unsigned> base_{1};     // calibrated operating point (the observable "settled" value)
    std::atomic<uint64_t> epoch_{1};    // bumped on every arm switch; stale-tagged samples are discarded
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Job> queue_;

    // calibrator state (guarded by calibMu_)
    std::mutex calibMu_;
    std::vector<double> baseSamples_, probeSamples_;
    bool onProbeArm_ = false;  // which arm the current sub-block measures
    size_t blockFill_ = 0;     // samples accepted into the current sub-block
    int probeDir_ = +1;        // start at active_n=1 and climb
    int upWins_ = 0;           // consecutive up-probe wins (climb replication hysteresis)
    bool logStats_ = false;
    std::chrono::steady_clock::time_point lastLog_;
};

// Route the parallel WC copy through the global pool. `threads` is now IGNORED (the pool self-sizes to the
// machine); the parameter is kept so the call sites are unchanged.
inline void ParallelStreamCopyFromWC(void* dst, const void* src, size_t bytes, unsigned /*threads*/) {
    CopyPool::Get().Run(dst, src, bytes);
}
}  // namespace

using Microsoft::WRL::ComPtr;

namespace osrcap {

namespace {

// BGRA -> UYVY (+ optional alpha plane). Each thread handles 4 horizontal pixels so every write to the raw
// output buffer is 4-byte aligned (2 UYVY words + 1 packed-alpha word). BT.601 full range, integer.
const char* kConvertHLSL = R"HLSL(
Texture2D<float4> gSrc : register(t0);
RWByteAddressBuffer gDst : register(u0);
cbuffer Params : register(b0) {
    uint gWidth;
    uint gHeight;
    uint gUyvySize;
    uint gWriteAlpha;
    uint gVideoOn;
    uint gVideoW;
    uint gVideoH;
    uint gVideoFormat;
};

// The video layer sits under the page: where the page is transparent, its pixels show through. UYVY is
// stored as a half-width RGBA texture (U Y0 V Y1 per texel) and decoded with the same limited-range
// matrices the renderer uses, so a composited frame matches one drawn in the page.
Texture2D<float4> gVideo : register(t1);
float3 videoAt(uint x, uint y, uint dstW, uint dstH) {
    uint vx = (dstW == gVideoW) ? x : x * gVideoW / dstW;
    uint vy = (dstH == gVideoH) ? y : y * gVideoH / dstH;
    if (gVideoFormat != 1) return gVideo.Load(int3(vx, vy, 0)).rgb;
    float4 s = gVideo.Load(int3(vx / 2, vy, 0));
    float luma = ((vx & 1) == 0) ? s.g : s.a;
    float yy = (luma - 16.0 / 255.0) * (255.0 / 219.0);
    float u = (s.r - 128.0 / 255.0) * (255.0 / 224.0);
    float v = (s.b - 128.0 / 255.0) * (255.0 / 224.0);
    float bt = (gVideoH >= 720) ? 1.0 : 0.0;
    float kr = lerp(1.402, 1.5748, bt), kb = lerp(1.772, 1.8556, bt);
    float gu = lerp(0.344136, 0.1873, bt), gv = lerp(0.714136, 0.4681, bt);
    return saturate(float3(yy + kr * v, yy - gu * u - gv * v, yy + kb * u));
}
// page over video, premultiplied: opaque page pixels are untouched
float4 overVideo(float4 page, uint x, uint y, uint dstW, uint dstH) {
    if (gVideoOn == 0) return page;
    if (page.a >= 0.999) return page;
    return float4(page.rgb + videoAt(x, y, dstW, dstH) * (1.0 - page.a), 1.0);
}
int toByte(float v) { return (int)(v * 255.0 + 0.5); }
[numthreads(16, 16, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    uint px = tid.x * 4;
    uint y = tid.y;
    if (px >= gWidth || y >= gHeight) return;
    float4 c0 = overVideo(gSrc.Load(int3(px + 0, y, 0)), px + 0, y, gWidth, gHeight);
    float4 c1 = overVideo(gSrc.Load(int3(px + 1, y, 0)), px + 1, y, gWidth, gHeight);
    float4 c2 = overVideo(gSrc.Load(int3(px + 2, y, 0)), px + 2, y, gWidth, gHeight);
    float4 c3 = overVideo(gSrc.Load(int3(px + 3, y, 0)), px + 3, y, gWidth, gHeight);
    int r0=toByte(c0.r),g0=toByte(c0.g),b0=toByte(c0.b);
    int r1=toByte(c1.r),g1=toByte(c1.g),b1=toByte(c1.b);
    int r2=toByte(c2.r),g2=toByte(c2.g),b2=toByte(c2.b);
    int r3=toByte(c3.r),g3=toByte(c3.g),b3=toByte(c3.b);
    int y0=(77*r0+150*g0+29*b0)>>8;
    int y1=(77*r1+150*g1+29*b1)>>8;
    int y2=(77*r2+150*g2+29*b2)>>8;
    int y3=(77*r3+150*g3+29*b3)>>8;
    int u01=clamp(((-43*r0-85*g0+128*b0)>>8)+128,0,255);
    int v01=clamp(((128*r0-107*g0-21*b0)>>8)+128,0,255);
    int u23=clamp(((-43*r2-85*g2+128*b2)>>8)+128,0,255);
    int v23=clamp(((128*r2-107*g2-21*b2)>>8)+128,0,255);
    uint word0 = (uint)u01 | ((uint)y0<<8) | ((uint)v01<<16) | ((uint)y1<<24);
    uint word1 = (uint)u23 | ((uint)y2<<8) | ((uint)v23<<16) | ((uint)y3<<24);
    uint uyvyOffset = y * gWidth * 2 + tid.x * 8;
    gDst.Store(uyvyOffset, word0);
    gDst.Store(uyvyOffset + 4, word1);
    if (gWriteAlpha) {
        uint aword = (uint)toByte(c0.a) | ((uint)toByte(c1.a)<<8) | ((uint)toByte(c2.a)<<16) | ((uint)toByte(c3.a)<<24);
        gDst.Store(gUyvySize + y * gWidth + tid.x * 4, aword);
    }
}
)HLSL";

struct Params {
    uint32_t width;
    uint32_t height;
    uint32_t uyvySize;
    uint32_t writeAlpha;
    uint32_t videoOn;
    uint32_t videoW;
    uint32_t videoH;
    uint32_t videoFormat;
};

// ---------------------------------------------------------------------------------------------------------
// Stage-2 sub-step B (plan §5 ladder / §10 harness soak): backend selection for the CONVERT copy-out.
// Preferred: D3D11On12 — a D3D12 device wrapped via D3D11On12CreateDevice runs every line of the existing
// D3D11 consume code unchanged; the convert output lands in a wrapped D3D12 UAV buffer, a D3D12 COPY-queue
// CopyBufferRegion moves it into a D3D12_HEAP_TYPE_READBACK buffer (WRITE_BACK / CPU-cached by spec), and a
// SINGLE-DEVICE copy-queue fence (SetEventOnCompletion + WaitForSingleObject) replaces the WaitGpu busy-spin.
// Soak-measured (on12_harness, GATE-C2): ~ -50% process CPU at identical 3x4K60 throughput — the spin dies
// and the copy-out becomes a cached memcpy, freeing the CPU the concurrent SpeedHQ compresses are starved of.
// Pre-registered NOT to move single-output done (~1.1x copy-out); the target is TWO-output sentReal + CPU.
// LADDER (never crash, always degrade): any 11On12/D3D12 init failure -> process-wide fallback to the classic
// D3D11 WC-staging path (byte-identical to the shipping build); D3D11 failure -> the JS side falls back to
// CPU OSR. NOT the abandoned cross-device ID3D11Fence (TDR'd): one device, one fence, one queue — the exact
// construction the harness gate matrix passed (byte-exact, zero TDR, clean teardown).
// FS_READBACK=d3d11 forces the classic path — a binary DIAGNOSTIC selector for in-app A/B verification
// (observability, like FS_CAP_STATS), not a tuned perf knob; there is no value for a human to sweep.
std::atomic<bool> g_on12Disabled{false};
std::atomic<int> g_backendState{0};  // 0 = no device yet, 1 = d3d11on12, 2 = d3d11 (classic / disabled)

bool On12Allowed() {
    static const bool envForced = [] {
        const char* v = std::getenv("FS_READBACK");
        return v && std::strcmp(v, "d3d11") == 0;
    }();
    if (envForced) return false;
    return !g_on12Disabled.load();
}

void LogBackendOnce(const char* name, const char* detail) {
    static std::atomic<bool> logged{false};
    bool expected = false;
    if (logged.compare_exchange_strong(expected, true)) {
        fprintf(stderr, "[READBACK] backend=%s (%s)\n", name, detail);
        fflush(stderr);
    }
}

void DisableOn12(const char* reason) {
    bool expected = false;
    if (g_on12Disabled.compare_exchange_strong(expected, true)) {
        g_backendState.store(2);
        fprintf(stderr, "[READBACK] d3d11on12 disabled (%s) -> classic d3d11 WC-staging path\n", reason);
        fflush(stderr);
    }
}

// Box-filter downscale of the shared BGRA texture straight to a small tightly-packed BGRA buffer, ON THE GPU.
// Used for server/stage/preview consumers of a mixed output: instead of reading back the full 4K frame and
// downscaling on the CPU, the GPU produces a small buffer so only a few MB cross PCIe. Output byte order is
// B,G,R,A (matches the CPU BGRA readback + nativeImage.createFromBitmap). Src/Dst dims fit in uint32 for 4K.
const char* kDownscaleHLSL = R"HLSL(
Texture2D<float4> gSrc : register(t0);
RWByteAddressBuffer gDst : register(u0);
cbuffer DsParams : register(b0) {
    uint gSrcW; uint gSrcH; uint gDstW; uint gDstH;
};
int toByte(float v) { return (int)(v * 255.0 + 0.5); }
[numthreads(8, 8, 1)]
void DSMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= gDstW || tid.y >= gDstH) return;
    uint sx0 = tid.x * gSrcW / gDstW;
    uint sx1 = (tid.x + 1) * gSrcW / gDstW; if (sx1 <= sx0) sx1 = sx0 + 1;
    uint sy0 = tid.y * gSrcH / gDstH;
    uint sy1 = (tid.y + 1) * gSrcH / gDstH; if (sy1 <= sy0) sy1 = sy0 + 1;
    float4 acc = float4(0, 0, 0, 0);
    uint cnt = 0;
    for (uint y = sy0; y < sy1; ++y) {
        for (uint x = sx0; x < sx1; ++x) { acc += gSrc.Load(int3(x, y, 0)); ++cnt; }
    }
    acc /= cnt;
    uint word = (uint)toByte(acc.b) | ((uint)toByte(acc.g) << 8) | ((uint)toByte(acc.r) << 16) | ((uint)toByte(acc.a) << 24);
    gDst.Store((tid.y * gDstW + tid.x) * 4, word);
}
)HLSL";

struct DsParams {
    uint32_t srcW;
    uint32_t srcH;
    uint32_t dstW;
    uint32_t dstH;
};

// Per-consumer target: box-downscale the shared BGRA to dstW x dstH and pack straight into the consumer's
// format, so a smaller output of the same render costs one small extra pass instead of its own render.
// Same integer block bounds as DSMain (float accumulate). One thread per PAIR of output pixels: UYVY/UYVA
// need the pair for chroma; BGRA writes both. Alpha plane (UYVA) follows the UYVY plane at gUyvySize.
const char* kScaleConvertHLSL = R"HLSL(
Texture2D<float4> gSrc : register(t0);
RWByteAddressBuffer gDst : register(u0);
cbuffer ScParams : register(b0) {
    uint gSrcW; uint gSrcH; uint gDstW; uint gDstH;
    uint gFormat; uint gUyvySize; uint gPad0; uint gPad1;
    uint gVideoOn; uint gVideoW; uint gVideoH; uint gVideoFormat;
};

// The video layer sits under the page: where the page is transparent, its pixels show through. UYVY is
// stored as a half-width RGBA texture (U Y0 V Y1 per texel) and decoded with the same limited-range
// matrices the renderer uses, so a composited frame matches one drawn in the page.
Texture2D<float4> gVideo : register(t1);
float3 videoAt(uint x, uint y, uint dstW, uint dstH) {
    uint vx = (dstW == gVideoW) ? x : x * gVideoW / dstW;
    uint vy = (dstH == gVideoH) ? y : y * gVideoH / dstH;
    if (gVideoFormat != 1) return gVideo.Load(int3(vx, vy, 0)).rgb;
    float4 s = gVideo.Load(int3(vx / 2, vy, 0));
    float luma = ((vx & 1) == 0) ? s.g : s.a;
    float yy = (luma - 16.0 / 255.0) * (255.0 / 219.0);
    float u = (s.r - 128.0 / 255.0) * (255.0 / 224.0);
    float v = (s.b - 128.0 / 255.0) * (255.0 / 224.0);
    float bt = (gVideoH >= 720) ? 1.0 : 0.0;
    float kr = lerp(1.402, 1.5748, bt), kb = lerp(1.772, 1.8556, bt);
    float gu = lerp(0.344136, 0.1873, bt), gv = lerp(0.714136, 0.4681, bt);
    return saturate(float3(yy + kr * v, yy - gu * u - gv * v, yy + kb * u));
}
// page over video, premultiplied: opaque page pixels are untouched
float4 overVideo(float4 page, uint x, uint y, uint dstW, uint dstH) {
    if (gVideoOn == 0) return page;
    if (page.a >= 0.999) return page;
    return float4(page.rgb + videoAt(x, y, dstW, dstH) * (1.0 - page.a), 1.0);
}
int toByte(float v) { return (int)(v * 255.0 + 0.5); }
float4 boxAvg(uint dx, uint dy) {
    if (gSrcW == gDstW && gSrcH == gDstH) return gSrc.Load(int3(dx, dy, 0));
    uint sx0 = dx * gSrcW / gDstW;
    uint sx1 = (dx + 1) * gSrcW / gDstW; if (sx1 <= sx0) sx1 = sx0 + 1;
    uint sy0 = dy * gSrcH / gDstH;
    uint sy1 = (dy + 1) * gSrcH / gDstH; if (sy1 <= sy0) sy1 = sy0 + 1;
    float4 acc = float4(0, 0, 0, 0);
    uint cnt = 0;
    for (uint y = sy0; y < sy1; ++y) {
        for (uint x = sx0; x < sx1; ++x) { acc += gSrc.Load(int3(x, y, 0)); ++cnt; }
    }
    return acc / cnt;
}
[numthreads(8, 8, 1)]
void SCMain(uint3 tid : SV_DispatchThreadID) {
    uint px = tid.x * 2;
    uint y = tid.y;
    if (px >= gDstW || y >= gDstH) return;
    float4 c0 = overVideo(boxAvg(px, y), px, y, gDstW, gDstH);
    float4 c1 = (px + 1 < gDstW) ? overVideo(boxAvg(px + 1, y), px + 1, y, gDstW, gDstH) : c0;
    if (gFormat == 0) {
        uint w0 = (uint)toByte(c0.b) | ((uint)toByte(c0.g) << 8) | ((uint)toByte(c0.r) << 16) | ((uint)toByte(c0.a) << 24);
        gDst.Store((y * gDstW + px) * 4, w0);
        if (px + 1 < gDstW) {
            uint w1 = (uint)toByte(c1.b) | ((uint)toByte(c1.g) << 8) | ((uint)toByte(c1.r) << 16) | ((uint)toByte(c1.a) << 24);
            gDst.Store((y * gDstW + px + 1) * 4, w1);
        }
        return;
    }
    int r0=toByte(c0.r),g0=toByte(c0.g),b0=toByte(c0.b);
    int r1=toByte(c1.r),g1=toByte(c1.g),b1=toByte(c1.b);
    int y0=(77*r0+150*g0+29*b0)>>8;
    int y1=(77*r1+150*g1+29*b1)>>8;
    int u=clamp(((-43*r0-85*g0+128*b0)>>8)+128,0,255);
    int v=clamp(((128*r0-107*g0-21*b0)>>8)+128,0,255);
    uint word = (uint)u | ((uint)y0<<8) | ((uint)v<<16) | ((uint)y1<<24);
    gDst.Store(y * gDstW * 2 + px * 2, word);
    if (gFormat == 2) {
        // alpha plane, one byte per pixel; this thread owns bytes px and px+1 -> write as a 16-bit store
        // via a 32-bit read-modify-write is not available on raw buffers, so the pair is packed per 4 pixels
        // by the thread whose px % 4 == 0 using its neighbour pair's alpha.
        if ((px & 3) == 0) {
            float4 c2 = (px + 2 < gDstW) ? boxAvg(px + 2, y) : c1;
            float4 c3 = (px + 3 < gDstW) ? boxAvg(px + 3, y) : c2;
            uint aword = (uint)toByte(c0.a) | ((uint)toByte(c1.a)<<8) | ((uint)toByte(c2.a)<<16) | ((uint)toByte(c3.a)<<24);
            gDst.Store(gUyvySize + y * gDstW + px, aword);
        }
    }
}
)HLSL";

// I420 (planar 4:2:0, BT.601 limited range) target: one thread per 8x2 block writes two Y words per row
// and one U + one V word (4 chroma samples), so every store is a whole 32-bit word. Widths that are not
// multiples of 8 keep their trailing partial words unwritten (guarded), which the caller sizes to avoid.
const char* kScaleI420HLSL = R"HLSL(
Texture2D<float4> gSrc : register(t0);
RWByteAddressBuffer gDst : register(u0);
cbuffer ScParams : register(b0) {
    uint gSrcW; uint gSrcH; uint gDstW; uint gDstH;
    uint gFormat; uint gUyvySize; uint gPad0; uint gPad1;
    uint gVideoOn; uint gVideoW; uint gVideoH; uint gVideoFormat;
};
Texture2D<float4> gVideo : register(t1);
float3 videoAt(uint x, uint y, uint dstW, uint dstH) {
    uint vx = (dstW == gVideoW) ? x : x * gVideoW / dstW;
    uint vy = (dstH == gVideoH) ? y : y * gVideoH / dstH;
    if (gVideoFormat != 1) return gVideo.Load(int3(vx, vy, 0)).rgb;
    float4 s = gVideo.Load(int3(vx / 2, vy, 0));
    float luma = ((vx & 1) == 0) ? s.g : s.a;
    float yy = (luma - 16.0 / 255.0) * (255.0 / 219.0);
    float u = (s.r - 128.0 / 255.0) * (255.0 / 224.0);
    float v = (s.b - 128.0 / 255.0) * (255.0 / 224.0);
    float bt = (gVideoH >= 720) ? 1.0 : 0.0;
    float kr = lerp(1.402, 1.5748, bt), kb = lerp(1.772, 1.8556, bt);
    float gu = lerp(0.344136, 0.1873, bt), gv = lerp(0.714136, 0.4681, bt);
    return saturate(float3(yy + kr * v, yy - gu * u - gv * v, yy + kb * u));
}
float4 overVideo(float4 page, uint x, uint y, uint dstW, uint dstH) {
    if (gVideoOn == 0) return page;
    if (page.a >= 0.999) return page;
    return float4(page.rgb + videoAt(x, y, dstW, dstH) * (1.0 - page.a), 1.0);
}
float4 boxAvg(uint dx, uint dy) {
    if (gSrcW == gDstW && gSrcH == gDstH) return overVideo(gSrc.Load(int3(dx, dy, 0)), dx, dy, gDstW, gDstH);
    uint sx0 = dx * gSrcW / gDstW;
    uint sx1 = (dx + 1) * gSrcW / gDstW; if (sx1 <= sx0) sx1 = sx0 + 1;
    uint sy0 = dy * gSrcH / gDstH;
    uint sy1 = (dy + 1) * gSrcH / gDstH; if (sy1 <= sy0) sy1 = sy0 + 1;
    float4 acc = float4(0, 0, 0, 0);
    uint cnt = 0;
    for (uint y = sy0; y < sy1; ++y) {
        for (uint x = sx0; x < sx1; ++x) { acc += gSrc.Load(int3(x, y, 0)); ++cnt; }
    }
    return overVideo(acc / cnt, dx, dy, gDstW, gDstH);
}
uint lumaOf(float4 c) {
    int r = (int)(c.r * 255.0 + 0.5), g = (int)(c.g * 255.0 + 0.5), b = (int)(c.b * 255.0 + 0.5);
    return (uint)clamp(((66 * r + 129 * g + 25 * b + 128) >> 8) + 16, 16, 235);
}
[numthreads(8, 8, 1)]
void SCI420Main(uint3 tid : SV_DispatchThreadID) {
    uint bx = tid.x * 8;
    uint by = tid.y * 2;
    if (bx >= gDstW || by >= gDstH) return;
    uint ySize = gDstW * gDstH;
    uint cw = gDstW / 2;
    uint ch = gDstH / 2;
    float4 c[16];
    [unroll] for (uint r = 0; r < 2; ++r) {
        [unroll] for (uint i = 0; i < 8; ++i) {
            uint dx = min(bx + i, gDstW - 1);
            uint dy = min(by + r, gDstH - 1);
            c[r * 8 + i] = boxAvg(dx, dy);
        }
    }
    [unroll] for (uint r2 = 0; r2 < 2; ++r2) {
        if (by + r2 >= gDstH) continue;
        uint rowOff = (by + r2) * gDstW + bx;
        if (bx + 3 < gDstW) {
            uint w0 = lumaOf(c[r2 * 8 + 0]) | (lumaOf(c[r2 * 8 + 1]) << 8) | (lumaOf(c[r2 * 8 + 2]) << 16) | (lumaOf(c[r2 * 8 + 3]) << 24);
            gDst.Store(rowOff, w0);
        }
        if (bx + 7 < gDstW) {
            uint w1 = lumaOf(c[r2 * 8 + 4]) | (lumaOf(c[r2 * 8 + 5]) << 8) | (lumaOf(c[r2 * 8 + 6]) << 16) | (lumaOf(c[r2 * 8 + 7]) << 24);
            gDst.Store(rowOff + 4, w1);
        }
    }
    uint cy = by / 2;
    uint cx = bx / 2;
    if (cy < ch && cx + 3 < cw) {
        uint uw = 0, vw = 0;
        [unroll] for (uint k = 0; k < 4; ++k) {
            float4 a = (c[k * 2] + c[k * 2 + 1] + c[8 + k * 2] + c[8 + k * 2 + 1]) * 0.25;
            int r = (int)(a.r * 255.0 + 0.5), g = (int)(a.g * 255.0 + 0.5), b = (int)(a.b * 255.0 + 0.5);
            uint u = (uint)clamp(((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128, 16, 240);
            uint v = (uint)clamp(((112 * r - 94 * g - 18 * b + 128) >> 8) + 128, 16, 240);
            uw |= u << (k * 8);
            vw |= v << (k * 8);
        }
        uint cOff = cy * cw + cx;
        gDst.Store(ySize + cOff, uw);
        gDst.Store(ySize + cw * ch + cOff, vw);
    }
}
)HLSL";

struct ScParams {
    uint32_t srcW;
    uint32_t srcH;
    uint32_t dstW;
    uint32_t dstH;
    uint32_t format;
    uint32_t uyvySize;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t videoOn;
    uint32_t videoW;
    uint32_t videoH;
    uint32_t videoFormat;
};

// BGRA -> RGBA channel swap on the GPU (for WebRTC's ImageData, which is RGBA). Same size as the source, so the
// only cost over a plain BGRA readback is the swizzle, which the GPU does for free while the readback runs.
const char* kSwizzleHLSL = R"HLSL(
Texture2D<float4> gSrc : register(t0);
RWByteAddressBuffer gDst : register(u0);
cbuffer SwParams : register(b0) {
    uint gWidth; uint gHeight;
};
int toByte(float v) { return (int)(v * 255.0 + 0.5); }
[numthreads(8, 8, 1)]
void SwizzleMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= gWidth || tid.y >= gHeight) return;
    float4 c = gSrc.Load(int3(tid.x, tid.y, 0));  // .r=R .g=G .b=B .a=A
    uint word = (uint)toByte(c.r) | ((uint)toByte(c.g) << 8) | ((uint)toByte(c.b) << 16) | ((uint)toByte(c.a) << 24);
    gDst.Store((tid.y * gWidth + tid.x) * 4, word);  // R,G,B,A byte order = RGBA
}
)HLSL";

// D3D11 constant buffers must be a multiple of 16 bytes, so pad to 16 (the shader only reads width/height).
struct SwParams {
    uint32_t width;
    uint32_t height;
    uint32_t pad0;
    uint32_t pad1;
};

struct ReadbackContext {
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11Device1> device1;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<ID3D11ComputeShader> shader;
    ComPtr<ID3D11Buffer> paramsCb;

    // BGRA->RGBA swizzle (WebRTC) shader + its params; reuses outBuf/outStaging for the output
    ComPtr<ID3D11ComputeShader> swizzleShader;
    ComPtr<ID3D11Buffer> swizzleParamsCb;

    // BGRA path staging
    ComPtr<ID3D11Texture2D> staging;
    UINT stagingW = 0, stagingH = 0;

    // compute path output buffer + its staging
    ComPtr<ID3D11Buffer> outBuf;
    ComPtr<ID3D11UnorderedAccessView> outUav;
    ComPtr<ID3D11Buffer> outStaging;
    size_t outBufSize = 0;

    // GPU-downscale (server/stage) path: its own shader + output buffer/staging
    ComPtr<ID3D11ComputeShader> dsShader;
    ComPtr<ID3D11Buffer> dsParamsCb;
    ComPtr<ID3D11Buffer> dsOutBuf;
    ComPtr<ID3D11UnorderedAccessView> dsOutUav;
    ComPtr<ID3D11Buffer> dsOutStaging;
    size_t dsOutBufSize = 0;

    // video layer composited under the page (osrcap::VideoLayer), uploaded on this device
    ComPtr<ID3D11Texture2D> videoTex;
    ComPtr<ID3D11ShaderResourceView> videoSrv;
    ComPtr<ID3D11Texture2D> dummyTex;
    ComPtr<ID3D11ShaderResourceView> dummySrv;
    uint32_t videoTexW = 0;
    uint32_t videoTexH = 0;
    uint32_t videoW = 0;
    uint32_t videoH = 0;
    int videoFormat = 0;
    bool videoOn = false;

    // per-consumer targets (osrcap::TargetSpec): downscale+convert in one pass, one slot per target
    ComPtr<ID3D11ComputeShader> scShader;
    ComPtr<ID3D11ComputeShader> scI420Shader;
    ComPtr<ID3D11Buffer> scParamsCb;
    struct TargetSlot {
        ComPtr<ID3D11Buffer> buf;
        ComPtr<ID3D11UnorderedAccessView> uav;
        ComPtr<ID3D11Buffer> staging;
        size_t bytes = 0;
        size_t pending = 0;  // bytes waiting in `staging` for this frame (0 = none)
    };
    std::vector<TargetSlot> targetSlots;

    bool inUse = false;

    // two-phase state: GPU pipeline fence + the pending result waiting to be copied out
    ComPtr<ID3D11Query> flushQuery;
    size_t pendingTotal = 0;
    bool pendingIsConvert = false;
    std::vector<uint8_t> pendingBgra;
    // pending GPU-downscaled buffer (0 = none for this frame); dsOutStaging holds it after the GPU wait
    size_t pendingScaledTotal = 0;

    // ---- Stage-2 sub-step B: D3D11On12 cached-readback state (see the ladder comment above Params) ----
    // Per-context D3D12 chain, mirroring the existing per-context D3D11 device design (each pool entry is
    // independent, no cross-context sync). All structural D3D objects — no perf constants.
    ComPtr<ID3D12Device> d12;
    ComPtr<ID3D12CommandQueue> d12Direct;  // backs the 11On12 device (all wrapped D3D11 work lands here)
    ComPtr<ID3D12CommandQueue> d12Copy;    // dedicated COPY queue: convert output -> READBACK heap DMA
    ComPtr<ID3D12Fence> d12Fence;          // SINGLE-device fence (the TDR-safe kind), monotonically valued
    UINT64 d12FenceVal = 0;
    HANDLE d12Event = nullptr;
    ComPtr<ID3D12CommandAllocator> d12CopyAlloc;
    ComPtr<ID3D12GraphicsCommandList> d12CopyList;
    ComPtr<ID3D11On12Device> on12;
    ComPtr<ID3D12Resource> convBuf12;  // DEFAULT-heap UAV buffer; wrapped into outBuf when outBufOn12
    ComPtr<ID3D12Resource> rbBuf;      // READBACK heap (WRITE_BACK / CPU-cached), persistently mapped
    void* rbPtr = nullptr;
    bool on12Active = false;      // this context's device chain is 11On12 (vs classic D3D11)
    bool on12OpenedOk = false;    // the 11On12 device has successfully opened an Electron shared texture
    bool outBufOn12 = false;      // outBuf/outUav currently wrap convBuf12 (vs classic buffer + outStaging)
    bool pendingOn12 = false;     // the pending convert output lives in rbBuf (vs outStaging)
    UINT64 pendingCopyFence = 0;  // copy-queue completion value for the pending frame's DMA
    bool deviceLost = false;      // fence watchdog fired / device removed -> rebuild the chain on next use

    ~ReadbackContext() {
        if (rbPtr && rbBuf) rbBuf->Unmap(0, nullptr);
        if (d12Event) CloseHandle(d12Event);
    }

    // Full device-chain reset (both backends): every device-derived object dies with the device. Used when a
    // backend init fails partway (fall back clean) and when the fence watchdog marks the device lost.
    void ResetDeviceChain() {
        shader.Reset();
        paramsCb.Reset();
        swizzleShader.Reset();
        swizzleParamsCb.Reset();
        staging.Reset();
        stagingW = stagingH = 0;
        outBuf.Reset();
        outUav.Reset();
        outStaging.Reset();
        outBufSize = 0;
        dsShader.Reset();
        dsParamsCb.Reset();
        dsOutBuf.Reset();
        dsOutUav.Reset();
        dsOutStaging.Reset();
        dsOutBufSize = 0;
        flushQuery.Reset();
        pendingBgra.clear();
        pendingTotal = pendingScaledTotal = 0;
        if (rbPtr && rbBuf) rbBuf->Unmap(0, nullptr);
        rbPtr = nullptr;
        rbBuf.Reset();
        convBuf12.Reset();
        outBufOn12 = false;
        pendingOn12 = false;
        pendingCopyFence = 0;
        on12.Reset();
        d12CopyList.Reset();
        d12CopyAlloc.Reset();
        d12Fence.Reset();
        d12FenceVal = 0;
        if (d12Event) { CloseHandle(d12Event); d12Event = nullptr; }
        context.Reset();
        device1.Reset();
        device.Reset();
        d12Copy.Reset();
        d12Direct.Reset();
        d12.Reset();
        on12Active = false;
        on12OpenedOk = false;
        deviceLost = false;
    }

    // Try the full 11On12 chain on the DEFAULT adapter (same adapter selection as the classic path). Any
    // failure returns false with the chain partially built — the caller resets and degrades.
    bool InitOn12() {
        if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d12)))) return false;
        D3D12_COMMAND_QUEUE_DESC qd = {};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (FAILED(d12->CreateCommandQueue(&qd, IID_PPV_ARGS(&d12Direct)))) return false;
        qd.Type = D3D12_COMMAND_LIST_TYPE_COPY;
        if (FAILED(d12->CreateCommandQueue(&qd, IID_PPV_ARGS(&d12Copy)))) return false;
        if (FAILED(d12->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&d12Fence)))) return false;
        d12Event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!d12Event) return false;
        if (FAILED(d12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&d12CopyAlloc))) ||
            FAILED(d12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, d12CopyAlloc.Get(), nullptr, IID_PPV_ARGS(&d12CopyList))))
            return false;
        d12CopyList->Close();  // created open; keep the per-frame Reset pattern uniform (as in the harness)
        IUnknown* queues[] = {d12Direct.Get()};
        if (FAILED(D3D11On12CreateDevice(d12.Get(), D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0, queues, 1, 0,
                                         &device, &context, nullptr)))
            return false;
        if (FAILED(device.As(&device1)) || FAILED(device.As(&on12))) return false;
        return true;
    }

    // Event-blocking fence wait (NO spin). On timeout/device-removal: mark the context lost (rebuilt on next
    // use) and disable 11On12 process-wide — a watchdog trip is the exact TDR class the abandoned fence build
    // taught us never to busy-retry; the classic path is the proven fallback.
    bool WaitFenceOn12(UINT64 v, DWORD ms) {
        if (!d12Fence) return true;
        if (d12Fence->GetCompletedValue() >= v) return true;
        if (FAILED(d12Fence->SetEventOnCompletion(v, d12Event))) { deviceLost = true; return false; }
        if (WaitForSingleObject(d12Event, ms) == WAIT_OBJECT_0) return true;
        deviceLost = true;
        DisableOn12(d12->GetDeviceRemovedReason() != S_OK ? "device removed" : "copy/convert fence watchdog timeout");
        return false;
    }

    bool EnsureDevice() {
        if (deviceLost) ResetDeviceChain();
        if (device1) return true;
        // §5 ladder step 1: D3D11On12 (cached readback). Failure degrades to the classic path — process-wide,
        // so later contexts don't re-pay a failing init per frame.
        if (On12Allowed()) {
            if (InitOn12()) {
                on12Active = true;
                int expected = 0;
                g_backendState.compare_exchange_strong(expected, 1);
                LogBackendOnce("d3d11on12", "cached D3D12 readback heap, copy-queue fence");
                return true;
            }
            DisableOn12("D3D12/11On12 device init failed");
            ResetDeviceChain();
        }
        // §5 ladder step 2: classic D3D11 (WC staging + WaitGpu) — byte-identical to the shipping build.
        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
        D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0;
        if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags, nullptr, 0, D3D11_SDK_VERSION, &device, &fl, &context))) return false;
        if (FAILED(device.As(&device1))) return false;
        int expected = 0;
        g_backendState.compare_exchange_strong(expected, 2);
        LogBackendOnce("d3d11", "WC staging + WaitGpu");
        return true;
    }

    // Open Electron's shared texture on this context's device, with the §5 ladder's one-shot open fallback:
    // if the 11On12 device has NEVER successfully opened a shared texture and the open fails, treat 11On12 as
    // incompatible with this driver's shared-handle path (structural, not a retry count), disable it
    // process-wide, rebuild this context classic, and retry the open once. Once a context HAS opened
    // successfully, later failures are treated as transient (resize churn, dead handle) exactly as today.
    bool OpenSharedTex(uintptr_t handle, ComPtr<ID3D11Texture2D>& shared, std::string& err) {
        if (!EnsureDevice()) { err = "D3D11CreateDevice failed"; return false; }
        HRESULT hr = device1->OpenSharedResource1(reinterpret_cast<HANDLE>(handle), IID_PPV_ARGS(&shared));
        if (FAILED(hr)) hr = device->OpenSharedResource(reinterpret_cast<HANDLE>(handle), IID_PPV_ARGS(&shared));
        if ((FAILED(hr) || !shared) && on12Active && !on12OpenedOk) {
            DisableOn12("OpenSharedResource failed on the 11On12 device");
            ResetDeviceChain();
            if (!EnsureDevice()) { err = "D3D11CreateDevice failed"; return false; }
            hr = device1->OpenSharedResource1(reinterpret_cast<HANDLE>(handle), IID_PPV_ARGS(&shared));
            if (FAILED(hr)) hr = device->OpenSharedResource(reinterpret_cast<HANDLE>(handle), IID_PPV_ARGS(&shared));
        }
        if (FAILED(hr) || !shared) { err = "OpenSharedResource failed"; return false; }
        if (on12Active) on12OpenedOk = true;
        return true;
    }

    bool EnsureShader() {
        if (shader) return true;
        ComPtr<ID3DBlob> blob, errBlob;
        HRESULT hr = D3DCompile(kConvertHLSL, strlen(kConvertHLSL), "convert.hlsl", nullptr, nullptr, "CSMain", "cs_5_0", 0, 0, &blob, &errBlob);
        if (FAILED(hr)) return false;
        if (FAILED(device->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &shader))) return false;
        D3D11_BUFFER_DESC cb = {};
        cb.ByteWidth = sizeof(Params);
        cb.Usage = D3D11_USAGE_DEFAULT;
        cb.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        return SUCCEEDED(device->CreateBuffer(&cb, nullptr, &paramsCb));
    }

    bool EnsureSwizzleShader() {
        if (swizzleShader && swizzleParamsCb) return true;
        ComPtr<ID3DBlob> blob, errBlob;
        HRESULT hr = D3DCompile(kSwizzleHLSL, strlen(kSwizzleHLSL), "swizzle.hlsl", nullptr, nullptr, "SwizzleMain", "cs_5_0", 0, 0, &blob, &errBlob);
        if (FAILED(hr)) return false;
        if (FAILED(device->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &swizzleShader))) return false;
        D3D11_BUFFER_DESC cb = {};
        cb.ByteWidth = sizeof(SwParams);
        cb.Usage = D3D11_USAGE_DEFAULT;
        cb.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        return SUCCEEDED(device->CreateBuffer(&cb, nullptr, &swizzleParamsCb));
    }

    bool EnsureStaging(UINT w, UINT h) {
        if (staging && stagingW == w && stagingH == h) return true;
        D3D11_TEXTURE2D_DESC d = {};
        d.Width = w; d.Height = h; d.MipLevels = 1; d.ArraySize = 1;
        d.Format = DXGI_FORMAT_B8G8R8A8_UNORM; d.SampleDesc.Count = 1;
        d.Usage = D3D11_USAGE_STAGING; d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ComPtr<ID3D11Texture2D> tex;
        if (FAILED(device->CreateTexture2D(&d, nullptr, &tex))) return false;
        staging = tex; stagingW = w; stagingH = h;
        return true;
    }

    // Release the on12 convert/readback pair (draining any in-flight copy-queue DMA first, so the DMA never
    // reads/writes a dying buffer). Safe no-op when nothing on12 exists.
    void ReleaseOn12OutBuffers() {
        if (pendingCopyFence) { WaitFenceOn12(pendingCopyFence, 2000); pendingCopyFence = 0; }
        if (rbPtr && rbBuf) rbBuf->Unmap(0, nullptr);
        rbPtr = nullptr;
        rbBuf.Reset();
        convBuf12.Reset();
        outBufOn12 = false;
    }

    // A CLASSIC-STAGING consumer (ReadbackConvert/ReadbackRgba/ConvertToStaging — they CopyResource into
    // outStaging) is about to use outBuf: if the installed pair is the on12 wrap (no staging; needs
    // Acquire/Release around every use), drop it so EnsureOutBuffer rebuilds the classic pair. Without this
    // a coincidental byte-size match would early-return the WRAPPED buffer into a classic CopyResource
    // (invalid) with a null outStaging. The on12 convert path (ConsumeShared) never calls this — its
    // DispatchConvert must early-return on the installed wrapped pair, exactly like the harness's injection.
    void InvalidateOn12OutBuffer() {
        if (!outBufOn12) return;
        ReleaseOn12OutBuffers();
        outBuf.Reset();
        outUav.Reset();
        outBufSize = 0;
    }

    // "A convert-output buffer of this size is installed" — satisfied by the classic pair, the harness's
    // injected wrap, or the live on12 wrap (EnsureOn12OutBuffer installs it before DispatchConvert runs).
    bool EnsureOutBuffer(size_t bytes) {
        if (outBuf && outBufSize == bytes) return true;
        if (outBufOn12) ReleaseOn12OutBuffers();
        outBuf.Reset(); outUav.Reset(); outStaging.Reset();
        D3D11_BUFFER_DESC bd = {};
        bd.ByteWidth = (UINT)bytes;
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        if (FAILED(device->CreateBuffer(&bd, nullptr, &outBuf))) return false;
        D3D11_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.Format = DXGI_FORMAT_R32_TYPELESS;
        ud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        ud.Buffer.FirstElement = 0;
        ud.Buffer.NumElements = (UINT)(bytes / 4);
        ud.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
        if (FAILED(device->CreateUnorderedAccessView(outBuf.Get(), &ud, &outUav))) return false;
        D3D11_BUFFER_DESC sd = {};
        sd.ByteWidth = (UINT)bytes;
        sd.Usage = D3D11_USAGE_STAGING;
        sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        if (FAILED(device->CreateBuffer(&sd, nullptr, &outStaging))) return false;
        outBufSize = bytes;
        return true;
    }

    // On12 output pair (harness EnsureSize, folded in): a D3D12 DEFAULT UAV buffer wrapped into D3D11 as
    // outBuf/outUav — so the UNCHANGED DispatchConvert writes straight into D3D12-owned memory — plus the
    // CPU-cached READBACK-heap buffer the copy queue DMAs into, persistently mapped. Buffers have
    // simultaneous-access state semantics (promote from / decay to COMMON at every ExecuteCommandLists), so
    // COMMON in/out lets the COPY queue promote to COPY_SOURCE implicitly. On ANY failure: false — the caller
    // falls back to the classic staging path for the frame (classic buffers work fine on the 11On12 device).
    bool EnsureOn12OutBuffer(size_t bytes) {
        if (outBufOn12 && outBufSize == bytes) return true;
        // drain everything in flight before releasing the old pair, then drop any classic leftovers too
        if (d12FenceVal && !WaitFenceOn12(d12FenceVal, 2000)) return false;
        ReleaseOn12OutBuffers();
        outBuf.Reset(); outUav.Reset(); outStaging.Reset();
        outBufSize = 0;

        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC bd = {};
        bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bd.Width = bytes;
        bd.Height = 1;
        bd.DepthOrArraySize = 1;
        bd.MipLevels = 1;
        bd.Format = DXGI_FORMAT_UNKNOWN;
        bd.SampleDesc.Count = 1;
        bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        bd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        if (FAILED(d12->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&convBuf12)))) return false;

        D3D11_RESOURCE_FLAGS f11 = {};
        f11.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        f11.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        ComPtr<ID3D11Buffer> buf11;
        if (FAILED(on12->CreateWrappedResource(convBuf12.Get(), &f11, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COMMON, IID_PPV_ARGS(&buf11)))) {
            convBuf12.Reset();
            return false;
        }

        // Same RAW R32_TYPELESS UAV desc as the classic EnsureOutBuffer; creation needs the wrap acquired.
        D3D11_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.Format = DXGI_FORMAT_R32_TYPELESS;
        ud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        ud.Buffer.FirstElement = 0;
        ud.Buffer.NumElements = (UINT)(bytes / 4);
        ud.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
        ComPtr<ID3D11UnorderedAccessView> uav;
        HRESULT hr;
        {
            ID3D11Resource* res[] = {buf11.Get()};
            on12->AcquireWrappedResources(res, 1);
            hr = device->CreateUnorderedAccessView(buf11.Get(), &ud, &uav);
            on12->ReleaseWrappedResources(res, 1);
        }
        if (FAILED(hr)) { convBuf12.Reset(); return false; }

        hp.Type = D3D12_HEAP_TYPE_READBACK;
        bd.Flags = D3D12_RESOURCE_FLAG_NONE;
        if (FAILED(d12->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rbBuf))) ||
            FAILED(rbBuf->Map(0, nullptr, &rbPtr))) {
            rbBuf.Reset();
            rbPtr = nullptr;
            convBuf12.Reset();
            return false;
        }

        outBuf = buf11;
        outUav = uav;
        outBufSize = bytes;
        outBufOn12 = true;
        return true;
    }

    bool EnsureDsShader() {
        if (dsShader) return true;
        ComPtr<ID3DBlob> blob, errBlob;
        HRESULT hr = D3DCompile(kDownscaleHLSL, strlen(kDownscaleHLSL), "downscale.hlsl", nullptr, nullptr, "DSMain", "cs_5_0", 0, 0, &blob, &errBlob);
        if (FAILED(hr)) return false;
        if (FAILED(device->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &dsShader))) return false;
        D3D11_BUFFER_DESC cb = {};
        cb.ByteWidth = sizeof(DsParams);
        cb.Usage = D3D11_USAGE_DEFAULT;
        cb.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        return SUCCEEDED(device->CreateBuffer(&cb, nullptr, &dsParamsCb));
    }

    bool EnsureDsOutBuffer(size_t bytes) {
        if (dsOutBuf && dsOutBufSize == bytes) return true;
        dsOutBuf.Reset(); dsOutUav.Reset(); dsOutStaging.Reset();
        D3D11_BUFFER_DESC bd = {};
        bd.ByteWidth = (UINT)bytes;
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        if (FAILED(device->CreateBuffer(&bd, nullptr, &dsOutBuf))) return false;
        D3D11_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.Format = DXGI_FORMAT_R32_TYPELESS;
        ud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        ud.Buffer.FirstElement = 0;
        ud.Buffer.NumElements = (UINT)(bytes / 4);
        ud.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
        if (FAILED(device->CreateUnorderedAccessView(dsOutBuf.Get(), &ud, &dsOutUav))) return false;
        D3D11_BUFFER_DESC sd = {};
        sd.ByteWidth = (UINT)bytes;
        sd.Usage = D3D11_USAGE_STAGING;
        sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        if (FAILED(device->CreateBuffer(&sd, nullptr, &dsOutStaging))) return false;
        dsOutBufSize = bytes;
        return true;
    }

    bool EnsureScShader() {
        if (scShader && scI420Shader) return true;
        ComPtr<ID3DBlob> blob, errBlob;
        HRESULT hr = D3DCompile(kScaleConvertHLSL, strlen(kScaleConvertHLSL), "scaleconvert.hlsl", nullptr, nullptr, "SCMain", "cs_5_0", 0, 0, &blob, &errBlob);
        if (FAILED(hr)) return false;
        if (FAILED(device->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &scShader))) return false;
        ComPtr<ID3DBlob> blob2, errBlob2;
        hr = D3DCompile(kScaleI420HLSL, strlen(kScaleI420HLSL), "scalei420.hlsl", nullptr, nullptr, "SCI420Main", "cs_5_0", 0, 0, &blob2, &errBlob2);
        if (FAILED(hr)) return false;
        if (FAILED(device->CreateComputeShader(blob2->GetBufferPointer(), blob2->GetBufferSize(), nullptr, &scI420Shader))) return false;
        if (scParamsCb) return true;
        D3D11_BUFFER_DESC cb = {};
        cb.ByteWidth = sizeof(ScParams);
        cb.Usage = D3D11_USAGE_DEFAULT;
        cb.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        return SUCCEEDED(device->CreateBuffer(&cb, nullptr, &scParamsCb));
    }

    bool EnsureTargetSlot(TargetSlot& slot, size_t bytes) {
        if (slot.buf && slot.bytes == bytes) return true;
        slot.buf.Reset(); slot.uav.Reset(); slot.staging.Reset();
        slot.bytes = 0;
        D3D11_BUFFER_DESC bd = {};
        bd.ByteWidth = (UINT)bytes;
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        if (FAILED(device->CreateBuffer(&bd, nullptr, &slot.buf))) return false;
        D3D11_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.Format = DXGI_FORMAT_R32_TYPELESS;
        ud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        ud.Buffer.FirstElement = 0;
        ud.Buffer.NumElements = (UINT)(bytes / 4);
        ud.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
        if (FAILED(device->CreateUnorderedAccessView(slot.buf.Get(), &ud, &slot.uav))) return false;
        D3D11_BUFFER_DESC sd = {};
        sd.ByteWidth = (UINT)bytes;
        sd.Usage = D3D11_USAGE_STAGING;
        sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        if (FAILED(device->CreateBuffer(&sd, nullptr, &slot.staging))) return false;
        slot.bytes = bytes;
        return true;
    }

    // Upload one video frame into this context's own texture, on the CALLING thread and this device — so
    // live input reaches the output without passing through the browser's GPU thread. UYVY is stored as a
    // half-width RGBA texture and decoded in the shader.
    bool SetVideoLayer(const osrcap::VideoLayer& video, std::string& err) {
        videoOn = false;
        if (!video.w || !video.h || !video.data) return true;
        const uint32_t texels = video.format == 1 ? video.w / 2 : video.w;
        const uint32_t rowBytes = texels * 4;
        if ((size_t)rowBytes * video.h > video.bytes) { err = "video layer buffer too small"; return false; }

        if (!videoTex || videoTexW != texels || videoTexH != video.h) {
            videoSrv.Reset();
            videoTex.Reset();
            D3D11_TEXTURE2D_DESC td = {};
            td.Width = texels;
            td.Height = video.h;
            td.MipLevels = 1;
            td.ArraySize = 1;
            td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            td.SampleDesc.Count = 1;
            // DEFAULT + UpdateSubresource rather than DYNAMIC + Map: the same one copy, and it is the
            // upload path the D3D11On12 device (the backend in use here) supports
            td.Usage = D3D11_USAGE_DEFAULT;
            td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            if (FAILED(device->CreateTexture2D(&td, nullptr, &videoTex))) { err = "video texture creation failed"; return false; }
            if (FAILED(device->CreateShaderResourceView(videoTex.Get(), nullptr, &videoSrv))) { err = "video SRV creation failed"; return false; }
            videoTexW = texels;
            videoTexH = video.h;
        }

        context->UpdateSubresource(videoTex.Get(), 0, nullptr, video.data, rowBytes, 0);
        videoOn = true;
        videoW = video.w;
        videoH = video.h;
        videoFormat = video.format;
        return true;
    }

    // the video SRV to bind at t1 (a 1x1 dummy when there is no layer, so the slot is always valid)
    ID3D11ShaderResourceView* VideoSrv() {
        if (videoOn && videoSrv) return videoSrv.Get();
        if (!dummySrv) {
            D3D11_TEXTURE2D_DESC td = {};
            td.Width = td.Height = 1;
            td.MipLevels = td.ArraySize = 1;
            td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            td.SampleDesc.Count = 1;
            td.Usage = D3D11_USAGE_DEFAULT;
            td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            const uint32_t black = 0;
            D3D11_SUBRESOURCE_DATA init = { &black, 4, 0 };
            ComPtr<ID3D11Texture2D> tex;
            if (FAILED(device->CreateTexture2D(&td, &init, &tex))) return nullptr;
            if (FAILED(device->CreateShaderResourceView(tex.Get(), nullptr, &dummySrv))) return nullptr;
            dummyTex = tex;
        }
        return dummySrv.Get();
    }

    // Queue one target's downscale+convert into its slot's staging buffer (the caller's GPU wait covers it).
    bool ScaleConvertToStaging(ID3D11Texture2D* shared, uint32_t srcW, uint32_t srcH, const osrcap::TargetSpec& t, TargetSlot& slot, std::string& err) {
        if (!EnsureScShader()) { err = "scale-convert shader init failed"; return false; }
        // 4-byte raw stores: round the buffer up so the last alpha word of a UYVA target fits
        size_t bytes = (osrcap::TargetBytes(t) + 3) & ~(size_t)3;
        if (!EnsureTargetSlot(slot, bytes)) { err = "target buffer creation failed"; return false; }
        ComPtr<ID3D11ShaderResourceView> srv;
        D3D11_SHADER_RESOURCE_VIEW_DESC sv = {};
        sv.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        sv.Texture2D.MipLevels = 1;
        if (FAILED(device->CreateShaderResourceView(shared, &sv, &srv))) { err = "target SRV creation failed"; return false; }
        ScParams p{ srcW, srcH, t.w, t.h, (uint32_t)t.format, (uint32_t)((size_t)t.w * 2 * t.h), 0, 0, videoOn ? 1u : 0u, videoW, videoH, (uint32_t)videoFormat };
        context->UpdateSubresource(scParamsCb.Get(), 0, nullptr, &p, 0, 0);
        ID3D11ShaderResourceView* srvs[] = { srv.Get(), VideoSrv() };
        ID3D11UnorderedAccessView* uavs[] = { slot.uav.Get() };
        ID3D11Buffer* cbs[] = { scParamsCb.Get() };
        const bool i420 = t.format == 4;
        context->CSSetShader(i420 ? scI420Shader.Get() : scShader.Get(), nullptr, 0);
        context->CSSetShaderResources(0, 2, srvs);
        context->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
        context->CSSetConstantBuffers(0, 1, cbs);
        if (i420) context->Dispatch((t.w + 7) / 8, (t.h + 1) / 2, 1);
        else context->Dispatch(((t.w + 1) / 2 + 7) / 8, (t.h + 7) / 8, 1);
        ID3D11ShaderResourceView* nullSrv[] = { nullptr, nullptr };
        ID3D11UnorderedAccessView* nullUav[] = { nullptr };
        context->CSSetShaderResources(0, 2, nullSrv);
        context->CSSetUnorderedAccessViews(0, 1, nullUav, nullptr);
        context->CopyResource(slot.staging.Get(), slot.buf.Get());
        slot.pending = osrcap::TargetBytes(t);
        return true;
    }

    // GPU box-downscale `shared` (BGRA) -> dsOutStaging (small tightly-packed BGRA). Queues GPU work only;
    // the caller's WaitGpu() covers completion, and ReadPending() maps dsOutStaging out.
    bool DownscaleToStaging(ID3D11Texture2D* shared, uint32_t srcW, uint32_t srcH, uint32_t dstW, uint32_t dstH, std::string& err) {
        if (!EnsureDsShader()) { err = "downscale shader init failed"; return false; }
        if (!EnsureDsOutBuffer((size_t)dstW * dstH * 4)) { err = "downscale output buffer creation failed"; return false; }
        ComPtr<ID3D11ShaderResourceView> srv;
        D3D11_SHADER_RESOURCE_VIEW_DESC sv = {};
        sv.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        sv.Texture2D.MipLevels = 1;
        if (FAILED(device->CreateShaderResourceView(shared, &sv, &srv))) { err = "downscale SRV creation failed"; return false; }
        DsParams p{ srcW, srcH, dstW, dstH };
        context->UpdateSubresource(dsParamsCb.Get(), 0, nullptr, &p, 0, 0);
        ID3D11ShaderResourceView* srvs[] = { srv.Get() };
        ID3D11UnorderedAccessView* uavs[] = { dsOutUav.Get() };
        ID3D11Buffer* cbs[] = { dsParamsCb.Get() };
        context->CSSetShader(dsShader.Get(), nullptr, 0);
        context->CSSetShaderResources(0, 1, srvs);
        context->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
        context->CSSetConstantBuffers(0, 1, cbs);
        context->Dispatch((dstW + 7) / 8, (dstH + 7) / 8, 1);
        ID3D11ShaderResourceView* nullSrv[] = { nullptr };
        ID3D11UnorderedAccessView* nullUav[] = { nullptr };
        context->CSSetShaderResources(0, 1, nullSrv);
        context->CSSetUnorderedAccessViews(0, 1, nullUav, nullptr);
        context->CopyResource(dsOutStaging.Get(), dsOutBuf.Get());
        return true;
    }

    bool ReadbackBgra(ID3D11Texture2D* shared, uint32_t width, uint32_t height, std::vector<uint8_t>& out, std::string& err) {
        if (!EnsureStaging(width, height)) { err = "staging texture creation failed"; return false; }
        context->CopyResource(staging.Get(), shared);
        D3D11_MAPPED_SUBRESOURCE map = {};
        if (FAILED(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &map))) { err = "Map failed"; return false; }
        const size_t rowBytes = (size_t)width * 4;
        out.resize(rowBytes * height);
        const uint8_t* src = (const uint8_t*)map.pData;
        // stream-copy each row: map.pData / RowPitch are 16-byte aligned (WC memory) -> use MOVNTDQA
        for (UINT y = 0; y < height; ++y) StreamCopyFromWC(out.data() + (size_t)y * rowBytes, src + (size_t)y * map.RowPitch, rowBytes);
        context->Unmap(staging.Get(), 0);
        return true;
    }

    bool ReadbackConvert(ID3D11Texture2D* shared, uint32_t width, uint32_t height, bool alpha, std::vector<uint8_t>& out, std::string& err) {
        InvalidateOn12OutBuffer();  // classic-staging consumer: never reuse an on12-wrapped outBuf
        if (!EnsureShader()) { err = "compute shader init failed"; return false; }
        const size_t uyvySize = (size_t)width * 2 * height;
        const size_t total = alpha ? uyvySize + (size_t)width * height : uyvySize;
        if (!EnsureOutBuffer(total)) { err = "output buffer creation failed"; return false; }

        ComPtr<ID3D11ShaderResourceView> srv;
        D3D11_SHADER_RESOURCE_VIEW_DESC sv = {};
        sv.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        sv.Texture2D.MipLevels = 1;
        if (FAILED(device->CreateShaderResourceView(shared, &sv, &srv))) { err = "SRV creation failed"; return false; }

        Params p{ width, height, (uint32_t)uyvySize, alpha ? 1u : 0u };
        context->UpdateSubresource(paramsCb.Get(), 0, nullptr, &p, 0, 0);

        ID3D11ShaderResourceView* srvs[] = { srv.Get() };
        ID3D11UnorderedAccessView* uavs[] = { outUav.Get() };
        ID3D11Buffer* cbs[] = { paramsCb.Get() };
        context->CSSetShader(shader.Get(), nullptr, 0);
        context->CSSetShaderResources(0, 1, srvs);
        context->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
        context->CSSetConstantBuffers(0, 1, cbs);
        const UINT gx = (width / 4 + 15) / 16;
        const UINT gy = (height + 15) / 16;
        context->Dispatch(gx, gy, 1);

        ID3D11ShaderResourceView* nullSrv[] = { nullptr };
        ID3D11UnorderedAccessView* nullUav[] = { nullptr };
        context->CSSetShaderResources(0, 1, nullSrv);
        context->CSSetUnorderedAccessViews(0, 1, nullUav, nullptr);

        context->CopyResource(outStaging.Get(), outBuf.Get());
        D3D11_MAPPED_SUBRESOURCE map = {};
        // Map blocks until the GPU has finished the dispatch + copy
        if (FAILED(context->Map(outStaging.Get(), 0, D3D11_MAP_READ, 0, &map))) { err = "Map failed"; return false; }
        out.resize(total);
        ParallelStreamCopyFromWC(out.data(), map.pData, total, 8);
        context->Unmap(outStaging.Get(), 0);
        return true;
    }

    // GPU BGRA->RGBA swizzle readback (format 3, WebRTC). Same size as BGRA; the swizzle runs on the GPU so the
    // main process never does the channel swap. Mirrors ReadbackConvert but outputs full-res RGBA.
    bool ReadbackRgba(ID3D11Texture2D* shared, uint32_t width, uint32_t height, std::vector<uint8_t>& out, std::string& err) {
        InvalidateOn12OutBuffer();  // classic-staging consumer: never reuse an on12-wrapped outBuf
        if (!EnsureSwizzleShader()) { err = "swizzle shader init failed"; return false; }
        const size_t total = (size_t)width * 4 * height;
        if (!EnsureOutBuffer(total)) { err = "output buffer creation failed"; return false; }

        ComPtr<ID3D11ShaderResourceView> srv;
        D3D11_SHADER_RESOURCE_VIEW_DESC sv = {};
        sv.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        sv.Texture2D.MipLevels = 1;
        if (FAILED(device->CreateShaderResourceView(shared, &sv, &srv))) { err = "SRV creation failed"; return false; }

        SwParams p{ width, height };
        context->UpdateSubresource(swizzleParamsCb.Get(), 0, nullptr, &p, 0, 0);
        ID3D11ShaderResourceView* srvs[] = { srv.Get() };
        ID3D11UnorderedAccessView* uavs[] = { outUav.Get() };
        ID3D11Buffer* cbs[] = { swizzleParamsCb.Get() };
        context->CSSetShader(swizzleShader.Get(), nullptr, 0);
        context->CSSetShaderResources(0, 1, srvs);
        context->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
        context->CSSetConstantBuffers(0, 1, cbs);
        context->Dispatch((width + 7) / 8, (height + 7) / 8, 1);

        ID3D11ShaderResourceView* nullSrv[] = { nullptr };
        ID3D11UnorderedAccessView* nullUav[] = { nullptr };
        context->CSSetShaderResources(0, 1, nullSrv);
        context->CSSetUnorderedAccessViews(0, 1, nullUav, nullptr);

        context->CopyResource(outStaging.Get(), outBuf.Get());
        D3D11_MAPPED_SUBRESOURCE map = {};
        if (FAILED(context->Map(outStaging.Get(), 0, D3D11_MAP_READ, 0, &map))) { err = "Map failed"; return false; }
        out.resize(total);
        ParallelStreamCopyFromWC(out.data(), map.pData, total, 8);
        context->Unmap(outStaging.Get(), 0);
        return true;
    }

    // Single-phase readback (mac/linux-style; Windows fallback + probe callers). Stays on the CLASSIC
    // staging path for all formats — on the 11On12 device classic buffers/staging work unchanged; the
    // Stage-2 READBACK-heap copy-out is wired only into the two-phase convert path (the NDI hot path).
    bool Readback(uintptr_t handle, uint32_t width, uint32_t height, int format, std::vector<uint8_t>& out, std::string& err) {
        ComPtr<ID3D11Texture2D> shared;
        if (!OpenSharedTex(handle, shared, err)) return false;

        ComPtr<IDXGIKeyedMutex> keyedMutex;
        bool haveKeyedMutex = SUCCEEDED(shared.As(&keyedMutex)) && keyedMutex;
        if (haveKeyedMutex && keyedMutex->AcquireSync(0, 1000) != S_OK) { err = "keyed mutex AcquireSync failed/timed out"; return false; }

        bool ok;
        if (format == 1 || format == 2) ok = ReadbackConvert(shared.Get(), width, height, format == 2, out, err);
        else if (format == 3) ok = ReadbackRgba(shared.Get(), width, height, out, err);
        else ok = ReadbackBgra(shared.Get(), width, height, out, err);

        if (haveKeyedMutex) keyedMutex->ReleaseSync(0);
        return ok;
    }

    // ---- two-phase: consume (GPU, releases the shared texture fast) then read (slow PCIe copy) ----

    // Block until the GPU has finished all queued work (so the shared texture has been fully read).
    void WaitGpu() {
        if (!flushQuery) {
            D3D11_QUERY_DESC qd = {};
            qd.Query = D3D11_QUERY_EVENT;
            device->CreateQuery(&qd, &flushQuery);
        }
        context->End(flushQuery.Get());
        context->Flush();
        while (context->GetData(flushQuery.Get(), nullptr, 0, 0) != S_OK) std::this_thread::yield();
    }

    // GPU-convert the shared texture into outBuf (queues the compute dispatch only — no staging copy, no CPU
    // read). Split out of ConvertToStaging so the Stage-2 D3D11On12 path (test/on12_harness.cc today, the
    // readback_win 11On12 path in sub-step B) can run the EXACT same convert and then hand outBuf to the D3D12
    // copy queue instead of the WC staging buffer. The live path below is unchanged: ConvertToStaging =
    // DispatchConvert + the staging CopyResource, exactly as before.
    bool DispatchConvert(ID3D11Texture2D* shared, uint32_t width, uint32_t height, bool alpha, std::string& err) {
        if (!EnsureShader()) { err = "compute shader init failed"; return false; }
        const size_t uyvySize = (size_t)width * 2 * height;
        const size_t total = alpha ? uyvySize + (size_t)width * height : uyvySize;
        if (!EnsureOutBuffer(total)) { err = "output buffer creation failed"; return false; }

        ComPtr<ID3D11ShaderResourceView> srv;
        D3D11_SHADER_RESOURCE_VIEW_DESC sv = {};
        sv.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        sv.Texture2D.MipLevels = 1;
        if (FAILED(device->CreateShaderResourceView(shared, &sv, &srv))) { err = "SRV creation failed"; return false; }

        Params p{ width, height, (uint32_t)uyvySize, alpha ? 1u : 0u, videoOn ? 1u : 0u, videoW, videoH, (uint32_t)videoFormat };
        context->UpdateSubresource(paramsCb.Get(), 0, nullptr, &p, 0, 0);
        ID3D11ShaderResourceView* srvs[] = { srv.Get(), VideoSrv() };
        ID3D11UnorderedAccessView* uavs[] = { outUav.Get() };
        ID3D11Buffer* cbs[] = { paramsCb.Get() };
        context->CSSetShader(shader.Get(), nullptr, 0);
        context->CSSetShaderResources(0, 2, srvs);
        context->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
        context->CSSetConstantBuffers(0, 1, cbs);
        context->Dispatch((width / 4 + 15) / 16, (height + 15) / 16, 1);
        ID3D11ShaderResourceView* nullSrv[] = { nullptr, nullptr };
        ID3D11UnorderedAccessView* nullUav[] = { nullptr };
        context->CSSetShaderResources(0, 2, nullSrv);
        context->CSSetUnorderedAccessViews(0, 1, nullUav, nullptr);
        return true;
    }

    // GPU-convert the shared texture into outStaging (no CPU read yet). Mirrors ReadbackConvert up to the copy.
    bool ConvertToStaging(ID3D11Texture2D* shared, uint32_t width, uint32_t height, bool alpha, std::string& err) {
        InvalidateOn12OutBuffer();  // classic-staging consumer: never reuse an on12-wrapped outBuf
        if (!DispatchConvert(shared, width, height, alpha, err)) return false;
        context->CopyResource(outStaging.Get(), outBuf.Get());
        return true;
    }

    // Phase 1: open + GPU-convert into staging, then WAIT for the GPU so the caller can release the shared
    // texture immediately (short — ~the GPU time). The slow PCIe read is deferred to ReadPending.
    // dstW/dstH > 0 additionally GPU-downscales the shared BGRA to a small buffer (for server/stage) in the
    // SAME GPU pass, so a mixed output reads back UYVY (NDI) + a few-MB scaled BGRA instead of the full frame.
    bool ConsumeShared(uintptr_t handle, uint32_t width, uint32_t height, int format, uint32_t dstW, uint32_t dstH, std::string& err) {
        return ConsumeShared(handle, width, height, format, dstW, dstH, std::vector<osrcap::TargetSpec>(), err);
    }

    bool ConsumeShared(uintptr_t handle, uint32_t width, uint32_t height, int format, uint32_t dstW, uint32_t dstH, const std::vector<osrcap::TargetSpec>& targets, std::string& err) {
        return ConsumeShared(handle, width, height, format, dstW, dstH, targets, osrcap::VideoLayer(), err);
    }

    bool ConsumeShared(uintptr_t handle, uint32_t width, uint32_t height, int format, uint32_t dstW, uint32_t dstH, const std::vector<osrcap::TargetSpec>& targets, const osrcap::VideoLayer& video, std::string& err) {
        ComPtr<ID3D11Texture2D> shared;
        if (!OpenSharedTex(handle, shared, err)) return false;
        // after OpenSharedTex: it creates this context's device on first use
        if (!SetVideoLayer(video, err)) return false;

        ComPtr<IDXGIKeyedMutex> keyedMutex;
        bool haveKeyedMutex = SUCCEEDED(shared.As(&keyedMutex)) && keyedMutex;
        if (haveKeyedMutex && keyedMutex->AcquireSync(0, 1000) != S_OK) { err = "keyed mutex AcquireSync failed/timed out"; return false; }

        bool ok = true;
        pendingOn12 = false;
        if (format == 1 || format == 2) {
            pendingIsConvert = true;
            const size_t uyvySize = (size_t)width * 2 * height;
            pendingTotal = (format == 2) ? uyvySize + (size_t)width * height : uyvySize;
            if (on12Active && EnsureOn12OutBuffer(pendingTotal)) {
                // Stage-2 path: the UNCHANGED DispatchConvert (kConvertHLSL, same UAV desc) writes into the
                // wrapped D3D12 buffer — EnsureOutBuffer inside it early-returns on the injected on12 pair.
                pendingOn12 = true;
                ID3D11Resource* wrapped[] = {outBuf.Get()};
                on12->AcquireWrappedResources(wrapped, 1);
                ok = DispatchConvert(shared.Get(), width, height, format == 2, err);
                on12->ReleaseWrappedResources(wrapped, 1);
            } else {
                // classic WC-staging path — also the per-frame fallback when the on12 buffers can't be built
                ok = ConvertToStaging(shared.Get(), width, height, format == 2, err);
            }
        } else {
            // BGRA (multi-consumer / non-off-main): no split benefit; do the full read now into a vector.
            pendingIsConvert = false;
            ok = ReadbackBgra(shared.Get(), width, height, pendingBgra, err);
            pendingTotal = pendingBgra.size();
        }
        // optional GPU downscale for server/stage, queued before the single GPU wait below (small WC copy-out;
        // its D3D11 objects live on whichever device this context has — unchanged on 11On12)
        pendingScaledTotal = 0;
        if (ok && dstW > 0 && dstH > 0) {
            if (DownscaleToStaging(shared.Get(), width, height, dstW, dstH, err)) pendingScaledTotal = (size_t)dstW * dstH * 4;
            else ok = false;
        }
        // per-consumer targets, queued in the same pass before the single GPU wait below
        for (auto& slot : targetSlots) slot.pending = 0;
        if (ok && !targets.empty()) {
            if (targetSlots.size() < targets.size()) targetSlots.resize(targets.size());
            for (size_t i = 0; i < targets.size() && ok; ++i) ok = ScaleConvertToStaging(shared.Get(), width, height, targets[i], targetSlots[i], err);
        }
        // Ensure the GPU is done reading `shared` before the caller releases it.
        if (on12Active) {
            // Harness-proven single-device sequence: submit all queued 11On12 work to the DIRECT queue, fence
            // it, and (convert path) queue the COPY-queue DMA into the READBACK heap NOW so it overlaps the
            // caller's turnaround — its completion (pendingCopyFence) is waited in ReadPending. The CPU block
            // here is an EVENT wait on the direct-queue fence — the WaitGpu busy-spin is gone on this path.
            context->Flush();
            const UINT64 vConv = ++d12FenceVal;
            d12Direct->Signal(d12Fence.Get(), vConv);
            if (ok && pendingOn12) {
                d12Copy->Wait(d12Fence.Get(), vConv);
                d12CopyAlloc->Reset();
                d12CopyList->Reset(d12CopyAlloc.Get(), nullptr);
                d12CopyList->CopyBufferRegion(rbBuf.Get(), 0, convBuf12.Get(), 0, pendingTotal);
                d12CopyList->Close();
                ID3D12CommandList* lists[] = {d12CopyList.Get()};
                d12Copy->ExecuteCommandLists(1, lists);
                pendingCopyFence = ++d12FenceVal;
                d12Copy->Signal(d12Fence.Get(), pendingCopyFence);
            }
            if (!WaitFenceOn12(vConv, 2000)) {
                ok = false;
                if (err.empty()) err = "d3d11on12 convert fence watchdog";
            }
        } else {
            WaitGpu();
        }
        if (haveKeyedMutex) keyedMutex->ReleaseSync(0);
        return ok;
    }

    // SINGLE DISPATCH (plan §11 fix #1): the whole readback on this one context in one call — Consume (open +
    // GPU-convert + WaitGpu, releasing the shared texture) then ReadPending (the slow copy-out) — with
    // `onGpuDone` fired in between, the instant the GPU is done with the shared texture and the keyed mutex is
    // released (identical release timing to the old two-phase Consume return), so the caller releases the
    // Electron texture just as early WITHOUT a second N-API dispatch or the JS-loop hop between the phases.
    bool ReadbackOnceOnCtx(uintptr_t handle, uint32_t width, uint32_t height, int format, uint32_t dstW, uint32_t dstH,
                           uint8_t* dst, size_t dstSize, uint8_t* scaledDst, size_t scaledSize,
                           const std::function<void()>& onGpuDone, std::string& err) {
        if (!ConsumeShared(handle, width, height, format, dstW, dstH, err)) return false;
        // GPU has finished reading the shared texture and the keyed mutex is released -> safe to release the
        // Electron texture now (before the slow copy-out below), exactly as the two-phase Consume did.
        if (onGpuDone) onGpuDone();
        return ReadPending(dst, dstSize, scaledDst, scaledSize, err);
    }

    // Phase 2: copy the pending result into `dst` (the slow WC/PCIe read for the convert path). When a GPU
    // downscale was requested, also copy the small scaled BGRA into `scaledDst` (cheap — only a few MB).
    bool ReadPending(uint8_t* dst, size_t dstSize, uint8_t* scaledDst, size_t scaledSize, std::string& err) {
        return ReadPending(dst, dstSize, scaledDst, scaledSize, std::vector<osrcap::TargetDst>(), err);
    }

    bool ReadPending(uint8_t* dst, size_t dstSize, uint8_t* scaledDst, size_t scaledSize, const std::vector<osrcap::TargetDst>& targetDsts, std::string& err) {
        size_t n = pendingTotal < dstSize ? pendingTotal : dstSize;
        if (!pendingIsConvert) {
            std::memcpy(dst, pendingBgra.data(), n);
            pendingBgra.clear();
        } else if (pendingOn12) {
            // Stage-2 copy-out: wait the COPY-queue fence (event block; the DMA was queued in ConsumeShared so
            // it overlapped the dispatch turnaround), then copy from the PERSISTENTLY-MAPPED CPU-cached
            // READBACK heap. Routed through the same self-calibrating CopyPool as the WC path — the pool's
            // [COPY-POOL] MB/s telemetry therefore shows the copy-out source speed directly (cached-class GB/s
            // vs WC-class ~1.5 GB/s), and its knee-seeker re-calibrates to the cached regime on its own.
            // (MOVNTDQA on WB/cached memory is an ordinary load — the NT hint only affects WC.)
            if (!WaitFenceOn12(pendingCopyFence, 2000)) { err = "d3d11on12 copy fence watchdog"; return false; }
            pendingCopyFence = 0;
            ParallelStreamCopyFromWC(dst, rbPtr, n, 8);
        } else {
            D3D11_MAPPED_SUBRESOURCE map = {};
            if (FAILED(context->Map(outStaging.Get(), 0, D3D11_MAP_READ, 0, &map))) { err = "Map failed"; return false; }
            ParallelStreamCopyFromWC(dst, map.pData, n, 8);
            context->Unmap(outStaging.Get(), 0);
        }
        if (pendingScaledTotal > 0 && scaledDst && scaledSize > 0) {
            size_t sn = pendingScaledTotal < scaledSize ? pendingScaledTotal : scaledSize;
            D3D11_MAPPED_SUBRESOURCE smap = {};
            if (FAILED(context->Map(dsOutStaging.Get(), 0, D3D11_MAP_READ, 0, &smap))) { err = "scaled Map failed"; return false; }
            ParallelStreamCopyFromWC(scaledDst, smap.pData, sn, 4);
            context->Unmap(dsOutStaging.Get(), 0);
        }
        pendingScaledTotal = 0;
        for (size_t i = 0; i < targetDsts.size() && i < targetSlots.size(); ++i) {
            TargetSlot& slot = targetSlots[i];
            if (!slot.pending || !targetDsts[i].data || !targetDsts[i].size) continue;
            size_t tn = slot.pending < targetDsts[i].size ? slot.pending : targetDsts[i].size;
            D3D11_MAPPED_SUBRESOURCE tmap = {};
            if (FAILED(context->Map(slot.staging.Get(), 0, D3D11_MAP_READ, 0, &tmap))) { err = "target Map failed"; return false; }
            ParallelStreamCopyFromWC(targetDsts[i].data, tmap.pData, tn, 4);
            context->Unmap(slot.staging.Get(), 0);
            slot.pending = 0;
        }
        return true;
    }
};

constexpr size_t kMaxPool = 16;
std::vector<std::unique_ptr<ReadbackContext>> g_pool;
std::mutex g_poolMutex;
std::condition_variable g_poolCv;

ReadbackContext* AcquireContext() {
    std::unique_lock<std::mutex> lock(g_poolMutex);
    for (;;) {
        for (auto& c : g_pool) {
            if (!c->inUse) { c->inUse = true; return c.get(); }
        }
        if (g_pool.size() < kMaxPool) {
            g_pool.push_back(std::make_unique<ReadbackContext>());
            g_pool.back()->inUse = true;
            return g_pool.back().get();
        }
        g_poolCv.wait(lock);
    }
}

void ReleaseContext(ReadbackContext* ctx) {
    {
        std::lock_guard<std::mutex> lock(g_poolMutex);
        ctx->inUse = false;
    }
    g_poolCv.notify_one();
}

// contexts holding a consumed-but-not-yet-read frame, keyed by the caller's key (output id)
std::map<std::string, ReadbackContext*> g_pending;
std::mutex g_pendingMutex;

}  // namespace

bool ReadbackHandle(uintptr_t handle, uint32_t width, uint32_t height, int format, std::vector<uint8_t>& out, std::string& err) {
    ReadbackContext* ctx = AcquireContext();
    bool ok = ctx->Readback(handle, width, height, format, out, err);
    ReleaseContext(ctx);
    return ok;
}

bool TargetsSupported() { return true; }

bool ReadbackConsume(uintptr_t handle, uint32_t width, uint32_t height, int format, const std::string& key, uint32_t dstW, uint32_t dstH, std::string& err) {
    return ReadbackConsume(handle, width, height, format, key, dstW, dstH, std::vector<TargetSpec>(), err);
}
bool ReadbackFinish(const std::string& key, uint8_t* dst, size_t dstSize, uint8_t* scaledDst, size_t scaledSize, std::string& err) {
    return ReadbackFinish(key, dst, dstSize, scaledDst, scaledSize, std::vector<TargetDst>(), err);
}

bool ReadbackConsume(uintptr_t handle, uint32_t width, uint32_t height, int format, const std::string& key, uint32_t dstW, uint32_t dstH, const std::vector<TargetSpec>& targets, std::string& err) {
    return ReadbackConsume(handle, width, height, format, key, dstW, dstH, targets, VideoLayer(), err);
}

bool VideoLayerSupported() { return true; }

bool ReadbackConsume(uintptr_t handle, uint32_t width, uint32_t height, int format, const std::string& key, uint32_t dstW, uint32_t dstH, const std::vector<TargetSpec>& targets, const VideoLayer& video, std::string& err) {
    ReadbackContext* ctx = AcquireContext();
    bool ok = ctx->ConsumeShared(handle, width, height, format, dstW, dstH, targets, video, err);
    if (!ok) {
        ReleaseContext(ctx);
        return false;
    }
    std::lock_guard<std::mutex> lock(g_pendingMutex);
    auto it = g_pending.find(key);
    if (it != g_pending.end()) ReleaseContext(it->second);  // stale (shouldn't happen with single-in-flight)
    g_pending[key] = ctx;
    return true;
}

bool ReadbackFinish(const std::string& key, uint8_t* dst, size_t dstSize, uint8_t* scaledDst, size_t scaledSize, const std::vector<TargetDst>& targetDsts, std::string& err) {
    ReadbackContext* ctx = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_pendingMutex);
        auto it = g_pending.find(key);
        if (it == g_pending.end()) {
            err = "no pending readback for key";
            return false;
        }
        ctx = it->second;
        g_pending.erase(it);
    }
    bool ok = ctx->ReadPending(dst, dstSize, scaledDst, scaledSize, targetDsts, err);
    ReleaseContext(ctx);
    return ok;
}

void ReadbackReleaseKey(const std::string& key) {
    std::lock_guard<std::mutex> lock(g_pendingMutex);
    auto it = g_pending.find(key);
    if (it != g_pending.end()) {
        ReleaseContext(it->second);
        g_pending.erase(it);
    }
}

// SINGLE-DISPATCH readback (plan §11 fix #1): borrow ONE context for the whole open+convert+wait+copy-out and
// release it at the end — no g_pending map (the context never leaves this call), so two concurrent captures
// for one output can't collide on a key and there is no cross-key cleanup path. `onGpuDone` fires once,
// between the GPU wait and the copy-out, on this (worker) thread.
bool ReadbackOnce(uintptr_t handle, uint32_t width, uint32_t height, int format, uint32_t dstW, uint32_t dstH,
                  uint8_t* dst, size_t dstSize, uint8_t* scaledDst, size_t scaledSize,
                  const std::function<void()>& onGpuDone, std::string& err) {
    ReadbackContext* ctx = AcquireContext();
    bool ok = ctx->ReadbackOnceOnCtx(handle, width, height, format, dstW, dstH, dst, dstSize, scaledDst, scaledSize, onGpuDone, err);
    ReleaseContext(ctx);
    return ok;
}

unsigned CopyPoolActiveWorkers() { return CopyPool::Get().ActiveWorkers(); }

// Stage-2 sub-step B telemetry ("observability, not knobs"): which copy-out backend the process is on.
// "none" until the first device init; "d3d11" is authoritative once 11On12 is disabled (ladder fallback).
const char* ReadbackBackend() {
    if (g_on12Disabled.load()) return "d3d11";
    switch (g_backendState.load()) {
        case 1: return "d3d11on12";
        case 2: return "d3d11";
        default: return On12Allowed() ? "none" : "d3d11";
    }
}

bool CopyPoolSelfTest(uint32_t bytes, uint32_t concurrency, uint32_t iterations, uint32_t& workers, uint32_t& mismatches, double& ms) {
    std::atomic<uint32_t> mism{0};
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> callers;
    for (uint32_t c = 0; c < concurrency; ++c) {
        callers.emplace_back([&, c]() {
            // 16-byte-aligned src (MOVNTDQA requires it), mirroring the real WC-mapped source.
            std::vector<__m128i> srcAligned((bytes + 15) / 16);
            uint8_t* src = reinterpret_cast<uint8_t*>(srcAligned.data());
            std::vector<uint8_t> dst(bytes);
            for (uint32_t i = 0; i < bytes; ++i) src[i] = static_cast<uint8_t>((i * 131u + c * 17u) & 0xffu);
            for (uint32_t it = 0; it < iterations; ++it) {
                std::memset(dst.data(), 0, bytes);
                ParallelStreamCopyFromWC(dst.data(), src, bytes, 8);
                if (std::memcmp(dst.data(), src, bytes) != 0) mism.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& th : callers) th.join();
    ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    mismatches = mism.load();
    // Report the calibrated active_n AFTER the run (the converged value under this synthetic load). On
    // cacheable RAM the VALUE is meaningless for WC/BAR sizing — the test only validates convergence behavior.
    workers = CopyPool::Get().ActiveWorkers();
    return mismatches == 0;
}

}  // namespace osrcap
