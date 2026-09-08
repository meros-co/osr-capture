// osr-capture: read back an Electron OSR shared GPU texture into a CPU BGRA buffer, off the main thread.
//
// The GPU->CPU copy runs in an N-API AsyncWorker (a libuv background thread) so it never blocks the JS
// main thread. Per-platform readback lives in readback_{win,mac,linux}; this file is the N-API surface.
//
// Windows/macOS: readback(handle: Buffer /* uintptr_t */, width, height) => Promise<Buffer>
// Linux:         readback({ planes: {fd,stride,offset,size}[], modifier }, width, height) => Promise<Buffer>

#include <napi.h>

#include <cstring>
#include <map>
#include <string>

#include "osr_readback.h"

namespace {

// Output byte size per format: 0=BGRA (w*4*h), 1=UYVY (w*2*h), 2=UYVA (w*2*h + w*h = w*3*h).
size_t OutputSize(uint32_t w, uint32_t h, int format) {
    size_t px = static_cast<size_t>(w) * h;
    if (format == 1) return px * 2;
    if (format == 2) return px * 3;
    return px * 4;
}

// Per-output pool of REUSED V8 output buffers, keyed by the caller's poolKey (FreeShow passes the output id).
// Reusing buffers avoids a per-frame ~16MB V8 allocation + zero-fill on the main thread (which caused GC
// jitter), and the worker fills the buffer off-thread so OnOK does no copy. POOL_N buffers rotate so the
// worker fills slot R%N while JS still holds the previous slot: with single-in-flight readback per output,
// a buffer is not overwritten for N readbacks, well after JS has finished reading it (send copies it out).
constexpr int POOL_N = 3;
struct BufferPool {
    size_t size = 0;
    int next = 0;
    Napi::Reference<Napi::Buffer<uint8_t>> bufs[POOL_N];
};

// Per-ENV pool map, held as N-API instance data. The addon is loaded in BOTH the main thread and the NDI
// worker thread. A single process-global map was (a) a data race — both threads insert/erase concurrently —
// and (b) worse, the worker's releasePool() could destroy Napi::References that were CREATED on the main env
// (main-path readback uses a bare `id` key; the worker uses `id#slot`), and destroying a reference on a
// different env than created it is undefined behaviour. Per-env instance data gives each thread its own pools,
// finalized on that env's own teardown (on the right thread), so references are only ever created and
// destroyed on their owning env, with no shared mutable state to race on.
struct AddonData {
    std::map<std::string, BufferPool> pools;
};

#if defined(__linux__)
// Parse the Linux `source` shape ({ planes: {fd,stride,offset,size}[], modifier }) shared by readback /
// readbackConsume / readbackOnce.
void ParseLinuxSource(Napi::Value src, std::vector<osrcap::DmabufPlane>& planes, uint64_t& modifier) {
    planes.clear();
    modifier = 0;
    if (!src.IsObject()) return;
    Napi::Object arg = src.As<Napi::Object>();
    if (arg.Get("planes").IsArray()) {
        Napi::Array arr = arg.Get("planes").As<Napi::Array>();
        for (uint32_t i = 0; i < arr.Length(); ++i) {
            Napi::Object p = arr.Get(i).As<Napi::Object>();
            osrcap::DmabufPlane dp;
            dp.fd = p.Get("fd").As<Napi::Number>().Int32Value();
            dp.stride = p.Get("stride").As<Napi::Number>().Uint32Value();
            dp.offset = static_cast<uint64_t>(p.Get("offset").As<Napi::Number>().Int64Value());
            dp.size = static_cast<uint64_t>(p.Get("size").As<Napi::Number>().Int64Value());
            planes.push_back(dp);
        }
    }
    Napi::Value mod = arg.Get("modifier");
    if (mod.IsBigInt()) {
        bool lossless = false;
        modifier = mod.As<Napi::BigInt>().Uint64Value(&lossless);
    } else if (mod.IsNumber()) {
        modifier = static_cast<uint64_t>(mod.As<Napi::Number>().Int64Value());
    }
}
#endif

class ReadbackWorker : public Napi::AsyncWorker {
public:
#if defined(_WIN32) || defined(__APPLE__)
    ReadbackWorker(Napi::Env env, uintptr_t handle, uint32_t w, uint32_t h, int format, uint8_t* dst, size_t dstSize, Napi::Buffer<uint8_t> result, Napi::Promise::Deferred deferred)
        : Napi::AsyncWorker(env), handle_(handle), w_(w), h_(h), format_(format), dst_(dst), dstSize_(dstSize), resultRef_(Napi::Persistent(result)), deferred_(deferred) {}
#elif defined(__linux__)
    ReadbackWorker(Napi::Env env, std::vector<osrcap::DmabufPlane> planes, uint64_t modifier, uint32_t w, uint32_t h, int format, std::string poolKey, uint8_t* dst, size_t dstSize, Napi::Buffer<uint8_t> result, Napi::Promise::Deferred deferred)
        : Napi::AsyncWorker(env), planes_(std::move(planes)), modifier_(modifier), poolKey_(std::move(poolKey)), w_(w), h_(h), format_(format), dst_(dst), dstSize_(dstSize), resultRef_(Napi::Persistent(result)), deferred_(deferred) {}
#endif

    void Execute() override {
        std::string err;
        bool ok = false;
#if defined(_WIN32) || defined(__APPLE__)
        ok = osrcap::ReadbackHandle(handle_, w_, h_, format_, out_, err);
#elif defined(__linux__)
        ok = osrcap::ReadbackDmabuf(planes_, modifier_, w_, h_, format_, poolKey_, out_, err);
#else
        err = "unsupported platform";
#endif
        if (!ok) {
            SetError(err.empty() ? "readback failed" : err);
            return;
        }
        // Copy into the caller-owned V8 buffer HERE (libuv worker thread) so OnOK (main thread) does no copy —
        // the full-frame memcpy no longer congests the JS event loop.
        if (out_.size() != dstSize_) {
            SetError("readback produced " + std::to_string(out_.size()) + " bytes, expected " + std::to_string(dstSize_));
            return;
        }
        std::memcpy(dst_, out_.data(), out_.size());
    }

    void OnOK() override {
        Napi::HandleScope scope(Env());
        deferred_.Resolve(resultRef_.Value());
    }

    void OnError(const Napi::Error& e) override { deferred_.Reject(e.Value()); }

private:
#if defined(_WIN32) || defined(__APPLE__)
    uintptr_t handle_ = 0;
#elif defined(__linux__)
    std::vector<osrcap::DmabufPlane> planes_;
    uint64_t modifier_ = 0;
    std::string poolKey_;
#endif
    uint32_t w_ = 0;
    uint32_t h_ = 0;
    int format_ = 0;
    uint8_t* dst_ = nullptr;
    size_t dstSize_ = 0;
    Napi::Reference<Napi::Buffer<uint8_t>> resultRef_;
    Napi::Promise::Deferred deferred_;
    std::vector<uint8_t> out_;
};

// Acquire the output buffer for this readback: a rotating pooled (reused) buffer when a poolKey is given,
// else a fresh per-call buffer (probe/other callers). Reallocates a pool when the frame size changes.
Napi::Buffer<uint8_t> AcquireOutputBuffer(Napi::Env env, const std::string& poolKey, size_t outSize) {
    if (poolKey.empty()) return Napi::Buffer<uint8_t>::New(env, outSize);
    AddonData* data = env.GetInstanceData<AddonData>();
    if (!data) return Napi::Buffer<uint8_t>::New(env, outSize);
    BufferPool& pool = data->pools[poolKey];
    if (pool.size != outSize) {
        for (int i = 0; i < POOL_N; ++i) pool.bufs[i] = Napi::Persistent(Napi::Buffer<uint8_t>::New(env, outSize));
        pool.size = outSize;
        pool.next = 0;
    }
    Napi::Buffer<uint8_t> b = pool.bufs[pool.next].Value();
    pool.next = (pool.next + 1) % POOL_N;
    return b;
}

Napi::Value Readback(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 3) {
        Napi::TypeError::New(env, "readback(handleOrInfo, width, height)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    uint32_t w = info[1].As<Napi::Number>().Uint32Value();
    uint32_t h = info[2].As<Napi::Number>().Uint32Value();
    int format = (info.Length() > 3 && info[3].IsNumber()) ? info[3].As<Napi::Number>().Int32Value() : 0;
    std::string poolKey = (info.Length() > 4 && info[4].IsString()) ? info[4].As<Napi::String>().Utf8Value() : std::string();
    auto deferred = Napi::Promise::Deferred::New(env);

    // Acquire the (reused, when pooled) output buffer up front; the worker fills it off-thread.
    size_t outSize = OutputSize(w, h, format);
    Napi::Buffer<uint8_t> result = AcquireOutputBuffer(env, poolKey, outSize);
    uint8_t* dst = result.Data();

#if defined(_WIN32) || defined(__APPLE__)
    uintptr_t handle = 0;
    Napi::Buffer<uint8_t> buf = info[0].As<Napi::Buffer<uint8_t>>();
    if (buf.Length() >= sizeof(uintptr_t)) std::memcpy(&handle, buf.Data(), sizeof(uintptr_t));
    (new ReadbackWorker(env, handle, w, h, format, dst, outSize, result, deferred))->Queue();
#elif defined(__linux__)
    std::vector<osrcap::DmabufPlane> planes;
    uint64_t modifier = 0;
    ParseLinuxSource(info[0], planes, modifier);
    (new ReadbackWorker(env, std::move(planes), modifier, w, h, format, poolKey, dst, outSize, result, deferred))->Queue();
#else
    deferred.Reject(Napi::Error::New(env, "unsupported platform").Value());
#endif

    return deferred.Promise();
}

#if defined(_WIN32) || defined(__APPLE__) || defined(__linux__)
// Two-phase readback (all three GPU backends): consume opens+GPU-converts the shared texture / IOSurface /
// imported dmabuf and returns once the GPU has read it (so the caller can release the Electron texture);
// finish does the copy into a pooled buffer. This keeps the shared texture pinned only for the GPU phase,
// not the whole readback. On Windows/Linux the copy is the slow bus read; on macOS unified memory it is a
// cached memcpy, but the early release still matters because it hands the compositor's frame pool its
// surface back a full copy sooner.
// Windows and macOS take a shared-texture handle; Linux takes dmabuf planes — hence the ctor/Execute split.
// targets: JS array of { width, height, format } (see osrcap::TargetSpec); absent/empty -> none
std::vector<osrcap::TargetSpec> ParseTargets(const Napi::CallbackInfo& info, size_t index) {
    std::vector<osrcap::TargetSpec> out;
    if (info.Length() <= index || !info[index].IsArray()) return out;
    Napi::Array arr = info[index].As<Napi::Array>();
    for (uint32_t i = 0; i < arr.Length(); ++i) {
        Napi::Value v = arr.Get(i);
        if (!v.IsObject()) continue;
        Napi::Object o = v.As<Napi::Object>();
        osrcap::TargetSpec t;
        t.w = o.Has("width") ? o.Get("width").As<Napi::Number>().Uint32Value() : 0;
        t.h = o.Has("height") ? o.Get("height").As<Napi::Number>().Uint32Value() : 0;
        t.format = o.Has("format") ? o.Get("format").As<Napi::Number>().Int32Value() : 0;
        if (t.w && t.h) out.push_back(t);
    }
    return out;
}

class ConsumeWorker : public Napi::AsyncWorker {
public:
#if defined(_WIN32) || defined(__APPLE__)
    ConsumeWorker(Napi::Env env, uintptr_t handle, uint32_t w, uint32_t h, int format, std::string key, uint32_t dstW, uint32_t dstH, std::vector<osrcap::TargetSpec> targets, Napi::Promise::Deferred deferred, const osrcap::VideoLayer& video = osrcap::VideoLayer(), Napi::Value videoData = Napi::Value())
        : Napi::AsyncWorker(env), handle_(handle), w_(w), h_(h), format_(format), key_(std::move(key)), dstW_(dstW), dstH_(dstH), targets_(std::move(targets)), deferred_(deferred), video_(video) {
        // the caller's video buffer is read on the worker thread: keep it alive for the call
        if (!videoData.IsEmpty() && videoData.IsBuffer()) videoRef_ = Napi::Persistent(videoData.As<Napi::Object>());
    }
#elif defined(__linux__)
    ConsumeWorker(Napi::Env env, std::vector<osrcap::DmabufPlane> planes, uint64_t modifier, uint32_t w, uint32_t h, int format, std::string key, uint32_t dstW, uint32_t dstH, std::vector<osrcap::TargetSpec> targets, Napi::Promise::Deferred deferred)
        : Napi::AsyncWorker(env), planes_(std::move(planes)), modifier_(modifier), w_(w), h_(h), format_(format), key_(std::move(key)), dstW_(dstW), dstH_(dstH), targets_(std::move(targets)), deferred_(deferred) {}
#endif
    void Execute() override {
        std::string err;
#if defined(_WIN32) || defined(__APPLE__)
        bool ok = video_.w ? osrcap::ReadbackConsume(handle_, w_, h_, format_, key_, dstW_, dstH_, targets_, video_, err)
                           : osrcap::ReadbackConsume(handle_, w_, h_, format_, key_, dstW_, dstH_, targets_, err);
#elif defined(__linux__)
        bool ok = targets_.empty() ? osrcap::ReadbackConsume(planes_, modifier_, w_, h_, format_, key_, dstW_, dstH_, err) : false;
        if (!targets_.empty()) err = "per-consumer targets are not supported by the Linux backend";
#endif
        if (!ok) SetError(err.empty() ? "consume failed" : err);
    }
    void OnOK() override {
        Napi::HandleScope s(Env());
        deferred_.Resolve(Env().Undefined());
    }
    void OnError(const Napi::Error& e) override { deferred_.Reject(e.Value()); }

private:
    osrcap::VideoLayer video_;
    Napi::ObjectReference videoRef_;
#if defined(_WIN32) || defined(__APPLE__)
    uintptr_t handle_;
#elif defined(__linux__)
    std::vector<osrcap::DmabufPlane> planes_;
    uint64_t modifier_ = 0;
#endif
    uint32_t w_, h_;
    int format_;
    std::string key_;
    uint32_t dstW_, dstH_;
    std::vector<osrcap::TargetSpec> targets_;
    Napi::Promise::Deferred deferred_;
};

class FinishWorker : public Napi::AsyncWorker {
public:
    // scaledDst/scaledSize + scaled buffer optional: when present, the promise resolves to { main, scaled }
    // (both reused pooled buffers); otherwise it resolves to just the main Buffer (backward-compatible).
    FinishWorker(Napi::Env env, std::string key, uint8_t* dst, size_t dstSize, Napi::Buffer<uint8_t> result, uint8_t* scaledDst, size_t scaledSize, bool hasScaled, Napi::Buffer<uint8_t> scaledResult, Napi::Promise::Deferred deferred)
        : Napi::AsyncWorker(env), key_(std::move(key)), dst_(dst), dstSize_(dstSize), resultRef_(Napi::Persistent(result)),
          scaledDst_(scaledDst), scaledSize_(scaledSize), hasScaled_(hasScaled), deferred_(deferred) {
        if (hasScaled) scaledRef_ = Napi::Persistent(scaledResult);
    }
    // per-consumer target buffers (one per TargetSpec given to consume); the promise then resolves an object
    // with a `targets` array even when there is no `scaled`
    void SetTargets(std::vector<Napi::Buffer<uint8_t>> bufs) {
        hasTargets_ = true;
        for (auto& b : bufs) {
            targetDsts_.push_back({ b.Data(), b.ByteLength() });
            targetRefs_.push_back(Napi::Persistent(b));
        }
    }
    void Execute() override {
        std::string err;
#if defined(__linux__)
        bool ok = targetDsts_.empty() ? osrcap::ReadbackFinish(key_, dst_, dstSize_, scaledDst_, scaledSize_, err) : false;
        if (!targetDsts_.empty()) err = "per-consumer targets are not supported by the Linux backend";
#else
        bool ok = osrcap::ReadbackFinish(key_, dst_, dstSize_, scaledDst_, scaledSize_, targetDsts_, err);
#endif
        if (!ok) SetError(err.empty() ? "finish failed" : err);
    }
    void OnOK() override {
        Napi::HandleScope s(Env());
        if (hasScaled_ || hasTargets_) {
            Napi::Object o = Napi::Object::New(Env());
            o.Set("main", resultRef_.Value());
            if (hasScaled_) o.Set("scaled", scaledRef_.Value());
            if (hasTargets_) {
                Napi::Array arr = Napi::Array::New(Env(), targetRefs_.size());
                for (size_t i = 0; i < targetRefs_.size(); ++i) arr.Set((uint32_t)i, targetRefs_[i].Value());
                o.Set("targets", arr);
            }
            deferred_.Resolve(o);
        } else {
            deferred_.Resolve(resultRef_.Value());
        }
    }
    void OnError(const Napi::Error& e) override { deferred_.Reject(e.Value()); }

private:
    std::string key_;
    uint8_t* dst_;
    size_t dstSize_;
    Napi::Reference<Napi::Buffer<uint8_t>> resultRef_;
    uint8_t* scaledDst_;
    size_t scaledSize_;
    bool hasScaled_;
    Napi::Reference<Napi::Buffer<uint8_t>> scaledRef_;
    bool hasTargets_ = false;
    std::vector<osrcap::TargetDst> targetDsts_;
    std::vector<Napi::Reference<Napi::Buffer<uint8_t>>> targetRefs_;
    Napi::Promise::Deferred deferred_;
};

// SINGLE-DISPATCH readback (plan §11 fix #1): one AsyncWorker runs the whole open+convert+WaitGpu+copy-out,
// and fires the JS `onRelease` callback (the caller's releaseTexture) from INSIDE Execute — via a
// ThreadSafeFunction — the instant the GPU is done with the shared texture, before the slow copy-out. This
// keeps the early-release (frame-pool-drain) benefit of the old two-phase Consume while collapsing the two
// N-API dispatches + the worker-JS-loop hop between them into ONE. Resolves to { main, scaled } when a GPU
// downscale was requested, else just the main Buffer (same shape as FinishWorker).
class ReadbackOnceWorker : public Napi::AsyncWorker {
public:
#if defined(_WIN32) || defined(__APPLE__)
    ReadbackOnceWorker(Napi::Env env, uintptr_t handle, uint32_t w, uint32_t h, int format, uint32_t dstW, uint32_t dstH,
                       uint8_t* dst, size_t dstSize, Napi::Buffer<uint8_t> result,
                       uint8_t* scaledDst, size_t scaledSize, bool hasScaled, Napi::Buffer<uint8_t> scaledResult,
                       Napi::ThreadSafeFunction release, Napi::Promise::Deferred deferred)
        : Napi::AsyncWorker(env), handle_(handle), w_(w), h_(h), format_(format), dstW_(dstW), dstH_(dstH),
          dst_(dst), dstSize_(dstSize), resultRef_(Napi::Persistent(result)),
          scaledDst_(scaledDst), scaledSize_(scaledSize), hasScaled_(hasScaled), release_(release), deferred_(deferred) {
        if (hasScaled) scaledRef_ = Napi::Persistent(scaledResult);
    }
#elif defined(__linux__)
    ReadbackOnceWorker(Napi::Env env, std::vector<osrcap::DmabufPlane> planes, uint64_t modifier, uint32_t w, uint32_t h, int format, std::string key, uint32_t dstW, uint32_t dstH,
                       uint8_t* dst, size_t dstSize, Napi::Buffer<uint8_t> result,
                       uint8_t* scaledDst, size_t scaledSize, bool hasScaled, Napi::Buffer<uint8_t> scaledResult,
                       Napi::ThreadSafeFunction release, Napi::Promise::Deferred deferred)
        : Napi::AsyncWorker(env), planes_(std::move(planes)), modifier_(modifier), key_(std::move(key)), w_(w), h_(h), format_(format), dstW_(dstW), dstH_(dstH),
          dst_(dst), dstSize_(dstSize), resultRef_(Napi::Persistent(result)),
          scaledDst_(scaledDst), scaledSize_(scaledSize), hasScaled_(hasScaled), release_(release), deferred_(deferred) {
        if (hasScaled) scaledRef_ = Napi::Persistent(scaledResult);
    }
#endif

    void Execute() override {
        std::string err;
        // onGpuDone: fire the JS onRelease callback on the worker's JS loop (non-blocking is fine — it just
        // posts releaseTexture to main). Guarded so it can only fire once even if the native layer were to
        // call it twice. The TSFN keeps one live ref; it is Released in OnOK/OnError regardless.
        bool released = false;
        auto onGpuDone = [&]() {
            if (released) return;
            released = true;
            release_.NonBlockingCall();
        };
#if defined(_WIN32) || defined(__APPLE__)
        bool ok = osrcap::ReadbackOnce(handle_, w_, h_, format_, dstW_, dstH_, dst_, dstSize_,
                                       scaledDst_, scaledSize_, onGpuDone, err);
#elif defined(__linux__)
        bool ok = osrcap::ReadbackOnce(planes_, modifier_, w_, h_, format_, key_, dstW_, dstH_, dst_, dstSize_,
                                       scaledDst_, scaledSize_, onGpuDone, err);
#endif
        // If the readback failed before the GPU wait, onGpuDone never fired; release the texture anyway so the
        // caller's frame pool can't leak the shared texture (the JS side also has a finally safety-release).
        if (!released) release_.NonBlockingCall();
        if (!ok) SetError(err.empty() ? "readback failed" : err);
    }

    void OnOK() override {
        Napi::HandleScope s(Env());
        release_.Release();  // drop the TSFN ref (its callback is a no-op after this)
        if (hasScaled_) {
            Napi::Object o = Napi::Object::New(Env());
            o.Set("main", resultRef_.Value());
            o.Set("scaled", scaledRef_.Value());
            deferred_.Resolve(o);
        } else {
            deferred_.Resolve(resultRef_.Value());
        }
    }
    void OnError(const Napi::Error& e) override {
        release_.Release();
        deferred_.Reject(e.Value());
    }

private:
#if defined(_WIN32) || defined(__APPLE__)
    uintptr_t handle_;
#elif defined(__linux__)
    std::vector<osrcap::DmabufPlane> planes_;
    uint64_t modifier_ = 0;
    std::string key_;
#endif
    uint32_t w_, h_;
    int format_;
    uint32_t dstW_, dstH_;
    uint8_t* dst_;
    size_t dstSize_;
    Napi::Reference<Napi::Buffer<uint8_t>> resultRef_;
    uint8_t* scaledDst_;
    size_t scaledSize_;
    bool hasScaled_;
    Napi::Reference<Napi::Buffer<uint8_t>> scaledRef_;
    Napi::ThreadSafeFunction release_;
    Napi::Promise::Deferred deferred_;
};

// readbackOnce(source, w, h, format, key, dstW, dstH, onRelease) -> Promise<main | { main, scaled }>. `key`
// selects this output's reused buffer pool (same as readbackFinish). `onRelease` is called (once) the moment
// the GPU has consumed the shared texture, before the slow copy-out — the caller releases the Electron texture.
Napi::Value ReadbackOnceJs(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    uint32_t w = info[1].As<Napi::Number>().Uint32Value();
    uint32_t h = info[2].As<Napi::Number>().Uint32Value();
    int format = (info.Length() > 3 && info[3].IsNumber()) ? info[3].As<Napi::Number>().Int32Value() : 0;
    std::string key = (info.Length() > 4 && info[4].IsString()) ? info[4].As<Napi::String>().Utf8Value() : std::string();
    uint32_t dstW = (info.Length() > 5 && info[5].IsNumber()) ? info[5].As<Napi::Number>().Uint32Value() : 0;
    uint32_t dstH = (info.Length() > 6 && info[6].IsNumber()) ? info[6].As<Napi::Number>().Uint32Value() : 0;
    Napi::Function onRelease = info[7].As<Napi::Function>();

    auto deferred = Napi::Promise::Deferred::New(env);
    size_t outSize = OutputSize(w, h, format);
    Napi::Buffer<uint8_t> result = AcquireOutputBuffer(env, key, outSize);
    bool hasScaled = dstW > 0 && dstH > 0;
    size_t scaledSize = hasScaled ? (size_t)dstW * dstH * 4 : 0;
    Napi::Buffer<uint8_t> scaledResult = hasScaled ? AcquireOutputBuffer(env, key + "#scaled", scaledSize) : Napi::Buffer<uint8_t>::New(env, 0);

    // one-shot TSFN (initialThreadCount 1) so the worker thread can invoke onRelease on the JS loop mid-Execute.
    Napi::ThreadSafeFunction release = Napi::ThreadSafeFunction::New(env, onRelease, "osrReadbackRelease", 0, 1);
#if defined(_WIN32) || defined(__APPLE__)
    uintptr_t handle = 0;
    Napi::Buffer<uint8_t> buf = info[0].As<Napi::Buffer<uint8_t>>();
    if (buf.Length() >= sizeof(uintptr_t)) std::memcpy(&handle, buf.Data(), sizeof(uintptr_t));
    (new ReadbackOnceWorker(env, handle, w, h, format, dstW, dstH, result.Data(), outSize, result,
                            hasScaled ? scaledResult.Data() : nullptr, scaledSize, hasScaled, scaledResult, release, deferred))
        ->Queue();
#elif defined(__linux__)
    std::vector<osrcap::DmabufPlane> planes;
    uint64_t modifier = 0;
    ParseLinuxSource(info[0], planes, modifier);
    (new ReadbackOnceWorker(env, std::move(planes), modifier, w, h, format, key, dstW, dstH, result.Data(), outSize, result,
                            hasScaled ? scaledResult.Data() : nullptr, scaledSize, hasScaled, scaledResult, release, deferred))
        ->Queue();
#endif
    return deferred.Promise();
}

Napi::Value ReadbackConsumeJs(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    uint32_t w = info[1].As<Napi::Number>().Uint32Value();
    uint32_t h = info[2].As<Napi::Number>().Uint32Value();
    int format = (info.Length() > 3 && info[3].IsNumber()) ? info[3].As<Napi::Number>().Int32Value() : 0;
    std::string key = (info.Length() > 4 && info[4].IsString()) ? info[4].As<Napi::String>().Utf8Value() : std::string();
    // optional GPU downscale target (server/stage): consume(source, w, h, format, key, dstW, dstH)
    uint32_t dstW = (info.Length() > 5 && info[5].IsNumber()) ? info[5].As<Napi::Number>().Uint32Value() : 0;
    uint32_t dstH = (info.Length() > 6 && info[6].IsNumber()) ? info[6].As<Napi::Number>().Uint32Value() : 0;
    auto deferred = Napi::Promise::Deferred::New(env);
#if defined(_WIN32) || defined(__APPLE__)
    uintptr_t handle = 0;
    Napi::Buffer<uint8_t> buf = info[0].As<Napi::Buffer<uint8_t>>();
    if (buf.Length() >= sizeof(uintptr_t)) std::memcpy(&handle, buf.Data(), sizeof(uintptr_t));
    // consume(..., targets, video): `video` = { width, height, format, data } composited under the page
    osrcap::VideoLayer video;
    Napi::Value videoData;
    if (info.Length() > 8 && info[8].IsObject()) {
        Napi::Object v = info[8].As<Napi::Object>();
        if (v.Get("data").IsBuffer()) {
            videoData = v.Get("data");
            Napi::Buffer<uint8_t> vb = videoData.As<Napi::Buffer<uint8_t>>();
            video.data = vb.Data();
            video.bytes = vb.ByteLength();
            video.w = v.Get("width").As<Napi::Number>().Uint32Value();
            video.h = v.Get("height").As<Napi::Number>().Uint32Value();
            video.format = v.Has("format") ? v.Get("format").As<Napi::Number>().Int32Value() : 0;
        }
    }
    (new ConsumeWorker(env, handle, w, h, format, std::move(key), dstW, dstH, ParseTargets(info, 7), deferred, video, videoData))->Queue();
#elif defined(__linux__)
    std::vector<osrcap::DmabufPlane> planes;
    uint64_t modifier = 0;
    ParseLinuxSource(info[0], planes, modifier);
    (new ConsumeWorker(env, std::move(planes), modifier, w, h, format, std::move(key), dstW, dstH, ParseTargets(info, 7), deferred))->Queue();
#endif
    return deferred.Promise();
}

Napi::Value ReadbackFinishJs(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    std::string key = info[0].As<Napi::String>().Utf8Value();
    uint32_t w = info[1].As<Napi::Number>().Uint32Value();
    uint32_t h = info[2].As<Napi::Number>().Uint32Value();
    int format = (info.Length() > 3 && info[3].IsNumber()) ? info[3].As<Napi::Number>().Int32Value() : 0;
    // optional GPU-downscaled BGRA (server/stage): finish(key, w, h, format, dstW, dstH) -> { main, scaled }
    uint32_t dstW = (info.Length() > 4 && info[4].IsNumber()) ? info[4].As<Napi::Number>().Uint32Value() : 0;
    uint32_t dstH = (info.Length() > 5 && info[5].IsNumber()) ? info[5].As<Napi::Number>().Uint32Value() : 0;
    auto deferred = Napi::Promise::Deferred::New(env);
    size_t outSize = OutputSize(w, h, format);
    // finish(key, w, h, format, dstW, dstH, dst?): an optional caller-owned Buffer receives the frame
    // directly (the caller then hands that same buffer to its consumers without another copy); it must be
    // at least outSize bytes and untouched until the promise settles. Otherwise a pooled buffer is used.
    Napi::Buffer<uint8_t> result;
    if (info.Length() > 6 && info[6].IsBuffer()) {
        result = info[6].As<Napi::Buffer<uint8_t>>();
        if (result.ByteLength() < outSize) {
            deferred.Reject(Napi::Error::New(env, "readbackFinish: destination buffer is smaller than the frame").Value());
            return deferred.Promise();
        }
    } else {
        result = AcquireOutputBuffer(env, key, outSize);
    }

    bool hasScaled = dstW > 0 && dstH > 0;
    size_t scaledSize = hasScaled ? (size_t)dstW * dstH * 4 : 0;
    // separate reused pool for the scaled buffer so it never aliases the main (UYVY) pool for this output
    Napi::Buffer<uint8_t> scaledResult = hasScaled ? AcquireOutputBuffer(env, key + "#scaled", scaledSize) : Napi::Buffer<uint8_t>::New(env, 0);
    // finish(..., dst?, targets?, targetDsts?): `targets` repeats the specs given to consume (so each output's
    // byte size is known); `targetDsts` optionally supplies caller-owned buffers, else pooled ones are used
    std::vector<osrcap::TargetSpec> targets = ParseTargets(info, 7);
    std::vector<Napi::Buffer<uint8_t>> targetBufs;
    for (size_t i = 0; i < targets.size(); ++i) {
        size_t bytes = osrcap::TargetBytes(targets[i]);
        Napi::Buffer<uint8_t> tb;
        if (info.Length() > 8 && info[8].IsArray() && info[8].As<Napi::Array>().Get((uint32_t)i).IsBuffer()) {
            tb = info[8].As<Napi::Array>().Get((uint32_t)i).As<Napi::Buffer<uint8_t>>();
            if (tb.ByteLength() < bytes) {
                deferred.Reject(Napi::Error::New(env, "readbackFinish: a target destination buffer is smaller than its frame").Value());
                return deferred.Promise();
            }
        } else {
            tb = AcquireOutputBuffer(env, key + "#t" + std::to_string(i), bytes);
        }
        targetBufs.push_back(tb);
    }
    auto* worker = new FinishWorker(env, std::move(key), result.Data(), outSize, result, hasScaled ? scaledResult.Data() : nullptr, scaledSize, hasScaled, scaledResult, deferred);
    if (!targets.empty()) worker->SetTargets(std::move(targetBufs));
    worker->Queue();
    return deferred.Promise();
}

#if defined(_WIN32)
// Stage-0 harness hook (Windows): _copyPoolSelfTest(bytes?, concurrency?, iterations?) -> { ok, mismatches,
// workers, ms }. Runs synchronously on the caller (blocks); hangs only if the pool deadlocks. Dev-only.
Napi::Value CopyPoolSelfTestJs(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    uint32_t bytes = (info.Length() > 0 && info[0].IsNumber()) ? info[0].As<Napi::Number>().Uint32Value() : (8u << 20);
    uint32_t concurrency = (info.Length() > 1 && info[1].IsNumber()) ? info[1].As<Napi::Number>().Uint32Value() : 6;
    uint32_t iterations = (info.Length() > 2 && info[2].IsNumber()) ? info[2].As<Napi::Number>().Uint32Value() : 200;
    uint32_t workers = 0, mismatches = 0;
    double ms = 0;
    bool ok = osrcap::CopyPoolSelfTest(bytes, concurrency, iterations, workers, mismatches, ms);
    Napi::Object o = Napi::Object::New(env);
    o.Set("ok", Napi::Boolean::New(env, ok));
    o.Set("mismatches", Napi::Number::New(env, mismatches));
    o.Set("workers", Napi::Number::New(env, workers));
    o.Set("ms", Napi::Number::New(env, ms));
    return o;
}

// Copy-pool telemetry: the self-calibrated active worker count (knee-seeker base, plan §11 CV#1). Read-only —
// there is deliberately NO setter and no env override; the pool sizes itself from measured throughput.
Napi::Value CopyPoolActiveWorkersJs(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), osrcap::CopyPoolActiveWorkers());
}
#endif  // _WIN32 (copy-pool harness)

// Backend telemetry. Windows (Stage-2 sub-step B): the copy-out backend in use ("d3d11on12" | "d3d11" |
// "none") — the §5 ladder picks it; FS_READBACK=d3d11 is a binary diagnostic selector, not a tunable.
// macOS: "metal" (IOSurface wrapped zero-copy as an MTLTexture) | "iosurface-cpu" | "none".
// Linux: "egl-gles3" (GPU readback active) | "cpu" (unavailable/forced/demoted) | "none". Read-only.
Napi::Value ReadbackBackendJs(const Napi::CallbackInfo& info) {
    return Napi::String::New(info.Env(), osrcap::ReadbackBackend());
}
#endif  // _WIN32 || __APPLE__ || __linux__

// Free a pool's reused buffers when its output is removed (else the per-output buffers leak for the process
// lifetime). No-op if the key is unknown.
Napi::Value ReleasePool(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() > 0 && info[0].IsString()) {
        std::string key = info[0].As<Napi::String>().Utf8Value();
        // Only ever touch THIS env's pools (see AddonData) so we never destroy another thread/env's references.
        if (AddonData* data = env.GetInstanceData<AddonData>()) {
            data->pools.erase(key);
            data->pools.erase(key + "#scaled");
        }
#if defined(_WIN32) || defined(__APPLE__) || defined(__linux__)
        osrcap::ReadbackReleaseKey(key);
#endif
    }
    return env.Undefined();
}

}  // namespace

namespace osrcap {
void RegisterConvert(Napi::Env env, Napi::Object exports);
}
void InitShm(Napi::Env env, Napi::Object exports);

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    // Per-env pool storage (see AddonData). N-API runs Init once per env (main thread + each worker), and
    // finalizes this instance data on that env's teardown, on the env's own thread — the correct place to
    // destroy its Napi::References.
    env.SetInstanceData(new AddonData());
    exports.Set("readback", Napi::Function::New(env, Readback));
    exports.Set("releasePool", Napi::Function::New(env, ReleasePool));
    // The two-phase/once exports are installed ONLY when this platform's GPU path is actually available —
    // FreeShow keys off `typeof readbackConsume === "function"` (ndiWorker two-phase preference,
    // OutputLifecycle hasGpuDownscale) and must fall back to single-phase `readback` when only a CPU path
    // exists. Windows always has D3D11 here (the §5 ladder degrades internally); macOS needs a Metal device;
    // Linux needs the EGL/GLES3 import to have initialized.
#if defined(_WIN32)
    exports.Set("readbackOnce", Napi::Function::New(env, ReadbackOnceJs));
    exports.Set("readbackConsume", Napi::Function::New(env, ReadbackConsumeJs));
    exports.Set("readbackFinish", Napi::Function::New(env, ReadbackFinishJs));
#elif defined(__APPLE__)
    if (osrcap::MacGpuReadbackInit()) {
        exports.Set("readbackOnce", Napi::Function::New(env, ReadbackOnceJs));
        exports.Set("readbackConsume", Napi::Function::New(env, ReadbackConsumeJs));
        exports.Set("readbackFinish", Napi::Function::New(env, ReadbackFinishJs));
    }
#elif defined(__linux__)
    if (osrcap::LinuxGpuReadbackInit()) {
        exports.Set("readbackOnce", Napi::Function::New(env, ReadbackOnceJs));
        exports.Set("readbackConsume", Napi::Function::New(env, ReadbackConsumeJs));
        exports.Set("readbackFinish", Napi::Function::New(env, ReadbackFinishJs));
    }
#endif
#if defined(_WIN32) || defined(__APPLE__) || defined(__linux__)
    exports.Set("_readbackBackend", Napi::Function::New(env, ReadbackBackendJs));
    exports.Set("targetsSupported", Napi::Boolean::New(env, osrcap::TargetsSupported()));
    exports.Set("videoLayerSupported", Napi::Boolean::New(env, osrcap::VideoLayerSupported()));
#endif
#if defined(_WIN32)
    exports.Set("_copyPoolSelfTest", Napi::Function::New(env, CopyPoolSelfTestJs));
    exports.Set("_copyPoolActiveWorkers", Napi::Function::New(env, CopyPoolActiveWorkersJs));
#endif
    osrcap::RegisterConvert(env, exports);
    InitShm(env, exports);
    return exports;
}

NODE_API_MODULE(osr_readback, Init)
