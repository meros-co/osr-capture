// Named shared memory for frames between FreeShow processes: the stream receive process writes a
// decoded frame into a ring slot, and the drawing window's preload maps the same region and reads the
// slot out. Cross-platform: a named file mapping on Windows, shm_open on POSIX. Electron's V8 forbids
// external ArrayBuffers, so the mapping is never exposed directly: shmWrite/shmRead copy between it and
// JS-owned buffers (one memcpy each way, the only copies a frame makes between the two processes).
#include <napi.h>
#include <string>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <memory>
#include <cstring>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

struct Mapping {
    void* ptr = nullptr;
    size_t size = 0;
    bool owner = false;
    // copies reading or writing ptr, from any thread; a copy holds a reference for its whole duration,
    // so the region cannot be released underneath it and no lock is held while it runs
    std::atomic<int> inflight{0};
    std::atomic<bool> unmapRequested{false};  // shmUnmap during copies: release when the last one lands
#if defined(_WIN32)
    HANDLE handle = nullptr;
#else
    int fd = -1;
#endif
};

// The mutex guards the TABLE (find/insert/erase), never a copy: holding it across a memcpy would make
// every ring wait behind whichever one is copying. Mappings are held by pointer so an entry keeps its
// address when the table rehashes.
std::unordered_map<std::string, std::unique_ptr<Mapping>> g_maps;
std::mutex g_mapsMutex;

#if defined(_WIN32)
std::wstring Widen(const std::string& s) {
    if (s.empty()) return std::wstring();
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring w(n, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &w[0], n);
    return w;
}
#endif

bool MapRegion(const std::string& name, size_t size, bool create, Mapping& out, std::string& err) {
#if defined(_WIN32)
    std::wstring wname = Widen("Local\\" + name);
    HANDLE h = nullptr;
    if (create) {
        h = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, (DWORD)((unsigned long long)size >> 32), (DWORD)(size & 0xffffffffu), wname.c_str());
    } else {
        h = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, wname.c_str());
    }
    if (!h) { err = create ? "CreateFileMapping failed" : "OpenFileMapping failed"; return false; }
    void* p = MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (!p) { CloseHandle(h); err = "MapViewOfFile failed"; return false; }
    out.handle = h;
#else
    std::string sname = "/" + name;
    int fd = create ? shm_open(sname.c_str(), O_CREAT | O_RDWR, 0600) : shm_open(sname.c_str(), O_RDWR, 0600);
    if (fd < 0) { err = "shm_open failed"; return false; }
    if (create && ftruncate(fd, (off_t)size) != 0) { close(fd); shm_unlink(sname.c_str()); err = "ftruncate failed"; return false; }
    void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { close(fd); if (create) shm_unlink(sname.c_str()); err = "mmap failed"; return false; }
    out.fd = fd;
#endif
    out.ptr = p;
    out.size = size;
    out.owner = create;
    return true;
}

void UnmapRegion(const std::string& name, Mapping& m) {
#if defined(_WIN32)
    if (m.ptr) UnmapViewOfFile(m.ptr);
    if (m.handle) CloseHandle(m.handle);
#else
    if (m.ptr) munmap(m.ptr, m.size);
    if (m.fd >= 0) close(m.fd);
    if (m.owner) shm_unlink(("/" + name).c_str());
#endif
    m.ptr = nullptr;
    m.size = 0;
    m.owner = false;
#if defined(_WIN32)
    m.handle = nullptr;
#else
    m.fd = -1;
#endif
}

// shmMap(name, bytes, create) -> true. The same name maps the same memory in every process; a name
// already mapped in this process is reused.
Napi::Value ShmMap(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 3 || !info[0].IsString() || !info[1].IsNumber() || !info[2].IsBoolean()) {
        Napi::TypeError::New(env, "shmMap(name, bytes, create)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    std::string name = info[0].As<Napi::String>().Utf8Value();
    size_t size = (size_t)info[1].As<Napi::Number>().Int64Value();
    bool create = info[2].As<Napi::Boolean>().Value();
    if (!size) {
        Napi::RangeError::New(env, "shmMap: bytes must be > 0").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    std::lock_guard<std::mutex> lock(g_mapsMutex);
    auto it = g_maps.find(name);
    if (it == g_maps.end()) {
        Mapping m;
        std::string err;
        if (!MapRegion(name, size, create, m, err)) {
            Napi::Error::New(env, "shmMap: " + err).ThrowAsJavaScriptException();
            return env.Undefined();
        }
        auto entry = std::make_unique<Mapping>();
        entry->ptr = m.ptr;
        entry->size = m.size;
        entry->owner = m.owner;
#if defined(_WIN32)
        entry->handle = m.handle;
#else
        entry->fd = m.fd;
#endif
        it = g_maps.emplace(name, std::move(entry)).first;
    }
    Mapping& m = *it->second;
    if (m.unmapRequested) {
        Napi::Error::New(env, "shmMap: mapping is being released").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    if (m.size < size) {
        Napi::RangeError::New(env, "shmMap: existing mapping is smaller than requested").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    return Napi::Boolean::New(env, true);
}

// Resolves (name, offset, view) to the mapped byte range and takes a reference on the mapping, so the
// copy that follows runs with no lock held. Throws on a bad range, taking no reference.
static bool AcquireRange(const Napi::CallbackInfo& info, const char* fn, Mapping** mapping, uint8_t** base, uint8_t** js, size_t* len) {
    Napi::Env env = info.Env();
    if (info.Length() < 3 || !info[0].IsString() || !info[1].IsNumber() || !info[2].IsTypedArray()) {
        Napi::TypeError::New(env, std::string(fn) + "(name, offset, bytes: Uint8Array)").ThrowAsJavaScriptException();
        return false;
    }
    std::string name = info[0].As<Napi::String>().Utf8Value();
    int64_t offset = info[1].As<Napi::Number>().Int64Value();
    Napi::TypedArray ta = info[2].As<Napi::TypedArray>();
    size_t bytes = ta.ByteLength();

    std::lock_guard<std::mutex> lock(g_mapsMutex);
    auto it = g_maps.find(name);
    if (it == g_maps.end()) {
        Napi::Error::New(env, std::string(fn) + ": not mapped").ThrowAsJavaScriptException();
        return false;
    }
    Mapping* m = it->second.get();
    if (m->unmapRequested.load()) {
        Napi::Error::New(env, std::string(fn) + ": mapping is being released").ThrowAsJavaScriptException();
        return false;
    }
    if (offset < 0 || (uint64_t)offset + bytes > m->size) {
        Napi::RangeError::New(env, std::string(fn) + ": range outside the mapping").ThrowAsJavaScriptException();
        return false;
    }
    m->inflight.fetch_add(1);
    *mapping = m;
    *base = (uint8_t*)m->ptr + offset;
    *js = (uint8_t*)ta.ArrayBuffer().Data() + ta.ByteOffset();
    *len = bytes;
    return true;
}

// Drop a reference taken by AcquireRange, releasing the region if an unmap was waiting on it.
static void ReleaseRange(const std::string& name, Mapping* m) {
    if (m->inflight.fetch_sub(1) != 1) return;
    if (!m->unmapRequested.load()) return;
    std::lock_guard<std::mutex> lock(g_mapsMutex);
    auto it = g_maps.find(name);
    if (it == g_maps.end() || it->second.get() != m) return;
    if (m->inflight.load() != 0) return;  // another copy started while we were taking the lock
    UnmapRegion(name, *m);
    g_maps.erase(it);
}

// shmWrite(name, offset, bytes): copy a JS buffer into the mapping at offset
Napi::Value ShmWrite(const Napi::CallbackInfo& info) {
    Mapping* m; uint8_t* base; uint8_t* js; size_t len;
    if (!AcquireRange(info, "shmWrite", &m, &base, &js, &len)) return info.Env().Undefined();
    memcpy(base, js, len);
    ReleaseRange(info[0].As<Napi::String>().Utf8Value(), m);
    return Napi::Number::New(info.Env(), (double)len);
}

// shmRead(name, offset, bytes): fill a JS buffer from the mapping at offset
Napi::Value ShmRead(const Napi::CallbackInfo& info) {
    Mapping* m; uint8_t* base; uint8_t* js; size_t len;
    if (!AcquireRange(info, "shmRead", &m, &base, &js, &len)) return info.Env().Undefined();
    memcpy(js, base, len);
    ReleaseRange(info[0].As<Napi::String>().Utf8Value(), m);
    return Napi::Number::New(info.Env(), (double)len);
}

// shmWriteAsync / shmReadAsync: the same copies on the libuv thread pool, resolving to the byte count.
// The JS buffer is referenced for the copy's lifetime and the mapping cannot go away underneath it
// (shmUnmap defers until in-flight copies land).
class ShmCopyWorker : public Napi::AsyncWorker {
public:
    ShmCopyWorker(Napi::Env env, const std::string& name, Mapping* mapping, uint8_t* base, Napi::Value js, uint8_t* jsPtr, size_t len, bool write)
        : Napi::AsyncWorker(env), deferred_(Napi::Promise::Deferred::New(env)), name_(name), mapping_(mapping), base_(base), jsPtr_(jsPtr), len_(len), write_(write) {
        ref_ = Napi::Persistent(js.As<Napi::Object>());
    }
    Napi::Promise GetPromise() { return deferred_.Promise(); }
    void Execute() override {
        if (write_) memcpy(base_, jsPtr_, len_);
        else memcpy(jsPtr_, base_, len_);
    }
    void OnOK() override {
        ReleaseRange(name_, mapping_);
        deferred_.Resolve(Napi::Number::New(Env(), (double)len_));
    }
    void OnError(const Napi::Error& e) override {
        ReleaseRange(name_, mapping_);
        deferred_.Reject(e.Value());
    }

private:
    Napi::Promise::Deferred deferred_;
    std::string name_;
    Mapping* mapping_;
    uint8_t* base_;
    uint8_t* jsPtr_;
    size_t len_;
    bool write_;
    Napi::ObjectReference ref_;
};

static Napi::Value ShmCopyAsync(const Napi::CallbackInfo& info, bool write) {
    Napi::Env env = info.Env();
    Mapping* m; uint8_t* base; uint8_t* js; size_t len;
    if (!AcquireRange(info, write ? "shmWriteAsync" : "shmReadAsync", &m, &base, &js, &len)) return env.Undefined();
    std::string name = info[0].As<Napi::String>().Utf8Value();
    auto* worker = new ShmCopyWorker(env, name, m, base, info[2], js, len, write);
    worker->Queue();
    return worker->GetPromise();
}
Napi::Value ShmWriteAsync(const Napi::CallbackInfo& info) { return ShmCopyAsync(info, true); }
Napi::Value ShmReadAsync(const Napi::CallbackInfo& info) { return ShmCopyAsync(info, false); }

// shmUnmap(name): release this process's mapping (the creator also removes the name on POSIX). With
// async copies in flight the release happens when the last of them lands.
Napi::Value ShmUnmap(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "shmUnmap(name)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    std::string name = info[0].As<Napi::String>().Utf8Value();
    std::lock_guard<std::mutex> lock(g_mapsMutex);
    auto it = g_maps.find(name);
    if (it == g_maps.end()) return Napi::Boolean::New(env, false);
    Mapping* m = it->second.get();
    // a copy holds a reference for its whole duration: mark the region and let the last one release it
    m->unmapRequested.store(true);
    if (m->inflight.load() > 0) return Napi::Boolean::New(env, true);
    UnmapRegion(name, *m);
    g_maps.erase(it);
    return Napi::Boolean::New(env, true);
}

}  // namespace

void InitShm(Napi::Env env, Napi::Object exports) {
    exports.Set("shmMap", Napi::Function::New(env, ShmMap));
    exports.Set("shmWrite", Napi::Function::New(env, ShmWrite));
    exports.Set("shmRead", Napi::Function::New(env, ShmRead));
    exports.Set("shmWriteAsync", Napi::Function::New(env, ShmWriteAsync));
    exports.Set("shmReadAsync", Napi::Function::New(env, ShmReadAsync));
    exports.Set("shmUnmap", Napi::Function::New(env, ShmUnmap));
}
