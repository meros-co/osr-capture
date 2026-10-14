// Windows regression test: use the shipping helper directly, as on12_harness does.
// Build: npx node-gyp rebuild
// Run: node test/copy-alignment.js
// CPU memory only; no GPU or Electron is needed to run this executable.
#include "../src/readback_win.cc"

int main(int argc, char** argv) {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    if (argc == 3 && std::strcmp(argv[1], "--guard") == 0) {
        const size_t bytes = strtoull(argv[2], nullptr, 10);
        if (bytes > 65) return 2;
        SYSTEM_INFO info;
        GetSystemInfo(&info);
        const size_t page = info.dwPageSize;
        auto* memory = static_cast<unsigned char*>(VirtualAlloc(nullptr, page * 2, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
        if (!memory) return 3;
        DWORD oldProtection;
        if (!VirtualProtect(memory + page, page, PAGE_NOACCESS, &oldProtection)) return 3;
        unsigned char destination[256];
        std::memset(destination, 0xa5, sizeof(destination));
        auto* source = memory + page - bytes;
        for (size_t i = 0; i < bytes; ++i) source[i] = static_cast<unsigned char>(i);
        // The source ends exactly at an inaccessible page, detecting overreads.
        StreamCopyFromWC(destination + 32, source, bytes);
        for (size_t i = 0; i < bytes; ++i) {
            if (destination[32 + i] != static_cast<unsigned char>(i)) return 4;
        }
        for (size_t i = 0; i < 32; ++i) {
            if (destination[i] != 0xa5 || destination[32 + bytes + i] != 0xa5) return 5;
        }
        VirtualFree(memory, 0, MEM_RELEASE);
        return 0;
    }
    if (argc != 5) return 2;
    const size_t offset = strtoull(argv[1], nullptr, 10);
    const size_t bytes = strtoull(argv[2], nullptr, 10);
    const size_t stride = strtoull(argv[3], nullptr, 10);
    const size_t rows = strtoull(argv[4], nullptr, 10);
    if (offset > 15 || bytes > 15360 || stride < bytes || stride > 15360 || rows == 0 || rows > 30) return 2;
    const size_t capacity = offset + stride * (rows - 1) + bytes + 64;
    auto* source = static_cast<unsigned char*>(VirtualAlloc(nullptr, capacity, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    auto* destination = static_cast<unsigned char*>(VirtualAlloc(nullptr, bytes * rows + 64, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if (!source || !destination) return 3;
    for (size_t i = 0; i < capacity; ++i) source[i] = static_cast<unsigned char>((i * 37 + 11) % 251);
    std::memset(destination, 0xa5, bytes * rows + 64);
    for (size_t row = 0; row < rows; ++row) {
        StreamCopyFromWC(destination + 32 + row * bytes, source + offset + row * stride, bytes);
        if (std::memcmp(destination + 32 + row * bytes, source + offset + row * stride, bytes)) return 4;
    }
    for (size_t i = 0; i < 32; ++i) {
        if (destination[i] != 0xa5 || destination[32 + bytes * rows + i] != 0xa5) return 5;
    }
    VirtualFree(source, 0, MEM_RELEASE);
    VirtualFree(destination, 0, MEM_RELEASE);
    return 0;
}
