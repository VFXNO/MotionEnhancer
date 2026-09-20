#pragma once

#include <windows.h>
#include <cstdint>

// QueryPerformanceCounter in 100 ns units (the WGC timestamp domain).
inline int64_t wallClock100ns() {
    static const int64_t frequency = [] {
        LARGE_INTEGER f = {};
        QueryPerformanceFrequency(&f);
        return f.QuadPart;
    }();
    LARGE_INTEGER now = {};
    QueryPerformanceCounter(&now);
    // Split to avoid overflowing int64 when the counter is large.
    const int64_t seconds = now.QuadPart / frequency;
    const int64_t remainder = now.QuadPart % frequency;
    return seconds * 10000000LL + remainder * 10000000LL / frequency;
}
