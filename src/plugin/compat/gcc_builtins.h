#pragma once
#ifdef _MSC_VER
#include <intrin.h>
inline unsigned long __builtin_ctzll(unsigned long long value) {
    unsigned long index;
    _BitScanForward64(&index, value);
    return index;
}
#endif