# x86_uvx.h

**Unified Vector intrinsics — A Compile-time Vector Length Agnostic (VLA) SIMD abstraction layer for x86.**

## Overview

This single-header library implements a high-performance, unified subset of data-parallel instructions inspired by the design philosophy of ARM SVE2 and RISC-V Vector (RVV).

Instead of hardcoding register widths, algorithms use abstract vector types and scale automatically via the compile-time constant `UV_LANES`. The implementation resolves completely at compile time via macro inlining with zero runtime abstraction overhead.

## Supported Target Levels

The library automatically detects the available SIMD extension and configures `UV_LANES` accordingly. Supported architectures include AVX10 (future extension), AVX-512, AVX2, SSE2, and scalar fallback.

| Level | Architecture | Register Width | UV_LANES (f32/i32) |
|:-----:|:------------:|:--------------:|:------------------:|
|   3   | AVX-512      | 512 bits       |           16       |
|   2   | AVX2         | 256 bits       |            8       |
|   1   | SSE2         | 128 bits       |            4       |
|   0   | Scalar Fallback| N/A           |            1       |

## Requirements

- **Minimum compiler flags**: `-msse2`
- **Target upscaling** is controlled completely by your compiler architecture flags (e.g., `-msse2`, `-mavx2`, `-mavx512f`)
- For AVX10 support, additional compiler flags such as `-mavx10` or appropriate EVEX flags may be required (depending on compiler and target platform)

## Vector Types

The library defines unified vector types that scale with the target level:

| Type       | Description                    |
|:-----------|:-------------------------------|
| `v_f32`    | Single-precision floating point |
| `v_f64`    | Double-precision floating point |
| `v_i32`    | Signed 32-bit integers         |
| `v_i16`    | Signed 16-bit integers         |
| `v_i8`     | Signed 8-bit integers          |
| `uv_mask`  | Bitmask for element selection  |
| `uv_mask8` | Extended bitmask for AVX10     |

## Key Functions

All functions are overloaded per data type and automatically scale to the available SIMD width. Operations include:

### Memory Operations
- `uv_load_f32(ptr)`, `uv_loadu_f32(ptr)`
- `uv_store_f32(ptr, v)`, `uv_storeu_f32(ptr, v)`
- Similar operations for `f64`, `i32`, `i16`, `i8`
- `uv_setzero_*()`, `uv_dup_*()`

### Arithmetic
- `uv_add_*()`, `uv_sub_*()`, `uv_mul_*()`, `uv_div_*()`
- `uv_max_*()`, `uv_min_*()`, `uv_abs_*()`, `uv_neg_*()`
- `uv_fmadd_*()` — Fused multiply-add (with fallback for older CPU features)
- `uv_fmsub_*()` — Fused multiply-sub (with fallback for older CPU features)

### Comparison
- `uv_cmpgt_*()`, `uv_cmplt_*()`, `uv_cmpeq_i8()`

### Selection
- `uv_select_*()` — masked selection

### Bitwise
- `uv_and_*()`, `uv_andnot_*()`, `uv_or_*()`, `uv_xor_*()`

### Conversions
- `uv_cvt_*()` — between integer and floating point types
- `uv_cvt_i16_i32()`, `uv_cvt_i32_i16()`, `uv_cvt_i8_i32()`, `uv_cvt_i32_i8()`

### Reduction
- `uv_reduce_add_f32(v)`, `uv_reduce_min_f32(v)`, `uv_reduce_max_f32(v)`

### 8-bit Integer Operations
- `uv_load_i8()`, `uv_store_i8()`
- `uv_cmpeq_i8()`, `uv_cmplt_i8()`
- `uv_shuffle_i8()`, `uv_permutexvar_i8()`
- `uv_select_i8()`, `uv_compress_i8()`

### Scalar Bit Manipulation
- `uv_popcnt_u64()`, `uv_tzcnt_u64()`, `uv_lzcnt_u64()`

### Utility
- `uv_lanes()`, `uv_registers()`
- `uv_clear_lanes()`
- `uv_alignment()`, `uv_alignment_mask()`
- `uv_prefetch(ptr, hint)`

> **Note:** Some operations have fallback implementations for older CPU features (e.g., `uv_max_i32()` without SSSE3, `uv_abs_i32()` without SSSE3). The library includes the `<immintrin.h>` header for SIMD intrinsics.

## Example Usage

```c
void buffer_multiply(float *dest, const float *src, size_t size)
{
    size_t i = 0;

    #define UNROLL_FACTOR (uv_registers() >= 16 ? (uv_registers() / 4) : 2)
    const size_t block_step = UNROLL_FACTOR * uv_lanes(32);
    for (; i + block_step <= size; i += block_step) {
        for (size_t u = 0; u < UNROLL_FACTOR; ++u) {
            size_t offset = i + (u * uv_lanes(32));

            v_f32 s_vec = uv_loadu_f32(src + offset);
            v_f32 d_vec = uv_loadu_f32(dest + offset);
            uv_storeu_f32(dest + offset, uv_mul_f32(d_vec, s_vec));
        }
    }
    for (; i < size; ++i) {
        dest[i] *= src[i];
    }
    #undef UNROLL_FACTOR
}
```

## Design Philosophy

The library is designed to provide a unified, vector-length agnostic SIMD interface that:

- Resolves completely at compile time with zero runtime abstraction overhead
- Scales automatically with available hardware
- Provides fallback implementations for older CPU features
- Encourages loop unrolling through compile-time constants
- Minimizes cache pollution via prefetch hints (`uv_prefetch()`)

## Author and Date

**Author**: Erik Hofman <erik@ehofman.com>  
**Date**: May 2026

## License

MIT License
