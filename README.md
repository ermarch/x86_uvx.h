# x86_uv.h

**Unified Vector intrinsics — A Compile-time Vector Length Agnostic (VLA) SIMD abstraction layer for x86.**

## Overview

This single-header library implements a high-performance, unified subset of data-parallel instructions inspired by the design philosophy of ARM SVE2 and RISC-V Vector (RVV). 

Instead of hardcoding register widths, algorithms use abstract vector types and scale automatically via the compile-time constant `UV_LANES`. The implementation resolves completely at compile time via macro inlining with zero runtime abstraction overhead.

## Supported Target Levels

The library automatically detects the available SIMD extension and configures `UV_LANES` accordingly:

| Level | Architecture | Register Width | UV_LANES (f32/i32) |
|:-----:|:------------:|:--------------:|:------------------:|
|   3   | AVX-512      | 512 bits       |           16       |
|   2   | AVX2         | 256 bits       |            8       |
|   1   | SSE2         | 128 bits       |            4       |
|   0   | Scalar Fallback| N/A           |            1       |

## Requirements

- **Minimum compiler flags**: `-msse2`
- **Target upscaling** is controlled completely by your compiler architecture flags (e.g., `-msse2`, `-mavx2`, `-mavx512f`)

## Example Usage

```c
void buffer_multiply(float *dest, const float *src, size_t size)
{
    size_t i = 0;

    #define UNROLL_FACTOR (uv_registers() >= 16 ? (uv_registers() / 4) : 2)
    const size_t block_step = UNROLL_FACTOR * uv_lanes();
    for (; i + block_step <= size; i += block_step) {
        for (size_t u = 0; u < UNROLL_FACTOR; ++u) {
            size_t offset = i + (u * uv_lanes());

            v_f32 s_vec = uv_loadu_f32(s + offset);
            v_f32 d_vec = uv_loadu_f32(d + offset);
            uv_storeu_f32(d + offset, uv_mul_f32(d_vec, s_vec));
        }
    }
    for (; i < size; ++i) {
        d[i] *= s[i];
    }
    #undef UNROLL_FACTOR

    uv_clear_lanes();
}
```

## Vector Types

The library defines unified vector types that scale with the target level:

| Type       | Description                    |
|:-----------|:-------------------------------|
| `v_f32`    | Single-precision floating point |
| `v_f64`    | Double-precision floating point |
| `v_i32`    | Signed 32-bit integers         |
| `v_i8`     | Signed 8-bit integers          |
| `uv_mask`  | Bitmask for element selection  |

## Key Functions

All functions are overloaded per data type and automatically scale to the available SIMD width.

### Arithmetic
- `uv_add_f32(a, b)`, `uv_sub_f32(a, b)`, `uv_mul_f32(a, b)`, `uv_div_f32(a, b)`
- `uv_max_f32(a, b)`, `uv_min_f32(a, b)`
- `uv_fma_f32(a, b, c)` — Fused multiply-add

### Comparison
- `uv_cmpgt_f32(a, b)`, `uv_cmplt_f32(a, b)`
- `uv_cmpeq_i8(a, b)`, `uv_cmplt_i8(a, b)`

### Selection
- `uv_select_f32(mask, true_vec, false_vec)`

### Bitwise
- `uv_and_f32(a, b)`, `uv_or_f32(a, b)`, `uv_xor_f32(a, b)`

### Memory Operations
- `uv_load_f32(ptr)`, `uv_loadu_f32(ptr)`
- `uv_store_f32(ptr, v)`, `uv_storeu_f32(ptr, v)`

### Conversions
- `uv_cvt_f32_i32(v)`, `uv_cvt_i32_f32(v)`

### Utility
- `uv_lanes()`, `uv_lanes64()`, `uv_registers()`
- `uv_clear_lanes()`

For a complete reference, see the header file.

## Author and Date

**Author**: Erik Hofman <erik@ehofman.com>  
**Date**: May 2026

## License

MIT License
