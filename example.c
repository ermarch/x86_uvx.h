
#include "x86_uvx.h"

/*
  * Example use:
  * - UNROLL_FACTOR establishes unroll depth dynamically from physical hardware
  *   register limits.
  * - block_step as the total jump block size is perfectly uniform and fixed at
  *   compile time.
  * - Outer Loop: Steps across the array chunks
  * - Inner Loop: Operates across the parallel-register registers.
  *   Because UNROLL_FACTOR is a constant, the compiler unrolls this loop 100%
  *   of the time.
  */
void buffer_multiply(float *dest, const float *src, size_t size)
{
    size_t i = 0;

    #define UNROLL_FACTOR (uv_registers() >= 16 ? (uv_registers() / 4) : 2)
    const size_t block_step = UNROLL_FACTOR * uv_lanes(32);
    for (; i + block_step <= size; i += block_step) {
        for (size_t u = 0; u < UNROLL_FACTOR; ++u) {
            size_t offset = i + (u * uv_lanes(32));

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

