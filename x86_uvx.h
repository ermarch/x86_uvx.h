#ifndef UVX_SIMD_H
#define UVX_SIMD_H

/**
 * @file      x86_uv.h
 * @brief     Unified Vector intrinsics.
 *            A Compile-time Vector Length Agnostic (VLA) SIMD abstraction layer
 *            for x86.
 *
 * @details   This single-header library implements a high-performance, unified
 *            subset of data-parallel instructions inspired by the design
 *            philosophy of ARM SVE2 and RISC-V Vector (RVV).
 *
 *            Instead of hardcoding register widths, algorithms use abstract
 *            vector types and scale automatically via the compile-time constant
 *            `UV_LANES`. The implementation resolves completely at compile time
 *            via macro inlining with zero runtime abstraction overhead.
 *
 * @note      Supported Target Levels:
 *            - LEVEL 3: AVX-512 (512-bit registers, UV_LANES = 16 for f32/i32)
 *            - LEVEL 2: AVX2    (256-bit registers, UV_LANES = 8  for f32/i32)
 *            - LEVEL 1: SSE2    (128-bit registers, UV_LANES = 4  for f32/i32)
 *            - LEVEL 0:         (Scalar Fallback,   UV_LANES = 1  for f32/i32)
 *
 * @attention Requirements:
 *            Must be compiled with minimum `-msse2` flag. Target upscaling is
 *            controlled completely by your compiler architecture flags
 *            (e.g., `-msse2`, `-mavx2`, `-mavx512f`).
 *
 * @author    Erik Hofman <erik@ehofman.com>
 * @date      May 2026
 * @license   MIT License
 */

#define __UVX_0__

 /***
  * Example use:
  * - UNROLL_FACTOR establishes unroll depth dynamically from physical hardware
  *   register limits.
  * - block_step as the total jump block size is perfectly uniform and fixed at
  *   compile time.
  * - Outer Loop: Steps across the array chunks
  * - Inner Loop: Operates across the parallel-register registers.
  *   Because UNROLL_FACTOR is a constant, the compiler unrolls this loop 100%
  *   of the time.
  *
  * void buffer_multiply(float *dest, const float *src, size_t size)
  * {
  *     size_t i = 0;
  *
  *     #define UNROLL_FACTOR (uv_registers() >= 16 ? (uv_registers() / 4) : 2)
  *     const size_t block_step = UNROLL_FACTOR * uv_lanes();
  *     for (; i + block_step <= size; i += block_step) {
  *         for (size_t u = 0; u < UNROLL_FACTOR; ++u) {
  *             size_t offset = i + (u * uv_lanes());
  *
  *             v_f32 s_vec = uv_loadu_f32(s + offset);
  *             v_f32 d_vec = uv_loadu_f32(d + offset);
  *             uv_storeu_f32(d + offset, uv_mul_f32(d_vec, s_vec));
  *         }
  *     }
  *     for (; i < size; ++i) {
  *         d[i] *= s[i];
  *     }
  *     #undef UNROLL_FACTOR
  *
  *     uv_clear_lanes();
  * }
  */

#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>

/*
 * Globals
 */
#define UV_HINT_T0               _MM_HINT_T0   // Fetch into L1 Cache
#define UV_HINT_T1               _MM_HINT_T1   // Fetch into L2 Cache
#define UV_HINT_T2               _MM_HINT_T2   // Fetch into L3 Cache
#define UV_HINT_NTA              _MM_HINT_NTA  // Non-Temporal (Minimize Cache Pollution)
#define uv_prefetch(ptr, hint)   _mm_prefetch((const char*)(ptr), (hint))

#define uv_lanes()               UV_LANES
#define uv_lanes64()             UV_LANES_F64
#define uv_registers()           UV_REGISTERS

#define UV_MAX_REGISTERS         32
#define UV_MAX_LANES             16

#define UV_ALIGNMENT             UV_LANES_I8
#define UV_MEMMASK               (UV_ALIGNMENT-1)

#if defined(__AVX10__)
  #if defined(__EVEX512__) || (defined(__AVX10_MAX_VEC_LEN__) && __AVX10_MAX_VEC_LEN__ >= 512)
    #define UV_REGISTERS 32

    typedef __mmask16 uv_mask;
    typedef __mmask32 uv_mask8;
    typedef __m512  v_f32;
    typedef __m512i v_i32;
    typedef __m512d v_f64;
    typedef __m512i v_i8;

    #define UV_LANES     16
    #define UV_LANES_F64  8
    #define UV_LANES_I8  64

    #define uv_clear_lanes() _mm256_zeroupper()
  #else
    #define UV_REGISTERS 16

    typedef __mmask8  uv_mask;
     typedef __m256i uv_mask_i8;
    typedef __m256  v_f32;
    typedef __m256i v_i32;
    typedef __m256d v_f64;
    typedef __m256i v_i8;

    #define UV_LANES      8
    #define UV_LANES_F64  4
    #define UV_LANES_I8  32

    #define uv_clear_lanes() _mm256_zeroupper()
  #endif

#elif defined(__AVX512F__)
  #define UV_REGISTERS 32

  typedef __mmask16 uv_mask;
  typedef __mmask64 uv_mask8;
  typedef __m512  v_f32;
  typedef __m512i v_i32;
  typedef __m512d v_f64;
  typedef __m512i v_i8;

  #define UV_LANES     16
  #define UV_LANES_F64  8
  #define UV_LANES_I8  64

  #define uv_clear_lanes() _mm256_zeroupper()

#elif defined(__FMA__) || defined(__AVX2__)
  #define UV_REGISTERS 16

  typedef __m256  uv_mask;
  typedef __m256i uv_mask_i8;
  typedef __m256  v_f32;
  typedef __m256i v_i32;
  typedef __m256d v_f64;
  typedef __m256i v_i8;

  #define UV_LANES        8
  #define UV_LANES_F64    4
  #define UV_LANES_I8    32

  #define uv_clear_lanes() _mm256_zeroupper()

#elif defined(__SSE2__)
  #if defined(__i386__) || defined(_M_IX86) || defined(__SSE__)
    #define UV_REGISTERS  8
  #else
    #define UV_REGISTERS 16
  #endif

  typedef __m128  uv_mask;
  typedef __m128i uv_mask_i8;
  typedef __m128  v_f32;
  typedef __m128i v_i32;
  typedef __m128d v_f64;
  typedef __m128i v_i8;

  #define UV_LANES        4
  #define UV_LANES_F64    2
  #define UV_LANES_I8    16

  #define uv_clear_lanes() ((void)0)

#else
  #define UV_REGISTERS    1

  typedef int     uv_mask;
  typedef int     uv_mask_i8;
  typedef float   v_f32;
  typedef int32_t v_i32;
  typedef double  v_f64;
  typedef int8_t  v_i8;

  #define UV_LANES        1
  #define UV_LANES_F64    1
  #define UV_LANES_I8     1

  #define uv_clear_lanes() ((void)0)
#endif


/*
 * Single-Precision floating point support
 */
#if defined(__AVX512F__)
  // AVX-512 hardware ignores standard alignment requirements for baseline
  // performance; maps uniformly
  #define uv_load_f32(ptr)       _mm512_loadu_ps(ptr)
  #define uv_loadu_f32(ptr)      _mm512_loadu_ps(ptr)
  #define uv_store_f32(ptr, v)   _mm512_storeu_ps(ptr, v)
  #define uv_storeu_f32(ptr, v)  _mm512_storeu_ps(ptr, v)
  #define uv_setzero_f32()       _mm512_setzero_ps()
  #define uv_dup_f32(val)        _mm512_set1_ps(val)

  #define uv_add_f32(a, b)       _mm512_add_ps(a, b)
  #define uv_sub_f32(a, b)       _mm512_sub_ps(a, b)
  #define uv_mul_f32(a, b)       _mm512_mul_ps(a, b)
  #define uv_div_f32(a, b)       _mm512_div_ps(a, b)
  #define uv_max_f32(a, b)       _mm512_max_ps(a, b)
  #define uv_min_f32(a, b)       _mm512_min_ps(a, b)

  #define uv_cmpgt_f32(a, b)     _mm512_cmp_ps_mask(a, b, 3)
  #define uv_cmplt_f32(a, b)     _mm512_cmp_ps_mask(a, b, 1)
  #define uv_select_f32(m, t, f) _mm512_mask_blend_ps(m, f, t)

  #define uv_and_f32(a, b)       _mm512_and_ps(a, b)
  #define uv_andnot_f32(a, b)    _mm512_andnot_ps(a, b)
  #define uv_or_f32(a, b)        _mm512_or_ps(a, b)
  #define uv_xor_f32(a, b)       _mm512_xor_ps(a, b)

  #define uv_cvt_f32_i32(v)      _mm512_cvtps_epi32(v)
  #define uv_cvt_i32_f32(v)      _mm512_cvtepi32_ps(v)

#elif defined (__FMA__) || defined(__AVX2__)
  #define uv_load_f32(ptr)       _mm256_load_ps(ptr)
  #define uv_loadu_f32(ptr)      _mm256_loadu_ps(ptr)
  #define uv_store_f32(ptr, v)   _mm256_store_ps(ptr, v)
  #define uv_storeu_f32(ptr, v)  _mm256_storeu_ps(ptr, v)
  #define uv_setzero_f32()       _mm256_setzero_ps()
  #define uv_dup_f32(val)        _mm256_set1_ps(val)

  #define uv_add_f32(a, b)       _mm256_add_ps(a, b)
  #define uv_sub_f32(a, b)       _mm256_sub_ps(a, b)
  #define uv_mul_f32(a, b)       _mm256_mul_ps(a, b)
  #define uv_div_f32(a, b)       _mm256_div_ps(a, b)
  #define uv_max_f32(a, b)       _mm256_max_ps(a, b)
  #define uv_min_f32(a, b)       _mm256_min_ps(a, b)

  #define uv_cmpgt_f32(a, b)     _mm256_cmp_ps(a, b, _CMP_GT_OS)
  #define uv_cmplt_f32(a, b)     _mm256_cmp_ps(a, b, _CMP_LT_OS)
  #if defined(__AVX10__)
    #define uv_select_f32(m, t, f) _mm256_mask_blend_ps(m, f, t)
  #else
    #define uv_select_f32(m, t, f) _mm256_blendv_ps(f, t, (__m256)(m))
  #endif

  #define uv_and_f32(a, b)       _mm256_and_ps(a, b)
  #define uv_andnot_f32(a, b)    _mm256_andnot_ps(a, b)
  #define uv_or_f32(a, b)        _mm256_or_ps(a, b)
  #define uv_xor_f32(a, b)       _mm256_xor_ps(a, b)

  #define uv_cvt_f32_i32(v)      _mm256_cvtps_epi32(v)
  #define uv_cvt_i32_f32(v)      _mm256_cvtepi32_ps(v)

#elif defined(__SSE2__)

  #define uv_load_f32(ptr)       _mm_load_ps(ptr)
  #define uv_loadu_f32(ptr)      _mm_loadu_ps(ptr)
  #define uv_store_f32(ptr, v)   _mm_store_ps(ptr, v)
  #define uv_storeu_f32(ptr, v)  _mm_storeu_ps(ptr, v)
  #define uv_setzero_f32()       _mm_setzero_ps()
  #define uv_dup_f32(val)        _mm_set1_ps(val)

  #define uv_add_f32(a, b)       _mm_add_ps(a, b)
  #define uv_sub_f32(a, b)       _mm_sub_ps(a, b)
  #define uv_mul_f32(a, b)       _mm_mul_ps(a, b)
  #define uv_div_f32(a, b)       _mm_div_ps(a, b)
  #define uv_max_f32(a, b)       _mm_max_ps(a, b)
  #define uv_min_f32(a, b)       _mm_min_ps(a, b)

  #define uv_cmpgt_f32(a, b)     _mm_cmpgt_ps(a, b)
  #define uv_cmplt_f32(a, b)     _mm_cmpgt_ps(b, a)
  #if defined(__SSE4_1__)
    #define uv_select_f32(m, t, f) _mm_blendv_ps(f, t, m)
  #else
    #define uv_select_f32(m, t, f) _mm_or_ps(_mm_andnot_ps(m, f), _mm_and_ps(m, t))
  #endif

  #define uv_and_f32(a, b)       _mm_and_ps(a, b)
  #define uv_andnot_f32(a, b)    _mm_andnot_ps(a, b)
  #define uv_or_f32(a, b)        _mm_or_ps(a, b)
  #define uv_xor_f32(a, b)       _mm_xor_ps(a, b)

  #define uv_cvt_f32_i32(v)      _mm_cvtps_epi32(v)
  #define uv_cvt_i32_f32(v)      _mm_cvtepi32_ps(v)

#else
  #define uv_load_f32(ptr)       (*(const float*)(ptr))
  #define uv_loadu_f32(ptr)      (*(const float*)(ptr))
  #define uv_store_f32(ptr, v)   (*(float*)(ptr) = (v))
  #define uv_storeu_f32(ptr, v)  (*(float*)(ptr) = (v))
  #define uv_setzero_f32()       (0.0f)
  #define uv_dup_f32(val)        (val)

  #define uv_add_f32(a, b)       ((a) + (b))
  #define uv_sub_f32(a, b)       ((a) - (b))
  #define uv_mul_f32(a, b)       ((a) * (b))
  #define uv_div_f32(a, b)       ((a) / (b))

  #define uv_max_f32(a, b)       ((a) > (b) ? (a) : (b))
  #define uv_min_f32(a, b)       ((a) < (b) ? (a) : (b))

  #define uv_cmpgt_f32(a, b)     ((a) > (b))
  #define uv_cmplt_f32(a, b)     ((a) < (b))
  #define uv_select_f32(m, t, f) ((m) ? (t) : (f))

  static inline v_f32 uv_and_f32(v_f32 a, v_f32 b) {
    union { float f; uint32_t i; } va, vb, vr;
    va.f = a; vb.f = b; vr.i = va.i & vb.i;
    return vr.f;
  }

  static inline v_f32 uv_andnot_f32(v_f32 a, v_f32 b) {
    union { float f; uint32_t i; } va, vb, vr;
    va.f = a; vb.f = b; vr.i = ~va.i & vb.i;
    return vr.f;
  }

  static inline v_f32 uv_or_f32(v_f32 a, v_f32 b) {
    union { float f; uint32_t i; } va, vb, vr;
    va.f = a; vb.f = b; vr.i = va.i | vb.i;
    return vr.f;
  }

  static inline v_f32 uv_xor_f32(v_f32 a, v_f32 b) {
    union { float f; uint32_t i; } va, vb, vr;
    va.f = a; vb.f = b; vr.i = va.i ^ vb.i;
    return vr.f;
  }

  #define uv_cvt_f32_i32(v)      ((int32_t)(v))
  #define uv_cvt_i32_f32(v)      ((float)(v))

#endif


/*
 * Single-Precision and Double-Precision Float-Multiply-Add support
 */
#if defined(__AVX512F__)
  #define uv_fma_f32(a, b, c)    _mm512_fmadd_ps(a, b, c)
  #define uv_fma_f64(a, b, c)    _mm512_fmadd_pd(a, b, c)

#elif defined(__FMA__)
  #define uv_fma_f32(a, b, c)    _mm256_fmadd_ps(a, b, c)
  #define uv_fma_f64(a, b, c)    _mm256_fmadd_pd(a, b, c)

#elif defined(__AVX2__)
  static inline __m256 uv_fma_f32(__m256 a, __m256 b, __m256 c) {
      __m256 intermediate_mul = _mm256_mul_ps(a, b);
      return _mm256_add_ps(intermediate_mul, c);
  }
  static inline __m256d uv_fma_f64(__m256d a, __m256d b, __m256d c) {
      return _mm256_add_pd(_mm256_mul_pd(a, b), c);
  }

#elif defined(__SSE2__)
  static inline __m128 uv_fma_f32(__m128 a, __m128 b, __m128 c) {
      return _mm_add_ps(_mm_mul_ps(a, b), c);
  }
  static inline __m128d uv_fma_f64(__m128d a, __m128d b, __m128d c) {
      return _mm_add_pd(_mm_mul_pd(a, b), c);
  }

#else
  #define uv_fma_f32(a, b, c)    (((a) * (b)) + (c))
  #define uv_fma_f64(a, b, c)    (((a) * (b)) + (c))

#endif


/*
 * Double-Precision floating point support
 */
#if defined(__AVX512F__)
  #define uv_load_f64(ptr)       _mm512_loadu_pd(ptr)
  #define uv_loadu_f64(ptr)      _mm512_loadu_pd(ptr)
  #define uv_store_f64(ptr, v)   _mm512_storeu_pd(ptr, v)
  #define uv_storeu_f64(ptr, v)  _mm512_storeu_pd(ptr, v)
  #define uv_setzero_f64()       _mm512_setzero_pd()
  #define uv_dup_f64(val)        _mm512_set1_pd(val)

  #define uv_add_f64(a, b)       _mm512_add_pd(a, b)
  #define uv_sub_f64(a, b)       _mm512_sub_pd(a, b)
  #define uv_mul_f64(a, b)       _mm512_mul_pd(a, b)
  #define uv_div_f64(a, b)       _mm512_div_pd(a, b)
  #define uv_max_f64(a, b)       _mm512_max_pd(a, b)
  #define uv_min_f64(a, b)       _mm512_min_pd(a, b)

  #define uv_and_f64(a, b)       _mm512_and_pd(a, b)
  #define uv_andnot_f64(a, b)    _mm512_andnot_pd(a, b)
  #define uv_or_f64(a, b)        _mm512_or_pd(a, b)
  #define uv_xor_f64(a, b)       _mm512_xor_pd(a, b)

#elif defined(__FMA__) || defined(__AVX2__)
  #define uv_load_f64(ptr)       _mm256_load_pd(ptr)
  #define uv_loadu_f64(ptr)      _mm256_loadu_pd(ptr)
  #define uv_store_f64(ptr, v)   _mm256_store_pd(ptr, v)
  #define uv_storeu_f64(ptr, v)  _mm256_storeu_pd(ptr, v)
  #define uv_setzero_f64()       _mm256_setzero_pd()
  #define uv_dup_f64(val)        _mm256_set1_pd(val)

  #define uv_add_f64(a, b)       _mm256_add_pd(a, b)
  #define uv_sub_f64(a, b)       _mm256_sub_pd(a, b)
  #define uv_mul_f64(a, b)       _mm256_mul_pd(a, b)
  #define uv_div_f64(a, b)       _mm256_div_pd(a, b)
  #define uv_max_f64(a, b)       _mm256_max_pd(a, b)
  #define uv_min_f64(a, b)       _mm256_min_pd(a, b)

  #define uv_and_f64(a, b)       _mm256_and_pd(a, b)
  #define uv_andnot_f64(a, b)    _mm256_andnot_pd(a, b)
  #define uv_or_f64(a, b)        _mm256_or_pd(a, b)
  #define uv_xor_f64(a, b)       _mm256_xor_pd(a, b)

#elif defined(__SSE2__)
  #define uv_load_f64(ptr)       _mm_load_pd(ptr)
  #define uv_loadu_f64(ptr)      _mm_loadu_pd(ptr)
  #define uv_store_f64(ptr, v)   _mm_store_pd(ptr, v)
  #define uv_storeu_f64(ptr, v)  _mm_storeu_pd(ptr, v)
  #define uv_setzero_f64()       _mm_setzero_pd()
  #define uv_dup_f64(val)        _mm_set1_pd(val)

  #define uv_add_f64(a, b)       _mm_add_pd(a, b)
  #define uv_sub_f64(a, b)       _mm_sub_pd(a, b)
  #define uv_mul_f64(a, b)       _mm_mul_pd(a, b)
  #define uv_div_f64(a, b)       _mm_div_pd(a, b)
  #define uv_max_f64(a, b)       _mm_max_pd(a, b)
  #define uv_min_f64(a, b)       _mm_min_pd(a, b)

  #define uv_and_f64(a, b)       _mm_and_pd(a, b)
  #define uv_andnot_f64(a, b)    _mm_andnot_pd(a, b)
  #define uv_or_f64(a, b)        _mm_or_pd(a, b)
  #define uv_xor_f64(a, b)       _mm_xor_pd(a, b)

#else
  #define uv_load_f64(ptr)       (*(const double*)(ptr))
  #define uv_loadu_f64(ptr)      (*(const double*)(ptr))
  #define uv_store_f64(ptr, v)   (*(double*)(ptr) = (v))
  #define uv_storeu_f64(ptr, v)  (*(double*)(ptr) = (v))
  #define uv_setzero_f64()       (0.0)
  #define uv_dup_f64(val)        (val)

  #define uv_add_f64(a, b)       ((a) + (b))
  #define uv_sub_f64(a, b)       ((a) - (b))
  #define uv_mul_f64(a, b)       ((a) * (b))
  #define uv_div_f64(a, b)       ((a) / (b))
  #define uv_max_f64(a, b)       ((a) > (b) ? (a) : (b))
  #define uv_min_f64(a, b)       ((a) < (b) ? (a) : (b))

  static inline v_f64 uv_and_f64(v_f64 a, v_f64 b) {
    union { double f; uint64_t i; } va, vb, vr;
    va.f = a; vb.f = b; vr.i = va.i & vb.i;
    return vr.f;
  }

  static inline v_f64 uv_andnot_f64(v_f64 a, v_f64 b) {
    union { double f; uint64_t i; } va, vb, vr;
    va.f = a; vb.f = b; vr.i = ~va.i & vb.i;
    return vr.f;
  }

  static inline v_f64 uv_or_f64(v_f64 a, v_f64 b) {
    union { double f; uint64_t i; } va, vb, vr;
    va.f = a; vb.f = b; vr.i = va.i | vb.i;
    return vr.f;
  }

  static inline v_f64 uv_xor_f64(v_f64 a, v_f64 b) {
    union { double f; uint64_t i; } va, vb, vr;
    va.f = a; vb.f = b; vr.i = va.i ^ vb.i;
    return vr.f;
  }
#endif


/*
 * 32-bit Integer support
 */
#if defined(__AVX512F__)
  #define uv_load_i32(ptr)       _mm512_loadu_si512((__m512i const*)(ptr))
  #define uv_loadu_i32(ptr)      _mm512_loadu_si512((__m512i const*)(ptr))
  #define uv_store_i32(ptr, v)   _mm512_storeu_si512((__m512i*)(ptr), (v))
  #define uv_storeu_i32(ptr, v)  _mm512_storeu_si512((__m512i*)(ptr), (v))
  #define uv_setzero_i32()       _mm512_setzero_si512()
  #define uv_dup_i32(val)        _mm512_set1_epi32(val)

  #define uv_add_i32(a, b)       _mm512_add_epi32(a, b)
  #define uv_sub_i32(a, b)       _mm512_sub_epi32(a, b)
  #define uv_mul_i32(a, b)       _mm512_mullo_epi32(a, b)
  #define uv_max_i32(a, b)       _mm512_max_epi32(a, b)
  #define uv_min_i32(a, b)       _mm512_min_epi32(a, b)
  #define uv_abs_i32(v)          _mm512_abs_epi32(v)

  #define uv_and_i32(a, b)       _mm512_and_si512(a, b)
  #define uv_or_i32(a, b)        _mm512_or_si512(a, b)
  #define uv_xor_i32(a, b)       _mm512_xor_si512(a, b)

  #define uv_shl_i32(v, imm)     _mm512_slli_epi32(v, imm)
  #define uv_shr_i32(v, imm)     _mm512_srai_epi32(v, imm)

#elif defined(__FMA__) || defined(__AVX2__)
  #define uv_load_i32(ptr)       _mm256_load_si256((__m256i const*)(ptr))
  #define uv_loadu_i32(ptr)      _mm256_loadu_si256((__m256i const*)(ptr))
  #define uv_store_i32(ptr, v)   _mm256_store_si256((__m256i*)(ptr), (v))
  #define uv_storeu_i32(ptr, v)  _mm256_storeu_si256((__m256i*)(ptr), (v))
  #define uv_setzero_i32()       _mm256_setzero_si256()
  #define uv_dup_i32(val)        _mm256_set1_epi32(val)

  #define uv_add_i32(a, b)       _mm256_add_epi32(a, b)
  #define uv_sub_i32(a, b)       _mm256_sub_epi32(a, b)
  #define uv_mul_i32(a, b)       _mm256_mullo_epi32(a, b)
  #define uv_max_i32(a, b)       _mm256_max_epi32(a, b)
  #define uv_min_i32(a, b)       _mm256_min_epi32(a, b)
  #define uv_abs_i32(v)          _mm256_abs_epi32(v)

  #define uv_and_i32(a, b)       _mm256_and_si256(a, b)
  #define uv_or_i32(a, b)        _mm256_or_si256(a, b)
  #define uv_xor_i32(a, b)       _mm256_xor_si256(a, b)

  #define uv_shl_i32(v, imm)     _mm256_slli_epi32(v, imm)
  #define uv_shr_i32(v, imm)     _mm256_srai_epi32(v, imm)

#elif defined(__SSE2__)
  #define uv_load_i32(ptr)       _mm_load_si128((__m128i const*)(ptr))
  #define uv_loadu_i32(ptr)      _mm_loadu_si128((__m128i const*)(ptr))
  #define uv_store_i32(ptr, v)   _mm_store_si128((__m128i*)(ptr), (v))
  #define uv_storeu_i32(ptr, v)  _mm_storeu_si128((__m128i*)(ptr), (v))
  #define uv_setzero_i32()       _mm_setzero_si128()
  #define uv_dup_i32(val)        _mm_set1_epi32(val)

  #define uv_add_i32(a, b)       _mm_add_epi32(a, b)
  #define uv_sub_i32(a, b)       _mm_sub_epi32(a, b)
  #define uv_mul_i32(a, b)       _mm_mullo_epi32(a, b)
  #define uv_max_i32(a, b)       _mm_max_epi32(a, b)
  #define uv_min_i32(a, b)       _mm_min_epi32(a, b)
  #if defined(__SSSE3__)
    #define uv_abs_i32(v)        _mm_abs_epi32(v)
  #else
    static inline __m128i uv_abs_i32(__m128i v) {
      __m128i sign = _mm_srai_epi32(v, 31);
      return _mm_sub_epi32(_mm_xor_si128(v, sign), sign);
    }
  #endif

  #define uv_and_i32(a, b)       _mm_and_si128(a, b)
  #define uv_or_i32(a, b)        _mm_or_si128(a, b)
  #define uv_xor_i32(a, b)       _mm_xor_si128(a, b)

  #define uv_shl_i32(v, imm)     _mm_slli_epi32(v, imm)
  #define uv_shr_i32(v, imm)     _mm_srai_epi32(v, imm)

#else
  #define uv_load_i32(ptr)       (*(const int32_t*)(ptr))
  #define uv_loadu_i32(ptr)      (*(const int32_t*)(ptr))
  #define uv_store_i32(ptr, v)   (*(int32_t*)(ptr) = (v))
  #define uv_storeu_i32(ptr, v)  (*(int32_t*)(ptr) = (v))
  #define uv_setzero_i32()       (0)
  #define uv_dup_i32(val)        (val)

  #define uv_add_i32(a, b)       ((a) + (b))
  #define uv_sub_i32(a, b)       ((a) - (b))
  #define uv_mul_i32(a, b)       ((a) * (b))
  #define uv_max_i32(a, b)       ((a) > (b) ? (a) : (b))
  #define uv_min_i32(a, b)       ((a) < (b) ? (a) : (b))
  #define uv_abs_i32(v)          ((v) < 0 ? -(v) : (v))

  #define uv_and_i32(a, b)       ((a) & (b))
  #define uv_or_i32(a, b)        ((a) | (b))
  #define uv_xor_i32(a, b)       ((a) ^ (b))

  #define uv_shl_i32(v, imm)     ((v) << (imm))
  #define uv_shr_i32(v, imm)     ((v) >> (imm))

#endif


/*
 * 8-bit Integer support
 */
#if defined(__AVX10__)
  #if defined(__EVEX512__) || (defined(__AVX10_MAX_VEC_LEN__) && __AVX10_MAX_VEC_LEN__ >= 512)
    #define uv_load_i8(ptr)      _mm512_load_si512((const void*)(ptr))
    #define uv_loadu_i8(ptr)     _mm512_loadu_si512((const void*)(ptr))
    #define uv_store_i8(ptr, v)  _mm512_store_si512((void*)(ptr), (v))
    #define uv_storeu_i8(ptr, v) _mm512_storeu_si512((void*)(ptr), (v))

    #define uv_cmpeq_i8(a, b)    _mm512_cmpeq_epi8_mask(a, b)
    #define uv_cmplt_i8(a, b)    _mm512_cmplt_epi8_mask(a, b)

    #define uv_shuffle_i8(a, b)  _mm512_shuffle_epi8(a, b)
    #define uv_permutexvar_i8(idx, a)  _mm512_permutexvar_epi8(idx, a)

    #define uv_select_i8(mask, t, f)   _mm512_mask_blend_epi8(mask, f, t)
    #define uv_compress_i8(mask, a)    _mm512_maskz_compress_epi8(mask, a)

  #else
    #define uv_load_i8(ptr)      _mm256_load_si256((const __m256i*)(ptr))
    #define uv_loadu_i8(ptr)     _mm256_loadu_si256((const __m256i*)(ptr))
    #define uv_store_i8(ptr, v)  _mm256_store_si256((__m256i*)(ptr), (v))
    #define uv_storeu_i8(ptr, v) _mm256_storeu_si256((__m256i*)(ptr), (v))

    #define uv_cmpeq_i8(a, b)    _mm256_cmpeq_epi8_mask(a, b)
    #define uv_cmplt_i8(a, b)    _mm256_cmplt_epi8_mask(a, b)

    #define uv_shuffle_i8(a, b)  _mm256_shuffle_epi8(a, b)
    #define uv_permutexvar_i8(idx, a)  _mm256_permutexvar_epi8(idx, a)

    #define uv_select_i8(mask, t, f)   _mm256_mask_blend_epi8(mask, f, t)
    #define uv_compress_i8(mask, a)    _mm256_maskz_compress_epi8(mask, a)
  #endif

#elif defined(__AVX512F__)
  #define uv_load_i8(ptr)        _mm512_load_si512((__m512i const*)(ptr))
  #define uv_loadu_i8(ptr)       _mm512_loadu_si512((__m512i const*)(ptr))
  #define uv_store_i8(ptr, v)    _mm512_store_si512((__m512i*)(ptr), (v))
  #define uv_storeu_i8(ptr, v)   _mm512_storeu_si512((__m512i*)(ptr), (v))

  #if defined(__AVX512BW__)
    #define uv_cmpeq_i8(a, b)    _mm512_cmpeq_epi8_mask(a, b)
    #define uv_cmplt_i8(a, b)    _mm512_cmplt_epi8_mask(a, b)

    #define uv_shuffle_i8(a, b)        _mm512_shuffle_epi8(a, b)
    #define uv_permutexvar_i8(idx, a)  _mm512_permutexvar_epi8(idx, a)

    #define uv_select_i8(mask, t, f)   _mm512_mask_blend_epi8(mask, f, t)
    #define uv_compress_i8(mask, a)    _mm512_maskz_compress_epi8(mask, a)
   #else
    static inline __m512i uv_shuffle_i8(__m512i a, __m512i b) {
     __m256i a_low  = _mm512_castsi512_si256(a);
      __m256i a_high = _mm512_extracti64x4_epi64(a, 1);
      __m256i b_low  = _mm512_castsi512_si256(b);
      __m256i b_high = _mm512_extracti64x4_epi64(b, 1);
      
      __m256i r_low  = _mm256_shuffle_epi8(a_low, b_low);
      __m256i r_high = _mm256_shuffle_epi8(a_high, b_high);
      
      return _mm512_inserti64x4(_mm512_castsi256_si512(r_low), r_high, 1);
    }

    #define uv_permutexvar_i8(idx, a) uv_shuffle_i8(a, idx)

    static inline __mmask64 uv_cmpeq_i8(__m512i a, __m512i b) {
      __m256i a_low  = _mm512_castsi512_si256(a);
      __m256i a_high = _mm512_extracti64x4_epi64(a, 1);
      __m256i b_low  = _mm512_castsi512_si256(b);
      __m256i b_high = _mm512_extracti64x4_epi64(b, 1);

      __m256i cmp_l = _mm256_cmpeq_epi8(a_low, b_low);
      __m256i cmp_h = _mm256_cmpeq_epi8(a_high, b_high);

      uint32_t mask_l = (uint32_t)_mm256_movemask_epi8(cmp_l);
      uint32_t mask_h = (uint32_t)_mm256_movemask_epi8(cmp_h);

      return ((__mmask64)mask_h << 32) | mask_l;
    }

    static inline __mmask64 uv_cmplt_i8(__m512i a, __m512i b) {
      __m256i a_low  = _mm512_castsi512_si256(a);
      __m256i a_high = _mm512_extracti64x4_epi64(a, 1);
      __m256i b_low  = _mm512_castsi512_si256(b);
      __m256i b_high = _mm512_extracti64x4_epi64(b, 1);

      __m256i cmp_l = _mm256_cmpgt_epi8(b_low, a_low);
      __m256i cmp_h = _mm256_cmpgt_epi8(b_high, a_high);

      uint32_t mask_l = (uint32_t)_mm256_movemask_epi8(cmp_l);
      uint32_t mask_h = (uint32_t)_mm256_movemask_epi8(cmp_h);

      return ((__mmask64)mask_h << 32) | mask_l;
    }

    static inline __m512i uv_select_i8(__mmask64 mask, __m512i t, __m512i f) {
      __m256i t_low  = _mm512_castsi512_si256(t);
      __m256i t_high = _mm512_extracti64x4_epi64(t, 1);
      __m256i f_low  = _mm512_castsi512_si256(f);
      __m256i f_high = _mm512_extracti64x4_epi64(f, 1);


      uint32_t mask_l = (uint32_t)(mask & 0xFFFFFFFFULL);
      uint32_t mask_h = (uint32_t)(mask >> 32);

      alignas(32) uint8_t bytes_l[32];
      alignas(32) uint8_t bytes_h[32];
      for (int i = 0; i < 32; ++i) {
        bytes_l[i] = ((mask_l >> i) & 1) ? 0x80 : 0x00;
        bytes_h[i] = ((mask_h >> i) & 1) ? 0x80 : 0x00;
      }

      __m256i ctrl_l = _mm256_loadu_si256((const __m256i*)bytes_l);
      __m256i ctrl_h = _mm256_loadu_si256((const __m256i*)bytes_h);

      __m256i r_low  = _mm256_blendv_epi8(f_low, t_low, ctrl_l);
      __m256i r_high = _mm256_blendv_epi8(f_high, t_high, ctrl_h);

      return _mm512_inserti64x4(_mm512_castsi256_si512(r_low), r_high, 1);
    }
  #endif

  #if defined(__AVX512VBMI2__)
    #define uv_compress_i8(mask, a)  _mm512_maskz_compress_epi8(mask, a)

  // Baseline AVX-512F requires this inline macro mapping fallback:
  #else
    static inline __m512i uv_compress_i8(__mmask64 mask, __m512i a) {
      alignas(64) int8_t src[64];
      alignas(64) int8_t dst[64] = {0};
      _mm512_storeu_si512((void*)src, a);
      int d_idx = 0;
      for (int s_idx = 0; s_idx < 64; ++s_idx) {
        if ((mask >> s_idx) & 1) { dst[d_idx++] = src[s_idx]; }
      }
      return _mm512_loadu_si512((const void*)dst);
    }
  #endif

#elif defined(__AVX2__)
  #define uv_load_i8(ptr)        _mm256_load_si256((__m256i const*)(ptr))
  #define uv_loadu_i8(ptr)       _mm256_loadu_si256((__m256i const*)(ptr))
  #define uv_store_i8(ptr, v)    _mm256_store_si256((__m256i*)(ptr), (v))
  #define uv_storeu_i8(ptr, v)   _mm256_storeu_si256((__m256i*)(ptr), (v))

  #define uv_cmpeq_i8(a, b)      _mm256_cmpeq_epi8(a, b)
  #define uv_cmplt_i8(a, b)      _mm256_cmplt_epi8(a, b)

  #define uv_select_i8(mask, t, f)   _mm256_blendv_epi8(f, t, mask)
  #define uv_shuffle_i8(a, b)        _mm256_shuffle_epi8(a, b)

  static inline __m256i uv_permutexvar_i8(__m256i idx, __m256i a) {
    __m256i low_a  = _mm256_permute2x128_si256(a, a, 0x00); // [Low, Low]
    __m256i high_a = _mm256_permute2x128_si256(a, a, 0x11); // [High, High]
    
    __m256i shuffled_low  = _mm256_shuffle_epi8(low_a, idx);
    __m256i shuffled_high = _mm256_shuffle_epi8(high_a, idx);
    
    __m256i lane_select   = _mm256_slli_epi32(idx, 3);
    return _mm256_blendv_epi8(shuffled_low, shuffled_high, lane_select);
  }

  static inline __m256i uv_compress_i8(__m256i mask, __m256i a) {
    alignas(32) int8_t src[32];
    alignas(32) int8_t m_bytes[32];
    alignas(32) int8_t dst[32] = {0};
    _mm256_storeu_si256((__m256i*)src, a);
    _mm256_storeu_si256((__m256i*)m_bytes, mask);
    int d_idx = 0;
    for (int s_idx = 0; s_idx < 32; ++s_idx) {
      if (m_bytes[s_idx] & 0x80) { dst[d_idx++] = src[s_idx]; }
    }
    return _mm256_loadu_si256((const __m256i*)dst);
  }

#elif defined(__SSE2__)
  #define uv_load_i8(ptr)        _mm_load_si128((__m128i const*)(ptr))
  #define uv_loadu_i8(ptr)       _mm_loadu_si128((__m128i const*)(ptr))
  #define uv_store_i8(ptr, v)    _mm_store_si128((__m128i*)(ptr), (v))
  #define uv_storeu_i8(ptr, v)   _mm_storeu_si128((__m128i*)(ptr), (v))

  #define uv_cmpeq_i8(a, b)      _mm_cmpeq_epi8(a, b)
  #define uv_cmplt_i8(a, b)      _mm_cmpgt_epi8(b, a)

  #define uv_select_i8(m, t, f)  _mm_or_si128(_mm_and_si128(m, t), _mm_andnot_si128(m, f))

  #if defined(__SSSE3__)
    #define uv_shuffle_i8(a, b)      _mm_shuffle_epi8(a, b)
    #define uv_permutexvar_i8(i, a)  _mm_shuffle_epi8(a, i)
  #else
    static inline __m128i uv_shuffle_i8(__m128i a, __m128i b) {
      alignas(16) int8_t src[16];
      alignas(16) int8_t ctrl[16];
      alignas(16) int8_t dst[16];
      _mm_storeu_si128((__m128i*)src, a);
      _mm_storeu_si128((__m128i*)ctrl, b);
      for (int i = 0; i < 16; ++i) {
        dst[i] = (ctrl[i] & 0x80) ? 0 : src[ctrl[i] & 0x0F];
      }
      return _mm_loadu_si128((const __m128i*)dst);
    }
    #define uv_permutexvar_i8(idx, a) uv_shuffle_i8(a, idx)
  #endif

  static inline __m128i uv_compress_i8(__m128i mask, __m128i a) {
    alignas(16) int8_t src[16];
    alignas(16) int8_t m_bytes[16];
    alignas(16) int8_t dst[16] = {0};
    _mm_storeu_si128((__m128i*)src, a);
    _mm_storeu_si128((__m128i*)m_bytes, mask);
    int d_idx = 0;
    for (int s_idx = 0; s_idx < 16; ++s_idx) {
      if (m_bytes[s_idx] & 0x80) { dst[d_idx++] = src[s_idx]; }
    }
    return _mm_loadu_si128((const __m128i*)dst);
  }

#else
  #define uv_load_i8(ptr)        (*(const int8_t*)(ptr))
  #define uv_loadu_i8(ptr)       (*(const int8_t*)(ptr))
  #define uv_store_i8(ptr, v)    (*(int8_t*)(ptr) = (v))
  #define uv_storeu_i8(ptr, v)   (*(int8_t*)(ptr) = (v))

  #define uv_cmpeq_i8(a, b)      ((a) == (b))
  #define uv_cmplt_i8(a, b)      ((a) < (b))

  #define uv_select_i8(mask, t, f)   ((mask) ? (t) : (f))
  #define uv_shuffle_i8(a, b)        (((b) & 0x80) ? 0 : (a))
  #define uv_permutexvar_i8(idx, a)  (((idx) & 0x80) ? 0 : (a))
  #define uv_compress_i8(mask, a)    ((mask) ? (a) : 0)
#endif


/*
 * Scalar Bit Manipulation Primitives
 */
#if defined(__GNUC__) || defined(__clang__)
  #define uv_popcnt_u64(a)       (int)__builtin_popcountll(a)
  #define uv_tzcnt_u64(a)        (int)__builtin_ctzll(a)
  #define uv_lzcnt_u64(a)        (int)__builtin_clzll(a)
#elif defined(_MSC_VER)
  #include <intrin.h>
  #define uv_popcnt_u64(a)       (int)__popcnt64(a)
  #if defined(__AVX2__) || defined(__BMI__)
    #define uv_tzcnt_u64(a)      (int)_tzcnt_u64(a)
    #define uv_lzcnt_u64(a)      (int)_lzcnt_u64(a)
  #else
    // Legacy fallback
    static inline int uv_tzcnt_u64(unsigned long long a) {
      unsigned long index;
      return _BitScanForward64(&index, a) ? (int)index : 64;
    }
    static inline int uv_lzcnt_u64(unsigned long long a) {
      unsigned long index;
      return _BitScanReverse64(&index, a) ? (int)(63 - index) : 64;
    }
  #endif
#else
  // Software fallback for unknown compiler environments
  static inline int uv_popcnt_u64(uint64_t v) {
    v = v - ((v >> 1) & 0x5555555555555555ULL);
    v = (v & 0x3333333333333333ULL) + ((v >> 2) & 0x3333333333333333ULL);
    return (int)((((v + (v >> 4)) & 0xF0F0F0F0F0F0F0FULL) * 0x101010101010101ULL) >> 56);
  }
  static inline int uv_tzcnt_u64(uint64_t a) {
    if (a == 0) return 64;
    int n = 0;
    if ((a & 0xFFFFFFFFULL) == 0) { n += 32; a >>= 32; }
    if ((a & 0xFFFFULL) == 0)     { n += 16; a >>= 16; }
    if ((a & 0xFFULL) == 0)       { n += 8;  a >>= 8;  }
    if ((a & 0xFULL) == 0)        { n += 4;  a >>= 4;  }
    if ((a & 0x3ULL) == 0)        { n += 2;  a >>= 2;  }
    if ((a & 0x1ULL) == 0)        { n += 1; }
    return n;
  }
  static inline int uv_lzcnt_u64(uint64_t a) {
    if (a == 0) return 64;
    int n = 0;
    if ((a & 0xFFFFFFFF00000000ULL) == 0) { n += 32; a <<= 32; }
    if ((a & 0xFFFF000000000000ULL) == 0) { n += 16; a <<= 16; }
    if ((a & 0xFF00000000000000ULL) == 0) { n += 8;  a <<= 8;  }
    if ((a & 0xF000000000000000ULL) == 0) { n += 4;  a <<= 4;  }
    if ((a & 0xC000000000000000ULL) == 0) { n += 2;  a <<= 2;  }
    if ((a & 0x8000000000000000ULL) == 0) { n += 1; }
    return n;
  }
#endif

#endif // UVX_SIMD_H
