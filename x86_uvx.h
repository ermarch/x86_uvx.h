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
 *            `UV_LANES_32`. The implementation resolves completely at compile
 *            time via macro inlining with zero runtime abstraction overhead.
 *
 * @note      Supported Target Levels:
 *            - LEVEL 3: AVX-512 (512-bit registers, UV_LANES_32 = 16 for f32)
 *            - LEVEL 2: AVX2    (256-bit registers, UV_LANES_32 = 8  for f32)
 *            - LEVEL 1: SSE2    (128-bit registers, UV_LANES_32 = 4  for f32)
 *            - LEVEL 0:         (Scalar Fallback,   UV_LANES_32 = 1  for f32)
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

#ifndef UVX_SIMD_H
#define UVX_SIMD_H

#define __UVX_0__

 /**
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
  *     const size_t block_step = UNROLL_FACTOR * uv_lanes(32);
  *     for (; i + block_step <= size; i += block_step) {
  *         for (size_t u = 0; u < UNROLL_FACTOR; ++u) {
  *             size_t offset = i + (u * uv_lanes(32));
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

#if defined(__cplusplus)
extern "C" {
#endif

#if (defined(__x86_64__) || defined(__i386__))
# include <immintrin.h>
#endif
#include <stdalign.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>

/*
 * Globals
 */

// Maximum number of registers for any implementation.
// Which is usefule for definining arrays that should support all architectures.
#define UV_MAX_REGISTERS         32
#define uv_registers()           UV_REGISTERS
#define uv_max_registers()       UV_MAX_REGISTERS

// Number of lanes for a given register width.
#define UV_LANES_8               ((UV_REGISTERS>1) ? (UV_REGISTER_WIDTH/8) : 1)
#define UV_LANES_16              ((UV_REGISTERS>1) ? (UV_REGISTER_WIDTH/16) : 1)
#define UV_LANES_32              ((UV_REGISTERS>1) ? (UV_REGISTER_WIDTH/32) : 1)
#define UV_LANES_64              ((UV_REGISTERS>1) ? (UV_REGISTER_WIDTH/64) : 1)
#define uv_lanes(a)              ((UV_REGISTERS>1) ? (UV_REGISTER_WIDTH/(a)): 1)

// Maximum number of lanes for any implementation.
// Which is usefule for definining arrays that should support all architectures.
#define UV_MAX_LANES_8           (512/8)
#define UV_MAX_LANES_16          (512/16)
#define UV_MAX_LANES_32          (512/32)
#define UV_MAX_LANES_64          (512/64)
#define uv_max_lanes(a)          (512/(a))

// Memory alignment for best performance.
#define UV_ALIGNMENT             ((UV_REGISTERS > 1) ? UV_LANES_8 : 8)
#define UV_ALIGNMENT_MASK        (UV_ALIGNMENT-1)
#define uv_alignment()           UV_ALIGNMENT
#define uv_alignment_mask()      UV_ALIGNMENT_MASK

// Prefetching hints for complex data handling.
// Prefetching for simple data layouts is best left tot the compiler.
//
// UV_HINT_T0:  Best when the data will be used immediately and repeatedly.
// UV_HINT_T1:  Best when the data will be needed soon, but you don't want to
//              evict data currently being used in L1.
// UV_HINT_T2:  Best for data that won't be required for a longer period of time
// UV_HINT_NTA: Best for streaming data that you will read exactly once.
#define UV_HINT_T0               _MM_HINT_T0
#define UV_HINT_T1               _MM_HINT_T1
#define UV_HINT_T2               _MM_HINT_T2
#define UV_HINT_NTA              _MM_HINT_NTA
#define uv_prefetch(ptr, hint)   _mm_prefetch((const char*)(ptr), (hint))

#if defined(__AVX10_1_512__) || defined(__EVEX512__) || defined(__AVX512F__)
    #define UV512
#elif defined(__AVX10_1_256__) || defined(__EVEX256__)
#elif defined(__AVX2__)
    #define UV256
#endif


#if defined(__AVX10__) || defined(__AVX512F__)
  #define UV_REGISTERS          32
  #if defined(UV512)
    #define UV_REGISTER_WIDTH  512

    typedef __mmask16 v_mask;
    typedef __mmask8 v_mask_f64;
    typedef __mmask64 v_mask_i8;

    typedef __m512d v_f64;
    typedef __m512  v_f32;

    typedef __m512i v_i64;
    typedef __m512i v_i32;
    typedef __m512i v_i16;
    typedef __m512i v_i8;
  #else
    #define UV_REGISTER_WIDTH  256

    typedef __mmask8  v_mask;
    typedef __mmask4  v_mask_f64;
    typedef __mmask32 v_mask_i8;

    typedef __m256d v_f64;
    typedef __m256  v_f32;

    typedef __m256i v_i64;
    typedef __m256i v_i32;
    typedef __m256i v_i16;
    typedef __m256i v_i8;
  #endif
  #define uv_clear_lanes()  _mm256_zeroupper()

#elif defined(__AVX2__) || defined(__FMA__)
  #define UV_REGISTERS        16
  #define UV_REGISTER_WIDTH  256

  typedef __m256  v_mask;
  typedef __m256d v_mask_f64;
  typedef __m256i v_mask_i8;

  typedef __m256d v_f64;
  typedef __m256  v_f32;

  typedef __m256i v_i64;
  typedef __m256i v_i32;
  typedef __m256i v_i16;
  typedef __m256i v_i8;

  #define uv_clear_lanes() _mm256_zeroupper()

#elif defined(__SSE2__)
  #if defined(__i386__) || defined(_M_IX86) || defined(__SSE__)
    #define UV_REGISTERS       8
  #else
    #define UV_REGISTERS      16
  #endif
  #define UV_REGISTER_WIDTH  128

  typedef __m128  v_mask;
  typedef __m128d v_mask_f64;
  typedef __m128i v_mask_i8;

  typedef __m128d v_f64;
  typedef __m128  v_f32;

  typedef __m128i v_i64;
  typedef __m128i v_i32;
  typedef __m128i v_i16;
  typedef __m128i v_i8;

  #define uv_clear_lanes() ((void)0)

#else
  #define UV_REGISTERS          1
  #define UV_REGISTER_WIDTH    32

  typedef int     v_mask;
  typedef int     v_mask_f64;
  typedef int     v_mask_i8;

  typedef double  v_f64;
  typedef float   v_f32;

  typedef int64_t v_i64;
  typedef int32_t v_i32;
  typedef int16_t v_i16;
  typedef int8_t  v_i8;

  #define uv_clear_lanes() ((void)0)
#endif


/*
 * Single-Precision floating point support
 */
#if defined(__AVX10__) || defined(__AVX512F__)
  // AVX-512 hardware ignores standard alignment requirements for baseline
  // performance; maps uniformly
  #define uv_load_f32(ptr)       _mm512_loadu_ps(ptr)
  #define uv_loadu_f32(ptr)      _mm512_loadu_ps(ptr)
  #define uv_store_f32(ptr, v)   _mm512_storeu_ps(ptr, v)
  #define uv_storeu_f32(ptr, v)  _mm512_storeu_ps(ptr, v)
  #define uv_setzero_f32()       _mm512_setzero_ps()
  #define uv_dup_f32(v)          _mm512_set1_ps(v)

  #define uv_add_f32(a, b)       _mm512_add_ps(a, b)
  #define uv_sub_f32(a, b)       _mm512_sub_ps(a, b)
  #define uv_mul_f32(a, b)       _mm512_mul_ps(a, b)
  #define uv_div_f32(a, b)       _mm512_div_ps(a, b)
  #define uv_max_f32(a, b)       _mm512_max_ps(a, b)
  #define uv_min_f32(a, b)       _mm512_min_ps(a, b)
  #define uv_abs_f32(a)          _mm512_abs_ps(a)
  static inline v_f32 uv_neg_f32(v_f32 v) {
    __m512i mask = _mm512_set1_epi32(0x80000000);
    return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(v), mask));
  }

  #if defined(__AVX512DQ__) || defined(__AVX10__) || defined(_MSC_VER)
    static inline v_f32 uv_not_f32(v_f32 v) {
      __m512i a_int = _mm512_castps_si512(v);
      return _mm512_castsi512_ps(_mm512_ternarylogic_epi32(a_int, a_int, a_int, 0x55));
    }
    #define uv_and_f32(a, b)     _mm512_and_ps(a, b)
    #define uv_andnot_f32(a, b)  _mm512_andnot_ps(a, b)
    #define uv_or_f32(a, b)      _mm512_or_ps(a, b)
    #define uv_xor_f32(a, b)     _mm512_xor_ps(a, b)
  #else
    static inline v_f32 uv_not_f32(v_f32 v) {
      return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(v), _mm512_castps_si512(_mm512_set1_ps(-1.0f))));
    }
    static inline v_f32 uv_and_f32(v_f32 a, v_f32 b) {
      return _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(a), _mm512_castps_si512(b)));
    }
    static inline v_f32 uv_andnot_f32(v_f32 a, v_f32 b) {
      return _mm512_castsi512_ps(_mm512_andnot_si512(_mm512_castps_si512(a), _mm512_castps_si512(b)));
    }
    static inline v_f32 uv_or_f32(v_f32 a, v_f32 b) {
      return _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(a), _mm512_castps_si512(b)));
    }
    static inline v_f32 uv_xor_f32(v_f32 a, v_f32 b) {
      return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(a), _mm512_castps_si512(b)));
    }
  #endif

  #define uv_cmpeq_f32(a, b)     _mm512_cmp_ps_mask(a, b, _CMP_EQ_OS)
  #define uv_cmplt_f32(a, b)     _mm512_cmp_ps_mask(a, b, _CMP_LT_OS)
  #define uv_cmple_f32(a, b)     _mm512_cmp_ps_mask(a, b, _CMP_LE_OS)
  #define uv_cmpne_f32(a, b)     _mm512_cmp_ps_mask(a, b, _CMP_NEQ_US)
  #define uv_cmpnlt_f32(a, b)    _mm512_cmp_ps_mask(a, b, _CMP_NLT_US)
  #define uv_cmpge_f32(a, b)     _mm512_cmp_ps_mask(a, b, _CMP_GE_OS)
  #define uv_cmpnle_f32(a, b)    _mm512_cmp_ps_mask(a, b, _CMP_NLE_US)
  #define uv_cmpgt_f32(a, b)     _mm512_cmp_ps_mask(a, b, _CMP_GT_OS)

  #define uv_select_f32(m, t, f) _mm512_mask_blend_ps(m, f, t)

  #define uv_cvt_f32_i32(v)      _mm512_cvtps_epi32(v)
  #define uv_cvt_i32_f32(v)      _mm512_cvtepi32_ps(v)

  #define uv_reduce_add_f32(v)   _mm512_reduce_add_ps(v)
  #define uv_reduce_min_f32(v)   _mm512_reduce_min_ps(v)
  #define uv_reduce_max_f32(v)   _mm512_reduce_max_ps(v)

#elif defined (__FMA__) || defined(__AVX2__)
  #define uv_load_f32(ptr)       _mm256_load_ps(ptr)
  #define uv_loadu_f32(ptr)      _mm256_loadu_ps(ptr)
  #define uv_store_f32(ptr, v)   _mm256_store_ps(ptr, v)
  #define uv_storeu_f32(ptr, v)  _mm256_storeu_ps(ptr, v)
  #define uv_setzero_f32()       _mm256_setzero_ps()
  #define uv_dup_f32(v)          _mm256_set1_ps(v)

  #define uv_add_f32(a, b)       _mm256_add_ps(a, b)
  #define uv_sub_f32(a, b)       _mm256_sub_ps(a, b)
  #define uv_mul_f32(a, b)       _mm256_mul_ps(a, b)
  #define uv_div_f32(a, b)       _mm256_div_ps(a, b)
  #define uv_max_f32(a, b)       _mm256_max_ps(a, b)
  #define uv_min_f32(a, b)       _mm256_min_ps(a, b)
  static inline v_f32 uv_abs_f32(v_f32 v) {
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    return _mm256_andnot_ps(sign_mask, v);
  }
  static inline v_f32 uv_neg_f32(v_f32 v) {
    __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
    return _mm256_xor_ps(v, sign_mask);
  }

  static inline v_f32 uv_not_f32(v_f32 v) {
    return _mm256_andnot_ps(v, _mm256_cmp_ps(v, v, _CMP_EQ_OS));
  }
  #define uv_and_f32(a, b)       _mm256_and_ps(a, b)
  #define uv_andnot_f32(a, b)    _mm256_andnot_ps(a, b)
  #define uv_or_f32(a, b)        _mm256_or_ps(a, b)
  #define uv_xor_f32(a, b)       _mm256_xor_ps(a, b)

  #define uv_cmpeq_f32(a, b)     _mm256_cmp_ps(a, b, _CMP_EQ_OS)
  #define uv_cmplt_f32(a, b)     _mm256_cmp_ps(a, b, _CMP_LT_OS)
  #define uv_cmple_f32(a, b)     _mm256_cmp_ps(a, b, _CMP_LE_OS)
  #define uv_cmpne_f32(a, b)     _mm256_cmp_ps(a, b, _CMP_NEQ_US)
  #define uv_cmpnlt_f32(a, b)    _mm256_cmp_ps(a, b, _CMP_NLT_US)
  #define uv_cmpge_f32(a, b)     _mm256_cmp_ps(a, b, _CMP_GE_OS)
  #define uv_cmpnle_f32(a, b)    _mm256_cmp_ps(a, b, _CMP_NLE_US)
  #define uv_cmpgt_f32(a, b)     _mm256_cmp_ps(a, b, _CMP_GT_OS)

  #if defined(__AVX10__)
    #define uv_select_f32(m, t, f) _mm256_mask_blend_ps(m, f, t)
  #else
    #define uv_select_f32(m, t, f) _mm256_blendv_ps(f, t, (__m256)(m))
  #endif

  #define uv_cvt_f32_i32(v)      _mm256_cvtps_epi32(v)
  #define uv_cvt_i32_f32(v)      _mm256_cvtepi32_ps(v)

  static inline float uv_reduce_add_f32(v_f32 v)
  {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 sums  = _mm_add_ps(vlow, vhigh);
    __m128 shuf  = _mm_movehl_ps(sums, sums);
    sums = _mm_add_ps(sums, shuf);
    shuf = _mm_shuffle_ps(sums, sums, _MM_SHUFFLE(1, 1, 1, 1));
    sums = _mm_add_ps(sums, shuf);
    return _mm_cvtss_f32(sums);
  }
  static inline float uv_reduce_min_f32(v_f32 v)
  {
    __m128 low  = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 min128 = _mm_min_ps(low, high);
    __m128 shuff1 = _mm_movehl_ps(min128, min128);
    __m128 min64  = _mm_min_ps(min128, shuff1);
    __m128 shuff2 = _mm_permute_ps(min64, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 min32  = _mm_min_ps(min64, shuff2);
    return _mm_cvtss_f32(min32);
  }
  static inline float uv_reduce_max_f32(v_f32 v)
  {
    __m128 low  = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 max128 = _mm_max_ps(low, high);
    __m128 shuff1 = _mm_movehl_ps(max128, max128);
    __m128 max64  = _mm_max_ps(max128, shuff1);
    __m128 shuff2 = _mm_permute_ps(max64, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 max32  = _mm_max_ps(max64, shuff2);
    return _mm_cvtss_f32(max32);
  }

#elif defined(__SSE2__)

  #define uv_load_f32(ptr)       _mm_load_ps(ptr)
  #define uv_loadu_f32(ptr)      _mm_loadu_ps(ptr)
  #define uv_store_f32(ptr, v)   _mm_store_ps(ptr, v)
  #define uv_storeu_f32(ptr, v)  _mm_storeu_ps(ptr, v)
  #define uv_setzero_f32()       _mm_setzero_ps()
  #define uv_dup_f32(v)          _mm_set1_ps(v)

  #define uv_add_f32(a, b)       _mm_add_ps(a, b)
  #define uv_sub_f32(a, b)       _mm_sub_ps(a, b)
  #define uv_mul_f32(a, b)       _mm_mul_ps(a, b)
  #define uv_div_f32(a, b)       _mm_div_ps(a, b)
  #define uv_max_f32(a, b)       _mm_max_ps(a, b)
  #define uv_min_f32(a, b)       _mm_min_ps(a, b)
  static inline v_f32 uv_abs_f32(v_f32 v) {
    const __m128 sign_mask = _mm_set1_ps(-0.0f);
    return _mm_andnot_ps(sign_mask, v);
  }
  static inline v_f32 uv_neg_f32(v_f32 v) {
    __m128 sign_mask = _mm_set1_ps(-0.0);
    return _mm_xor_ps(v, sign_mask);
  }

  static inline v_f32 uv_not_f32(v_f32 v) {
    return _mm_andnot_ps(v, _mm_cmpeq_ps(v, v));
  }
  #define uv_and_f32(a, b)       _mm_and_ps(a, b)
  #define uv_andnot_f32(a, b)    _mm_andnot_ps(a, b)
  #define uv_or_f32(a, b)        _mm_or_ps(a, b)
  #define uv_xor_f32(a, b)       _mm_xor_ps(a, b)

  #define uv_cmpeq_f32(a, b)     _mm_cmpeq_ps(a, b)
  #define uv_cmplt_f32(a, b)     _mm_cmplt_ps(a, b)
  #define uv_cmple_f32(a, b)     _mm_cmple_ps(a, b)
  #define uv_cmpne_f32(a, b)     _mm_cmpneq_ps(a, b)
  #define uv_cmpnlt_f32(a, b)    _mm_cmpnlt_ps(a, b)
  #define uv_cmpge_f32(a, b)     _mm_cmpge_ps(a, b)
  #define uv_cmpnle_f32(a, b)    _mm_cmpnle_ps(a, b)
  #define uv_cmpgt_f32(a, b)     _mm_cmpgt_ps(a, b)

  #if defined(__SSE4_1__)
    #define uv_select_f32(m, t, f) _mm_blendv_ps(f, t, m)
  #else
    #define uv_select_f32(m, t, f) _mm_or_ps(_mm_andnot_ps(m, f), _mm_and_ps(m, t))
  #endif

  #define uv_cvt_f32_i32(v)      _mm_cvtps_epi32(v)
  #define uv_cvt_i32_f32(v)      _mm_cvtepi32_ps(v)

  static inline float uv_reduce_add_f32(v_f32 v)
  {
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
  }
  static inline float uv_reduce_min_f32(v_f32 v) {
    __m128 shuff1 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 min64 = _mm_min_ps(v, shuff1);
    __m128 shuff2 = _mm_shuffle_ps(min64, min64, _MM_SHUFFLE(1, 0, 3, 2));
    __m128 min32 = _mm_min_ps(min64, shuff2);
    float result;
    _mm_store_ss(&result, min32);
    return result;
  }
  static inline float uv_reduce_max_f32(v_f32 v) {
    __m128 shuff1 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 max64 = _mm_max_ps(v, shuff1);
    __m128 shuff2 = _mm_shuffle_ps(max64, max64, _MM_SHUFFLE(1, 0, 3, 2));
    __m128 max32 = _mm_max_ps(max64, shuff2);
    float result;
    _mm_store_ss(&result, max32);
    return result;
  }

#else
  #define uv_load_f32(ptr)       (*(const float*)(ptr))
  #define uv_loadu_f32(ptr)      (*(const float*)(ptr))
  #define uv_store_f32(ptr, v)   (*(float*)(ptr) = (v))
  #define uv_storeu_f32(ptr, v)  (*(float*)(ptr) = (v))
  #define uv_setzero_f32()       (0.0f)
  #define uv_dup_f32(v)          (v)

  #define uv_add_f32(a, b)       ((a) + (b))
  #define uv_sub_f32(a, b)       ((a) - (b))
  #define uv_mul_f32(a, b)       ((a) * (b))
  #define uv_div_f32(a, b)       ((a) / (b))
  #define uv_max_f32(a, b)       ((a) > (b) ? (a) : (b))
  #define uv_min_f32(a, b)       ((a) < (b) ? (a) : (b))
  #define uv_abs_f32(v)          (fabsf(v))
  #define uv_neg_f32(v)          (-(v))

  static inline v_f32 uv_not_f32(v_f32 v) {
    uint32_t ai, ri;
    memcpy(&ai, &v, 4);
    ri = ~ai;
    v_f32 res;
    memcpy(&res, &ri, 4);
    return res;
  }
  static inline v_f32 uv_and_f32(v_f32 a, v_f32 b) {
    uint32_t ai, bi, ri;
    memcpy(&ai, &a, 4);
    memcpy(&bi, &b, 4);
    ri = ai & bi;
    v_f32 res;
    memcpy(&res, &ri, 4);
    return res;
  }

  static inline v_f32 uv_andnot_f32(v_f32 a, v_f32 b) {
    uint32_t ai, bi, ri;
    memcpy(&ai, &a, 4);
    memcpy(&bi, &b, 4);
    ri = ~ai & bi;
    v_f32 res;
    memcpy(&res, &ri, 4);
    return res;
  }

  static inline v_f32 uv_or_f32(v_f32 a, v_f32 b) {
    uint32_t ai, bi, ri;
    memcpy(&ai, &a, 4);
    memcpy(&bi, &b, 4);
    ri = ai | bi;
    v_f32 res;
    memcpy(&res, &ri, 4);
    return res;
  }

  static inline v_f32 uv_xor_f32(v_f32 a, v_f32 b) {
    uint32_t ai, bi, ri;
    memcpy(&ai, &a, 4);
    memcpy(&bi, &b, 4);
    ri = ai ^ bi;
    v_f32 res;
    memcpy(&res, &ri, 4);
    return res;
  }

  #define uv_cmpeq_f32(a, b)     ((a) == (b))
  #define uv_cmplt_f32(a, b)     ((a) < (b))
  #define uv_cmple_f32(a, b)     ((a) <= (b))
  #define uv_cmpne_f32(a, b)     ((a) != (b))
  #define uv_cmpnlt_f32(a, b)    (!((a) < (b)))
  #define uv_cmpge_f32(a, b)     ((a) >= (b))
  #define uv_cmpnle_f32(a, b)    (!((a) <= (b)))
  #define uv_cmpgt_f32(a, b)     ((a) > (b))

  #define uv_select_f32(m, t, f) ((m) ? (t) : (f))

  static inline int32_t uv_cvt_f32_i32(float v) {
    v = v + 12582912.0f;
    return *(int32_t*)&v - 0x4B400000;
  }
  static inline float uv_cvt_i32_f32(int32_t v) {
    v = v + 0x4B400000;
    return *(float*)&v - 12582912.0f;
  }

  #define uv_reduce_add_f32(v)   (v)
  #define uv_reduce_min_f32(v)   (v)
  #define uv_reduce_max_f32(v)   (v)

#endif


/*
 * Single-Precision and Double-Precision Float-Multiply-Add support
 */
#if defined(__AVX10__) || defined(__AVX512F__)
  #define uv_fmadd_f32(a, b, c)    _mm512_fmadd_ps(a, b, c)
  #define uv_fmsub_f32(a, b, c)    _mm512_fmsub_ps(a, b, c)
  #define uv_fmadd_f64(a, b, c)    _mm512_fmadd_pd(a, b, c)
  #define uv_fmsub_f64(a, b, c)    _mm512_fmsub_pd(a, b, c)

#elif defined(__FMA__)
  #define uv_fmadd_f32(a, b, c)    _mm256_fmadd_ps(a, b, c)
  #define uv_fmsub_f32(a, b, c)    _mm256_fmsub_ps(a, b, c)
  #define uv_fmadd_f64(a, b, c)    _mm256_fmadd_pd(a, b, c)
  #define uv_fmsub_f64(a, b, c)    _mm256_fmsub_pd(a, b, c)

#elif defined(__AVX2__)
  static inline v_f32 uv_fmadd_f32(v_f32 a, v_f32 b, v_f32 c) {
      __m256 intermediate_mul = _mm256_mul_ps(a, b);
      return _mm256_add_ps(intermediate_mul, c);
  }
  #define uv_fmsub_f32(a, b, c)    uv_fmadd_f32(a, b, uv_neg_f32(c))
  static inline v_f64 uv_fmadd_f64(v_f64 a, v_f64 b, v_f64 c) {
      return _mm256_add_pd(_mm256_mul_pd(a, b), c);
  }
  #define uv_fmsub_f64(a, b, c)    uv_fmadd_f64(a, b, uv_neg_f64(c))

#elif defined(__SSE2__)
  static inline v_f32 uv_fmadd_f32(v_f32 a, v_f32 b, v_f32 c) {
      return _mm_add_ps(_mm_mul_ps(a, b), c);
  }
  #define uv_fmsub_f32(a, b, c)    uv_fmadd_f32(a, b, uv_neg_f32(c))
  static inline v_f64 uv_fmadd_f64(v_f64 a, v_f64 b, v_f64 c) {
      return _mm_add_pd(_mm_mul_pd(a, b), c);
  }
  #define uv_fmsub_f64(a, b, c)    uv_fmadd_f64(a, b, uv_neg_f64(c))

#else
  #define uv_fmadd_f32(a, b, c)    (((a) * (b)) + (c))
  #define uv_fmsub_f32(a, b, c)    (((a) * (b)) - (c))
  #define uv_fmadd_f64(a, b, c)    (((a) * (b)) + (c))
  #define uv_fmsub_f64(a, b, c)    (((a) * (b)) - (c))

#endif


/*
 * Double-Precision floating point support
 */
#if defined(__AVX10__) || defined(__AVX512F__)
  #define uv_load_f64(ptr)       _mm512_loadu_pd(ptr)
  #define uv_loadu_f64(ptr)      _mm512_loadu_pd(ptr)
  #define uv_store_f64(ptr, v)   _mm512_storeu_pd(ptr, v)
  #define uv_storeu_f64(ptr, v)  _mm512_storeu_pd(ptr, v)
  #define uv_setzero_f64()       _mm512_setzero_pd()
  #define uv_dup_f64(v)          _mm512_set1_pd(v)

  #define uv_add_f64(a, b)       _mm512_add_pd(a, b)
  #define uv_sub_f64(a, b)       _mm512_sub_pd(a, b)
  #define uv_mul_f64(a, b)       _mm512_mul_pd(a, b)
  #define uv_div_f64(a, b)       _mm512_div_pd(a, b)
  #define uv_max_f64(a, b)       _mm512_max_pd(a, b)
  #define uv_min_f64(a, b)       _mm512_min_pd(a, b)
  #define uv_abs_f64(v)          _mm512_abs_pd(v)
  static inline v_f64 uv_neg_f64(v_f64 v) {
    __m512i mask = _mm512_set1_epi64(0x8000000000000000ULL);
    return _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(v), mask));
  }

  static inline v_f64 uv_not_f64(v_f64 v) {
    __m512i a = _mm512_castpd_si512(v);
    return _mm512_castsi512_pd(_mm512_ternarylogic_epi64(v, v, v, 0x55));
  }
  #if defined(__AVX512DQ__) || defined(__AVX10__) || defined(_MSC_VER)
    #define uv_and_f64(a, b)     _mm512_and_pd(a, b)
    #define uv_andnot_f64(a, b)  _mm512_andnot_pd(a, b)
    #define uv_or_f64(a, b)        _mm512_or_pd(a, b)
    #define uv_xor_f64(a, b)       _mm512_xor_pd(a, b)
  #else
    static inline v_f64 uv_and_f64(v_f64 a, v_f64 b) {
      return _mm512_castsi512_pd(_mm512_and_si512(_mm512_castpd_si512(a), _mm512_castpd_si512(b)));
    }
    static inline v_f64 uv_andnot_f64(v_f64 a, v_f64 b) {
      return _mm512_castsi512_pd(_mm512_andnot_si512(_mm512_castpd_si512(a), _mm512_castpd_si512(b)));
    }
    static inline v_f64 uv_or_f64(v_f64 a, v_f64 b) {
      return _mm512_castsi512_pd(_mm512_or_si512(_mm512_castpd_si512(a), _mm512_castpd_si512(b)));
    }
    static inline v_f64 uv_xor_f64(v_f64 a, v_f64 b) {
      return _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(a), _mm512_castpd_si512(b)));
    }
  #endif

  #define uv_cmpeq_f64(a, b)     _mm512_cmp_pd_mask(a, b, _CMP_EQ_OS)
  #define uv_cmplt_f64(a, b)     _mm512_cmp_pd_mask(a, b, _CMP_LT_OS)
  #define uv_cmple_f64(a, b)     _mm512_cmp_pd_mask(a, b, _CMP_LE_OS)
  #define uv_cmpne_f64(a, b)     _mm512_cmp_pd_mask(a, b, _CMP_NEQ_US)
  #define uv_cmpnlt_f64(a, b)    _mm512_cmp_pd_mask(a, b, _CMP_NLT_US)
  #define uv_cmpge_f64(a, b)     _mm512_cmp_pd_mask(a, b, _CMP_GE_OS)
  #define uv_cmpnle_f64(a, b)    _mm512_cmp_pd_mask(a, b, _CMP_NLE_US)
  #define uv_cmpgt_f64(a, b)     _mm512_cmp_ps_mask(a, b, _CMP_GT_OS)

  #define uv_select_f64(m, t, f) _mm512_mask_blend_pd(m, f, t)

#elif defined(__FMA__) || defined(__AVX2__)
  #define uv_load_f64(ptr)       _mm256_load_pd(ptr)
  #define uv_loadu_f64(ptr)      _mm256_loadu_pd(ptr)
  #define uv_store_f64(ptr, v)   _mm256_store_pd(ptr, v)
  #define uv_storeu_f64(ptr, v)  _mm256_storeu_pd(ptr, v)
  #define uv_setzero_f64()       _mm256_setzero_pd()
  #define uv_dup_f64(v)          _mm256_set1_pd(v)

  #define uv_add_f64(a, b)       _mm256_add_pd(a, b)
  #define uv_sub_f64(a, b)       _mm256_sub_pd(a, b)
  #define uv_mul_f64(a, b)       _mm256_mul_pd(a, b)
  #define uv_div_f64(a, b)       _mm256_div_pd(a, b)
  #define uv_max_f64(a, b)       _mm256_max_pd(a, b)
  #define uv_min_f64(a, b)       _mm256_min_pd(a, b)
  static inline v_f64 uv_abs_f64(__m256d v) {
    __m256d sign_mask = _mm256_set1_pd(-0.0);
    return _mm256_andnot_pd(sign_mask, v);
  }
  static inline v_f64 uv_neg_f64(v_f64 v) {
    __m256d sign_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000ULL));
    return _mm256_xor_pd(v, sign_mask);
  }

  static inline v_f64 uv_not_f64(v_f64 v) {
    return _mm256_andnot_pd(v, _mm256_cmp_pd(v, v, _CMP_EQ_OS));
  }
  #define uv_and_f64(a, b)       _mm256_and_pd(a, b)
  #define uv_andnot_f64(a, b)    _mm256_andnot_pd(a, b)
  #define uv_or_f64(a, b)        _mm256_or_pd(a, b)
  #define uv_xor_f64(a, b)       _mm256_xor_pd(a, b)

  #define uv_cmpeq_f64(a, b)     _mm256_cmp_pd(a, b, _CMP_EQ_OS)
  #define uv_cmplt_f64(a, b)     _mm256_cmp_pd(a, b, _CMP_LT_OS)
  #define uv_cmple_f64(a, b)     _mm256_cmp_pd(a, b, _CMP_LE_OS)
  #define uv_cmpne_f64(a, b)     _mm256_cmp_pd(a, b, _CMP_NEQ_US)
  #define uv_cmpnlt_f64(a, b)    _mm256_cmp_pd(a, b, _CMP_NLT_US)
  #define uv_cmpge_f64(a, b)     _mm256_cmp_pd(a, b, _CMP_GE_OS)
  #define uv_cmpnle_f64(a, b)    _mm256_cmp_pd(a, b, _CMP_NLE_US)
  #define uv_cmpgt_f64(a, b)     _mm256_cmp_pd(a, b, _CMP_GT_OS)

  #if defined(__AVX10__)
    #define uv_select_f64(m, t, f) _mm256_mask_blend_pd(m, f, t)
  #else
    #define uv_select_f64(m, t, f) _mm256_blendv_pd(f, t, (__m256d)(m))
  #endif

#elif defined(__SSE2__)
  #define uv_load_f64(ptr)       _mm_load_pd(ptr)
  #define uv_loadu_f64(ptr)      _mm_loadu_pd(ptr)
  #define uv_store_f64(ptr, v)   _mm_store_pd(ptr, v)
  #define uv_storeu_f64(ptr, v)  _mm_storeu_pd(ptr, v)
  #define uv_setzero_f64()       _mm_setzero_pd()
  #define uv_dup_f64(v)          _mm_set1_pd(v)

  #define uv_add_f64(a, b)       _mm_add_pd(a, b)
  #define uv_sub_f64(a, b)       _mm_sub_pd(a, b)
  #define uv_mul_f64(a, b)       _mm_mul_pd(a, b)
  #define uv_div_f64(a, b)       _mm_div_pd(a, b)
  #define uv_max_f64(a, b)       _mm_max_pd(a, b)
  #define uv_min_f64(a, b)       _mm_min_pd(a, b)
  static inline v_f64 uv_abs_f64(v_f64 v) {
    const __m128d sign_mask = _mm_set1_pd(-0.0f);
    return _mm_andnot_pd(sign_mask, v);
  }
  static inline v_f64 uv_neg_f64(v_f64 v) {
    __m128d mask = _mm_set1_pd(-0.0);
    return _mm_xor_pd(v, mask);
  }

  static inline v_f64 uv_not_f64(v_f64 v) {
    return _mm_andnot_pd(v, _mm_cmpeq_pd(v, v));
  }
  #define uv_and_f64(a, b)       _mm_and_pd(a, b)
  #define uv_andnot_f64(a, b)    _mm_andnot_pd(a, b)
  #define uv_or_f64(a, b)        _mm_or_pd(a, b)
  #define uv_xor_f64(a, b)       _mm_xor_pd(a, b)

  static inline v_f64 uv_cmpeq_f64(v_f64 a, v_f64 b) {
      return _mm_cmpeq_pd(a, b);
  }
  static inline v_f64 uv_cmpgt_f64(v_f64 a, v_f64 b) {
      return _mm_cmpgt_pd(a, b);
  }
  static inline v_f64 uv_cmplt_f64(v_f64 a, v_f64 b) {
      return _mm_cmpgt_pd(b, a);
  }
  static inline v_f64 uv_cmpne_f64(v_f64 a, v_f64 b) {
      v_f64 eq = _mm_cmpeq_pd(a, b);
      return _mm_xor_pd(eq, _mm_set1_pd(-1.0));
  }
  static inline v_f64 uv_cmple_f64(v_f64 a, v_f64 b) {
      v_f64 gt = _mm_cmpgt_pd(a, b);
      return _mm_xor_pd(gt, _mm_set1_pd(-1.0));
  }
  static inline v_f64 uv_cmpge_f64(v_f64 a, v_f64 b) {
      v_f64 gt = _mm_cmpgt_pd(b, a);
      return _mm_xor_pd(gt, _mm_set1_pd(-1.0));
  }
  static inline v_f64 uv_cmpnlt_f64(v_f64 a, v_f64 b) {
      return uv_cmpge_f64(a, b);
  }
  static inline v_f64 uv_cmpnle_f64(v_f64 a, v_f64 b) {
      return _mm_cmpgt_pd(a, b);
  }

  #if defined(__SSE4_1__)
    #define uv_select_f64(m, t, f) _mm_blendv_pd(f, t, m)
  #else
    #define uv_select_f64(m, t, f) _mm_or_pd(_mm_andnot_pd(m, f), _mm_and_pd(m, t))
  #endif

#else
  #define uv_load_f64(ptr)       (*(const double*)(ptr))
  #define uv_loadu_f64(ptr)      (*(const double*)(ptr))
  #define uv_store_f64(ptr, v)   (*(double*)(ptr) = (v))
  #define uv_storeu_f64(ptr, v)  (*(double*)(ptr) = (v))
  #define uv_setzero_f64()       (0.0)
  #define uv_dup_f64(v)          (v)

  #define uv_add_f64(a, b)       ((a) + (b))
  #define uv_sub_f64(a, b)       ((a) - (b))
  #define uv_mul_f64(a, b)       ((a) * (b))
  #define uv_div_f64(a, b)       ((a) / (b))
  #define uv_max_f64(a, b)       ((a) > (b) ? (a) : (b))
  #define uv_min_f64(a, b)       ((a) < (b) ? (a) : (b))
  #define uv_abs_f64(v)          (fabsf(v))
  #define uv_neg_f64(v)          (-(v))

  static inline v_f64 uv_not_f64(v_f64 v) {
    uint64_t ai, ri;
    memcpy(&ai, &v, 8);
    ri = ~ai;
    v_f64 res;
    memcpy(&res, &ri, 8);
    return res;
  }
  static inline v_f64 uv_and_f64(v_f64 a, v_f64 b) {
    uint64_t ai, bi, ri;
    memcpy(&ai, &a, 8);
    memcpy(&bi, &b, 8);
    ri = ai & bi;
    v_f64 res;
    memcpy(&res, &ri, 8);
    return res;
  }

  static inline v_f64 uv_andnot_f64(v_f64 a, v_f64 b) {
    uint64_t ai, bi, ri;
    memcpy(&ai, &a, 8);
    memcpy(&bi, &b, 8);
    ri = ~ai & bi;
    v_f64 res;
    memcpy(&res, &ri, 8);
    return res;
  }

  static inline v_f64 uv_or_f64(v_f64 a, v_f64 b) {
    uint64_t ai, bi, ri;
    memcpy(&ai, &a, 8);
    memcpy(&bi, &b, 8);
    ri = ai | bi;
    v_f64 res;
    memcpy(&res, &ri, 8);
    return res;
  }

  static inline v_f64 uv_xor_f64(v_f64 a, v_f64 b) {
    uint64_t ai, bi, ri;
    memcpy(&ai, &a, 8);
    memcpy(&bi, &b, 8);
    ri = ai ^ bi;
    v_f64 res;
    memcpy(&res, &ri, 8);
    return res;
  }

  #define uv_cmpeq_f64(a, b)     ((a) == (b))
  #define uv_cmplt_f64(a, b)     ((a) < (b))
  #define uv_cmple_f64(a, b)     ((a) <= (b))
  #define uv_cmpne_f64(a, b)     ((a) != (b))
  #define uv_cmpnlt_f64(a, b)    (!((a) < (b)))
  #define uv_cmpge_f64(a, b)     ((a) >= (b))
  #define uv_cmpnle_f64(a, b)    (!((a) <= (b)))
  #define uv_cmpgt_f64(a, b)     ((a) > (b))

  #define uv_select_f64(m, t, f) ((m) ? (t) : (f))
#endif

#include <stdio.h>
/*
 * 64-bit Integer support
 */
#if defined(__AVX10__) || defined(__AVX512F__)
  #define uv_load_i64(ptr)       _mm512_loadu_si512((__m512i const*)(ptr))
  #define uv_loadu_i64(ptr)      _mm512_loadu_si512((__m512i const*)(ptr))
  #define uv_store_i64(ptr, v)   _mm512_storeu_si512((__m512i*)(ptr), (v))
  #define uv_storeu_i64(ptr, v)  _mm512_storeu_si512((__m512i*)(ptr), (v))
  #define uv_setzero_i64()       _mm512_setzero_si512()
  #define uv_dup_i64(v)          _mm512_set1_epi64(v)

  #define uv_add_i64(a, b)       _mm512_add_epi64(a, b)
  #define uv_sub_i64(a, b)       _mm512_sub_epi64(a, b)
  #if defined(__AVX512DQ__) || defined(__AVX10__) || defined(_MSC_VER)
    #define uv_mul_i64(a, b)       _mm512_mullo_epi64(a, b)
  #else
    static inline v_i64 uv_mul_i64(v_i64 a, v_i64 b) {
      __m512i low_low = _mm512_mul_epu32(a, b);
      __m512i a_high = _mm512_srli_epi64(a, 32);
      __m512i b_high = _mm512_srli_epi64(b, 32);
      __m512i high_low = _mm512_mul_epu32(a_high, b);
      __m512i low_high = _mm512_mul_epu32(a, b_high);
      __m512i cross_terms = _mm512_add_epi64(high_low, low_high);
      __m512i cross_shifted = _mm512_slli_epi64(cross_terms, 32);
      return _mm512_add_epi64(low_low, cross_shifted);
    }
  #endif
  #define uv_div_i64(a, b)       _mm512_div_epi64(a, b)
  #define uv_max_i64(a, b)       _mm512_max_epi64(a, b)
  #define uv_min_i64(a, b)       _mm512_min_epi64(a, b)
  #define uv_abs_i64(v)          _mm512_abs_epi64(v)
  static inline v_i64 uv_neg_i64(v_i64 v) {
    __m512i zero = _mm512_setzero_si512();
    return _mm512_sub_epi64(zero, v);
  }

  static inline v_i64 uv_not_i64(v_i64 v) {
    return _mm512_ternarylogic_epi64(v, v, v, 0x55);
  }
  #define uv_and_i64(a, b)       _mm512_and_si512(a, b)
  #define uv_andnot_i64(a, b)    _mm512_andnot_si512(a, b)
  #define uv_or_i64(a, b)        _mm512_or_si512(a, b)
  #define uv_xor_i64(a, b)       _mm512_xor_si512(a, b)

  #define uv_cmpeq_i64(a, b)     _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_EQ)
  #define uv_cmplt_i64(a, b)     _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_LT)
  #define uv_cmple_i64(a, b)     _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_LE)
  #define uv_cmpne_i64(a, b)     _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_NE)
  #define uv_cmpnlt_i64(a, b)    _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_NLT)
  #define uv_cmpge_i64(a, b)     _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_GE)
  #define uv_cmpnle_i64(a, b)    _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_NLE)
  #define uv_cmpgt_i64(a, b)     _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_GT)

  #define uv_select_i64(m, t, f) _mm512_mask_blend_epi64(m, f, t)

  #define uv_shl_i64(v, imm)     _mm512_slli_epi64(v, imm)
  #define uv_shr_i64(v, imm)     _mm512_srai_epi64(v, imm)

  static inline v_i64 uv_cvt_f64_i64(v_f64 v) {
    #if defined(__AVX512DQ__) || defined(__AVX10__) || defined(_MSC_VER)
      return _mm512_cvttpd_epi64(v);
    #else
      __m256d v_low  = _mm512_extractf64x4_pd(v, 0);
      __m256d v_high = _mm512_extractf64x4_pd(v, 1);
      __m128i ints_low  = _mm256_cvttpd_epi32(v_low);
      __m128i ints_high = _mm256_cvttpd_epi32(v_high);
      __m256i combined_ints = _mm256_set_m128i(ints_high, ints_low);
      return _mm512_cvtepi32_epi64(combined_ints);
    #endif
  }
  static inline v_f64 uv_cvt_i64_f64(v_i64 v) {
    #if defined(__AVX512DQ__) || defined(__AVX10__) || defined(_MSC_VER)
      return _mm512_cvtepi64_pd(v);
    #else
    __m256i low_ints = _mm512_cvtepi64_epi32(v);
    __m512i shifted_high = _mm512_srli_epi64(v, 32);
    __m256i high_ints = _mm512_cvtepi64_epi32(shifted_high);
    __m512d high_double = _mm512_cvtepi32_pd(high_ints);
    __m512d low_double = _mm512_cvtepi32_pd(low_ints);
    __m512i low_ints_512 = _mm512_cvtepi32_epi64(low_ints);
    __mmask16 sign_mask = _mm512_cmp_epi64_mask(low_ints_512, _mm512_setzero_si512(), _MM_CMPINT_LT);
    __m512d correction = _mm512_set1_pd(4294967296.0);
    low_double = _mm512_mask_add_pd(low_double, sign_mask, low_double, correction);
    __m512d scale = _mm512_set1_pd(4294967296.0);
    return _mm512_fmadd_pd(high_double, scale, low_double);
    #endif
  }


#elif defined(__FMA__) || defined(__AVX2__)
  #define uv_load_i64(ptr)       _mm256_load_si256((__m256i const*)(ptr))
  #define uv_loadu_i64(ptr)      _mm256_loadu_si256((__m256i const*)(ptr))
  #define uv_store_i64(ptr, v)   _mm256_store_si256((__m256i*)(ptr), (v))
  #define uv_storeu_i64(ptr, v)  _mm256_storeu_si256((__m256i*)(ptr), (v))
  #define uv_setzero_i64()       _mm256_setzero_si256()
  #define uv_dup_i64(v)          _mm256_set1_epi64x(v)

  static inline v_i64 uv_not_i64(v_i64 v) {
    return _mm256_xor_si256(v, _mm256_set1_epi64x(-1));
  }
  #define uv_and_i64(a, b)       _mm256_and_si256(a, b)
  #define uv_andnot_i64(a, b)    _mm256_andnot_si256(a, b)
  #define uv_or_i64(a, b)        _mm256_or_si256(a, b)
  #define uv_xor_i64(a, b)       _mm256_xor_si256(a, b)

  #define uv_cmpeq_i64(a, b)     _mm256_cmpeq_epi64(a, b)
  #define uv_cmpgt_i64(a, b)     _mm256_cmpgt_epi64(a, b)
  #define uv_cmplt_i64(a, b)     _mm256_cmpgt_epi64(b, a)
  static inline v_i64 uv_cmple_i64(v_i64 a, v_i64 b) {
      return uv_not_i64(_mm256_cmpgt_epi64(a, b));
  }
  static inline v_i64 uv_cmpne_i64(v_i64 a, v_i64 b) {
      return uv_not_i64(_mm256_cmpeq_epi64(a, b));
  }
  static inline v_i64 uv_cmpge_i64(v_i64 a, v_i64 b) {
      return uv_not_i64(_mm256_cmpgt_epi64(b, a));
  }
  static inline v_i64 uv_cmpnlt_i64(v_i64 a, v_i64 b) {
      return uv_cmpge_i64(a, b);
  }
  static inline v_i64 uv_cmpnle_i64(v_i64 a, v_i64 b) {
      return _mm256_cmpgt_epi64(a, b);
  }

  #if defined(__AVX10__)
    #define uv_select_i64(m, t, f) _mm256_mask_blend_epi64(m, f, t)
  #else
    #define uv_select_i64(m, t, f) _mm256_blendv_epi8(f, t ,m)
  #endif

  #define uv_shl_i64(v, imm)     _mm256_slli_epi64(v, imm)
  static inline v_i64 uv_shr_i64(v_i64 v, unsigned int imm) {
    if (imm == 0) return v;
    if (imm >= 63) return _mm256_srai_epi32(v, 31);
    __m256i logical_shifted = _mm256_srli_epi64(v, imm);
    __m256i sign_bits = _mm256_srai_epi32(v, imm);
    __m256i upper_32_mask = _mm256_set1_epi64x(0xFFFFFFFF00000000ULL);
    return _mm256_blendv_epi8(logical_shifted, sign_bits, upper_32_mask);
  }

  #define uv_add_i64(a, b)       _mm256_add_epi64(a, b)
  #define uv_sub_i64(a, b)       _mm256_sub_epi64(a, b)
  static inline v_i64 uv_mul_i64(v_i64 a, v_i64 b) {
    __m256i low_low = _mm256_mul_epu32(a, b);
    __m256i a_high = _mm256_srli_epi64(a, 32);
    __m256i b_high = _mm256_srli_epi64(b, 32);
    __m256i high_low = _mm256_mul_epu32(a_high, b);
    __m256i low_high = _mm256_mul_epu32(a, b_high);
    __m256i cross_terms = _mm256_add_epi64(high_low, low_high);
    __m256i cross_terms_shifted = _mm256_slli_epi64(cross_terms, 32);
    return _mm256_add_epi64(low_low, cross_terms_shifted);
  }
  static inline v_i64 uv_max_i64(v_i64 a, v_i64 b) {
    return  _mm256_or_si256(_mm256_and_si256(uv_cmpgt_i64(a, b), a),
                            _mm256_andnot_si256(uv_cmpgt_i64(a, b), b));
  }
  static inline v_i64 uv_min_i64(v_i64 a, v_i64 b) {
    return _mm256_or_si256(_mm256_and_si256(uv_cmplt_i64(a, b), a),
                           _mm256_andnot_si256(uv_cmplt_i64(a, b), b));
  }
  static inline v_i64 uv_abs_i64(v_i64 v) {
    __m256i mask = uv_shl_i64(v, 31);
    __m256i xored = _mm256_xor_si256(v, mask);
    return _mm256_sub_epi64(xored, mask);
  }
  static inline v_i64 uv_neg_i64(v_i64 v) {
    __m256i zero = _mm256_setzero_si256();
    return _mm256_sub_epi64(zero, v);
  }

  static inline v_i64 uv_cvt_f64_i64(v_f64 v) {
    __m128i v32 = _mm256_cvttpd_epi32(v);
    __m256i v64_low = _mm256_cvtepi32_epi64(v32);
    __m128i v32_high = _mm_srli_si128(v32, 8);
    __m256i v64_high = _mm256_cvtepi32_epi64(v32_high);
    return _mm256_permute2f128_si256(v64_low, v64_high, 0x20);
  }
  static inline v_f64 uv_cvt_i64_f64(v_i64 v){
    __m256i magic_i_lo = _mm256_set1_epi64x(0x4330000000000000);
    __m256i magic_i_hi32 = _mm256_set1_epi64x(0x4530000080000000);
    __m256i magic_i_all = _mm256_set1_epi64x(0x4530000080100000);
    __m256d magic_d_all = _mm256_castsi256_pd(magic_i_all);
    __m256i v_lo = _mm256_blend_epi32(magic_i_lo, v, 0b01010101);
    __m256i v_hi = _mm256_srli_epi64(v, 32);
            v_hi = _mm256_xor_si256(v_hi, magic_i_hi32);
    __m256d v_hi_dbl = _mm256_sub_pd(_mm256_castsi256_pd(v_hi), magic_d_all);
    return  _mm256_add_pd(v_hi_dbl, _mm256_castsi256_pd(v_lo));
  }


#elif defined(__SSE2__)
  #define uv_load_i64(ptr)       _mm_load_si128((__m128i const*)(ptr))
  #define uv_loadu_i64(ptr)      _mm_loadu_si128((__m128i const*)(ptr))
  #define uv_store_i64(ptr, v)   _mm_store_si128((__m128i*)(ptr), (v))
  #define uv_storeu_i64(ptr, v)  _mm_storeu_si128((__m128i*)(ptr), (v))
  #define uv_setzero_i64()       _mm_setzero_si128()
  #define uv_dup_i64(v)          _mm_set1_epi64x(v)

  // uv_cmpgt_i64 is needed by uv_max_i64 and uv_min_i64
  #if defined(__SSE4_1__)
    #define uv_cmpeq_i64(a, b) _mm_cmpeq_epi64(a, b)
  #else
    static inline v_i64 uv_cmpeq_i64(v_i64 a, v_i64 b) {
      __m128i eq32 = _mm_cmpeq_epi32(a, b);
      __m128i eq32_shift = _mm_shuffle_epi32(eq32, _MM_SHUFFLE(3,3,1,1));
      return _mm_and_si128(eq32, eq32_shift);
    }
  #endif
  #if defined(__SSE4_2__)
    #define uv_cmpgt_i64(a, b)   _mm_cmpgt_epi64(a, b)
  #else
    static inline __m128i uv_cmpgt_i64(__m128i a, __m128i b) {
      const __m128i sign = _mm_set1_epi32(0x80000000);
      __m128i a_hi = _mm_shuffle_epi32(a, _MM_SHUFFLE(3,3,1,1));
      __m128i b_hi = _mm_shuffle_epi32(b, _MM_SHUFFLE(3,3,1,1));
      __m128i hi_gt = _mm_cmpgt_epi32(a_hi, b_hi);
      __m128i hi_eq = _mm_cmpeq_epi32(a_hi, b_hi);
      __m128i a_lo = _mm_xor_si128(a, sign);
      __m128i b_lo = _mm_xor_si128(b, sign);
      __m128i lo_gt = _mm_cmpgt_epi32(a_lo, b_lo);
      return _mm_or_si128(hi_gt, _mm_and_si128(hi_eq, lo_gt));
    }
  #endif
  static inline v_i64 uv_cmplt_i64(v_i64 a, v_i64 b) {
      return uv_cmpgt_i64(b, a);
  }
  static inline v_i64 uv_cmpne_i64(v_i64 a, v_i64 b) {
      v_i64 eq = uv_cmpeq_i64(a, b);
      return _mm_xor_si128(eq, _mm_set1_epi64x(-1));
  }
  static inline v_i64 uv_cmple_i64(v_i64 a, v_i64 b) {
      v_i64 gt = uv_cmpgt_i64(a, b);
      return _mm_xor_si128(gt, _mm_set1_epi64x(-1));
  }
  static inline v_i64 uv_cmpge_i64(v_i64 a, v_i64 b) {
      v_i64 gt = uv_cmpgt_i64(b, a);
      return _mm_xor_si128(gt, _mm_set1_epi64x(-1));
  }
  static inline v_i64 uv_cmpnlt_i64(v_i64 a, v_i64 b) {
      return uv_cmpge_i64(a, b);
  }
  static inline v_i64 uv_cmpnle_i64(v_i64 a, v_i64 b) {
      return uv_cmpgt_i64(a, b);
  }

  #if defined(__SSE4_1__)
    #define uv_select_i64(m, t, f) _mm_blendv_epi8(f, t, m)
  #else
    #define uv_select_i64(m, t, f) _mm_or_si128(_mm_andnot_si128(m, f), _mm_and_si128(m, t))
  #endif

  #define uv_add_i64(a, b)       _mm_add_epi64(a, b)
  #define uv_sub_i64(a, b)       _mm_sub_epi64(a, b)
  static inline v_i64 uv_mul_i64(v_i64 a, v_i64 b) {
    __m128i res_lo = _mm_mul_epu32(a, b);
    __m128i a_hi = _mm_srli_epi64(a, 32);
    __m128i b_hi = _mm_srli_epi64(b, 32);
    __m128i cross1 = _mm_mul_epu32(a_hi, b);
    __m128i cross2 = _mm_mul_epu32(a, b_hi);
    __m128i cross_sum = _mm_add_epi64(cross1, cross2);
    __m128i cross_sum_shifted = _mm_slli_epi64(cross_sum, 32);
    return _mm_add_epi64(res_lo, cross_sum_shifted);
  }
  static inline v_i64 uv_max_i64(v_i64 a, v_i64 b) {
    return  _mm_or_si128(_mm_and_si128(uv_cmpgt_i64(a, b), a),
                         _mm_andnot_si128(uv_cmpgt_i64(a, b), b));
  }
  static inline v_i64 uv_min_i64(v_i64 a, v_i64 b) {
    return _mm_or_si128(_mm_and_si128(uv_cmplt_i64(a, b), a),
                        _mm_andnot_si128(uv_cmplt_i64(a, b), b));
  }
  static inline v_i64 uv_abs_i64(v_i64 v) {
    __m128i mask = _mm_srai_epi32(v, 31);
    __m128i xored = _mm_xor_si128(v, mask);
    return _mm_sub_epi64(xored, mask);
  }
  static inline v_i64 uv_neg_i64(v_i64 v) {
    __m128i zero = _mm_setzero_si128();
    return _mm_sub_epi64(zero, v);
  }

  static inline v_i64 uv_not_i64(v_i64 v) {
    return _mm_xor_si128(v, _mm_set1_epi64x(-1));
  }
  #define uv_and_i64(a, b)       _mm_and_si128(a, b)
  #define uv_andnot_i64(a, b)    _mm_andnot_si128(a, b)
  #define uv_or_i64(a, b)        _mm_or_si128(a, b)
  #define uv_xor_i64(a, b)       _mm_xor_si128(a, b)

  #define uv_shl_i64(v, imm)     _mm_slli_epi64(v, imm)
  static inline v_i64 uv_shr_i64(v_i64 v, unsigned int imm) {
    if (imm >= 64) imm = 64;
    __m128i logical_shift = _mm_srli_epi64(v, imm);
    __m128i sign_high = _mm_shuffle_epi32(v, _MM_SHUFFLE(3, 3, 1, 1));
    __m128i sign_mask = _mm_cmplt_epi32(sign_high, _mm_setzero_si128());
    sign_mask = _mm_shuffle_epi32(sign_mask, _MM_SHUFFLE(3, 2, 1, 0));
    __m128i fill_mask = _mm_srli_epi64(sign_mask, 64 - imm);
    // 4. Combine the logical shift with the fill mask
    return _mm_or_si128(logical_shift, fill_mask);
  }

  static inline v_i64 uv_cvt_f64_i64(v_f64 v) {
    double d0 = _mm_cvtsd_f64(v);
    double d1 = _mm_cvtsd_f64(_mm_unpackhi_pd(v, v));
    const int64_t i[2] = { (int64_t)d0, (int64_t)d1 };
    return _mm_load_si128((__m128i const*)i);
  }
  static inline v_f64 uv_cvt_i64_f64(v_i64 v) {
    int64_t i0 = _mm_extract_epi64(v, 0);
    int64_t i1 = _mm_extract_epi64(v, 1);
    const double d[2] = { (double)i0, (double)i1 };
    return _mm_load_pd(d);
  }

#else
  #define uv_load_i64(ptr)       (*(const int64_t*)(ptr))
  #define uv_loadu_i64(ptr)      (*(const int64_t*)(ptr))
  #define uv_store_i64(ptr, v)   (*(int64_t*)(ptr) = (v))
  #define uv_storeu_i64(ptr, v)  (*(int64_t*)(ptr) = (v))
  #define uv_setzero_i64()       (0ULL)
  #define uv_dup_i64(v)          (v)

  #define uv_add_i64(a, b)       ((a) + (b))
  #define uv_sub_i64(a, b)       ((a) - (b))
  #define uv_mul_i64(a, b)       ((a) * (b))
  #define uv_max_i64(a, b)       ((a) > (b) ? (a) : (b))
  #define uv_min_i64(a, b)       ((a) < (b) ? (a) : (b))
  #define uv_abs_i64(v)          ((v) < 0 ? -(v) : (v))
  #define uv_neg_i64(v)          (-(v))

  #define uv_not_i64(v)          (~(v))
  #define uv_and_i64(a, b)       ((a) & (b))
  #define uv_andnot_i64(a, b)    (~(a) & (b))
  #define uv_or_i64(a, b)        ((a) | (b))
  #define uv_xor_i64(a, b)       ((a) ^ (b))

  #define uv_cmpeq_i64(a, b)     ((a) == (b))
  #define uv_cmplt_i64(a, b)     ((a) < (b))
  #define uv_cmple_i64(a, b)     ((a) <= (b))
  #define uv_cmpne_i64(a, b)     ((a) != (b))
  #define uv_cmpnlt_i64(a, b)    (!((a) < (b)))
  #define uv_cmpge_i64(a, b)     ((a) >= (b))
  #define uv_cmpnle_i64(a, b)    (!((a) <= (b)))
  #define uv_cmpgt_i64(a, b)     ((a) > (b))

  #define uv_select_i64(m, t, f) ((m) ? (t) : (f))

  #define uv_shl_i64(v, imm)     ((v) << (imm))
  #define uv_shr_i64(v, imm)     ((v) >> (imm))

  #define uv_cvt_f64_i64(v)      (int64_t)(v)
  #define uv_cvt_i64_f64(v)      (double)(v)
#endif


/*
 * 32-bit Integer support
 */
#if defined(__AVX10__) || defined(__AVX512F__)
  #define uv_load_i32(ptr)       _mm512_loadu_si512((__m512i const*)(ptr))
  #define uv_loadu_i32(ptr)      _mm512_loadu_si512((__m512i const*)(ptr))
  #define uv_store_i32(ptr, v)   _mm512_storeu_si512((__m512i*)(ptr), (v))
  #define uv_storeu_i32(ptr, v)  _mm512_storeu_si512((__m512i*)(ptr), (v))
  #define uv_setzero_i32()       _mm512_setzero_si512()
  #define uv_dup_i32(v)          _mm512_set1_epi32(v)

  #define uv_add_i32(a, b)       _mm512_add_epi32(a, b)
  #define uv_sub_i32(a, b)       _mm512_sub_epi32(a, b)
  #define uv_mul_i32(a, b)       _mm512_mullo_epi32(a, b)
  #define uv_max_i32(a, b)       _mm512_max_epi32(a, b)
  #define uv_min_i32(a, b)       _mm512_min_epi32(a, b)
  #define uv_abs_i32(v)          _mm512_abs_epi32(v)
  static inline v_i32 uv_neg_i32(v_i32 v) {
    __m512i zero = _mm512_setzero_si512();
    return _mm512_sub_epi32(zero, v);
  }

  static inline v_i32 uv_not_i32(v_i32 v) {
    return _mm512_xor_si512(v, _mm512_set1_epi32(-1));
  }
  #define uv_and_i32(a, b)       _mm512_and_si512(a, b)
  #define uv_andnot_i32(a, b)    _mm512_andnot_si512(a, b)
  #define uv_or_i32(a, b)        _mm512_or_si512(a, b)
  #define uv_xor_i32(a, b)       _mm512_xor_si512(a, b)

  #define uv_cmpeq_i32(a, b)     _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_EQ)
  #define uv_cmplt_i32(a, b)     _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_LT)
  #define uv_cmple_i32(a, b)     _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_LE)
  #define uv_cmpne_i32(a, b)     _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_NE)
  #define uv_cmpnlt_i32(a, b)    _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_NLT)
  #define uv_cmpge_i32(a, b)     _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_GE)
  #define uv_cmpnle_i32(a, b)    _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_NLE)
  #define uv_cmpgt_i32(a, b)     _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_GT)

  #define uv_select_i32(m, t, f) _mm512_mask_blend_epi32(m, f, t)

  #define uv_shl_i32(v, imm)     _mm512_slli_epi32(v, imm)
  #define uv_shr_i32(v, imm)     _mm512_srai_epi32(v, imm)

  static inline v_i32 uv_cvt_i16_i32(v_i32 v) {
    __m256i low_half = _mm512_castsi512_si256(v);
    return _mm512_cvtepi16_epi32(low_half);
  }

  static inline v_i32 uv_cvt_i32_i16(v_i32 v) {
    #if defined(__AVX512VL__) || defined(__AVX10__) || defined(_MSC_VER)
        __m256i packed = _mm512_cvtepi32_epi16(v);
        return _mm512_castsi256_si512(packed);
    #else
        __m512i low_bits = _mm512_and_si512(v, _mm512_set1_epi32(0xFFFF));
        __m512i shifted = _mm512_srli_epi32(low_bits, 16);

        alignas(32) float tmp[16];
        _mm512_store_ps(tmp, (_mm512_castsi512_ps(v)));
        int32_t *itmp = (int32_t*)tmp;
        int16_t *otmp = (int16_t*)tmp;
        for (int i = 0; i < 16; ++i) {
            otmp[i] = (int16_t)itmp[i];
        }
        return _mm512_castps_si512(_mm512_load_ps(tmp));
    #endif
  }

  static inline v_i32 uv_cvt_i8_i32(v_i32 v) {
    __m128i low_quarter = _mm512_castsi512_si128(v);
    return _mm512_cvtepi8_epi32(low_quarter);
  }

  static inline v_i32 uv_cvt_i32_i8(v_i32 v) {
    #if defined(__AVX512VL__) || defined(__AVX10__) || defined(_MSC_VER)
        __m128i packed = _mm512_cvtepi32_epi8(v);
        return _mm512_castsi128_si512(packed);
    #else
        alignas(32) float tmp[16];
        _mm512_store_ps(tmp, _mm512_castsi512_ps(v));

        int32_t *src_ptr = (int32_t*)tmp;
        int8_t  *dst_ptr = (int8_t*)tmp;

        for (int i = 0; i < 16; ++i) {
            dst_ptr[i] = (int8_t)src_ptr[i];
        }

        for (int i = 16; i < 64; ++i) {
            dst_ptr[i] = 0;
        }

        return _mm512_castps_si512(_mm512_load_ps(tmp));
    #endif
  }

#elif defined(__FMA__) || defined(__AVX2__)
  #define uv_load_i32(ptr)       _mm256_load_si256((__m256i const*)(ptr))
  #define uv_loadu_i32(ptr)      _mm256_loadu_si256((__m256i const*)(ptr))
  #define uv_store_i32(ptr, v)   _mm256_store_si256((__m256i*)(ptr), (v))
  #define uv_storeu_i32(ptr, v)  _mm256_storeu_si256((__m256i*)(ptr), (v))
  #define uv_setzero_i32()       _mm256_setzero_si256()
  #define uv_dup_i32(v)          _mm256_set1_epi32(v)

  #define uv_add_i32(a, b)       _mm256_add_epi32(a, b)
  #define uv_sub_i32(a, b)       _mm256_sub_epi32(a, b)
  #define uv_mul_i32(a, b)       _mm256_mullo_epi32(a, b)
  #define uv_max_i32(a, b)       _mm256_max_epi32(a, b)
  #define uv_min_i32(a, b)       _mm256_min_epi32(a, b)
  #define uv_abs_i32(v)          _mm256_abs_epi32(v)
  static inline v_i32 uv_neg_i32(v_i32 v) {
   __m256i zero = _mm256_setzero_si256();
    return _mm256_sub_epi32(zero, v);
  }

  static inline v_i32 uv_not_i32(v_i32 v) {
    return _mm256_xor_si256(v, _mm256_set1_epi32(-1));
  }
  #define uv_and_i32(a, b)       _mm256_and_si256(a, b)
  #define uv_andnot_i32(a, b)    _mm256_andnot_si256(a, b)
  #define uv_or_i32(a, b)        _mm256_or_si256(a, b)
  #define uv_xor_i32(a, b)       _mm256_xor_si256(a, b)

  static inline v_i32 uv_cmpeq_i32(v_i32 a, v_i32 b) {
      return _mm256_cmpeq_epi32(a, b);
  }
  static inline v_i32 uv_cmpgt_i32(v_i32 a, v_i32 b) {
      return _mm256_cmpgt_epi32(a, b);
  }
  static inline v_i32 uv_cmplt_i32(v_i32 a, v_i32 b) {
      return _mm256_cmpgt_epi32(b, a);
  }
  static inline v_i32 uv_cmpne_i32(v_i32 a, v_i32 b) {
      v_i32 eq = _mm256_cmpeq_epi32(a, b);
      return _mm256_xor_si256(eq, _mm256_set1_epi32(-1.0));
  }
  static inline v_i32 uv_cmple_i32(v_i32 a, v_i32 b) {
      v_i32 gt = _mm256_cmpgt_epi32(a, b);
      return _mm256_xor_si256(gt, _mm256_set1_epi32(-1.0));
  }
  static inline v_i32 uv_cmpge_i32(v_i32 a, v_i32 b) {
      v_i32 gt = _mm256_cmpgt_epi32(b, a);
      return _mm256_xor_si256(gt, _mm256_set1_epi32(-1.0));
  }
  static inline v_i32 uv_cmpnlt_i32(v_i32 a, v_i32 b) {
      return uv_cmpge_i32(a, b);
  }
  static inline v_i32 uv_cmpnle_i32(v_i32 a, v_i32 b) {
      return _mm256_cmpgt_epi32(a, b);
  }

  #if defined(__AVX10__)
    #define uv_select_i32(m, t, f) _mm256_mask_blend_epi32(m, f, t)
  #else
    #define uv_select_i32(m, t, f) _mm256_blendv_epi8(f, t ,m)
  #endif

  #define uv_shl_i32(v, imm)     _mm256_slli_epi32(v, imm)
  #define uv_shr_i32(v, imm)     _mm256_srai_epi32(v, imm)

  static inline v_i32 uv_cvt_i16_i32(v_i32 v) {
    return _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v));
  }

  static inline v_i32 uv_cvt_i32_i16(v_i32 v) {
    __m256i packed = _mm256_packs_epi32(v, _mm256_setzero_si256());
    return _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));
  }

  static inline v_i32 uv_cvt_i8_i32(v_i32 v) {
    return _mm256_cvtepi8_epi32(_mm256_castsi256_si128(v));
  }

  static inline v_i32 uv_cvt_i32_i8(v_i32 v) {
    __m256i packed16 = _mm256_packs_epi32(v, _mm256_setzero_si256());
    packed16 = _mm256_permute4x64_epi64(packed16, _MM_SHUFFLE(3, 1, 2, 0));

    __m128i low16  = _mm256_castsi256_si128(packed16);
    __m128i packed8 = _mm_packs_epi16(low16, _mm_setzero_si128());
    return _mm256_castsi128_si256(packed8);
  }

#elif defined(__SSE2__)
  #define uv_load_i32(ptr)       _mm_load_si128((__m128i const*)(ptr))
  #define uv_loadu_i32(ptr)      _mm_loadu_si128((__m128i const*)(ptr))
  #define uv_store_i32(ptr, v)   _mm_store_si128((__m128i*)(ptr), (v))
  #define uv_storeu_i32(ptr, v)  _mm_storeu_si128((__m128i*)(ptr), (v))
  #define uv_setzero_i32()       _mm_setzero_si128()
  #define uv_dup_i32(v)          _mm_set1_epi32(v)

  #define uv_add_i32(a, b)       _mm_add_epi32(a, b)
  #define uv_sub_i32(a, b)       _mm_sub_epi32(a, b)
  #if defined(__SSE4_1__)
    #define uv_mul_i32(a, b)     _mm_mullo_epi32(a, b)
    #define uv_max_i32(a, b)     _mm_max_epi32(a, b)
    #define uv_min_i32(a, b)     _mm_min_epi32(a, b)
  #else
    static inline v_i32 uv_mul_i32(v_i32 a, v_i32 b) {
      __m128i mul02 = _mm_mul_epu32(a, b);
      __m128i a13   = _mm_shuffle_epi32(a, _MM_SHUFFLE(3, 3, 1, 1));
      __m128i b13   = _mm_shuffle_epi32(b, _MM_SHUFFLE(3, 3, 1, 1));
      __m128i mul13 = _mm_mul_epu32(a13, b13);
      __m128i res_lo = _mm_unpacklo_epi32(mul02, mul13);
      __m128i res_hi = _mm_unpackhi_epi32(mul02, mul13);
      return _mm_unpacklo_epi64(res_lo, res_hi);
    }
    static inline v_i32 uv_max_i32(v_i32 a, v_i32 b) {
      return  _mm_or_si128(_mm_and_si128(_mm_cmpgt_epi32(a, b), a),
                           _mm_andnot_si128(_mm_cmpgt_epi32(a, b), b));
    }
    static inline v_i32 uv_min_i32(v_i32 a, v_i32 b) {
      return _mm_or_si128(_mm_and_si128(_mm_cmplt_epi32(a, b), a),
                          _mm_andnot_si128(_mm_cmplt_epi32(a, b), b));
    }
  #endif
  #if defined(__SSSE3__)
    #define uv_abs_i32(v)        _mm_abs_epi32(v)
  #else
    static inline v_i32 uv_abs_i32(v_i32 v) {
      __m128i sign = _mm_srai_epi32(v, 31);
      return _mm_sub_epi32(_mm_xor_si128(v, sign), sign);
    }
  #endif
  static inline v_i32 uv_neg_i32(v_i32 v) {
   __m128i zero = _mm_setzero_si128();
    return _mm_sub_epi32(zero, v);
  }

  static inline v_i32 uv_not_i32(v_i32 v) {
    return _mm_xor_si128(v, _mm_set1_epi32(-1));
  }
  #define uv_and_i32(a, b)       _mm_and_si128(a, b)
  #define uv_andnot_i32(a, b)    _mm_andnot_si128(a, b)
  #define uv_or_i32(a, b)        _mm_or_si128(a, b)
  #define uv_xor_i32(a, b)       _mm_xor_si128(a, b)

  static inline v_i32 uv_cmpeq_i32(v_i32 a, v_i32 b) {
      return _mm_cmpeq_epi32(a, b);
  }
  static inline v_i32 uv_cmpgt_i32(v_i32 a, v_i32 b) {
      return _mm_cmpgt_epi32(a, b);
  }
  static inline v_i32 uv_cmplt_i32(v_i32 a, v_i32 b) {
      return _mm_cmpgt_epi32(b, a);
  }
  static inline v_i32 uv_cmpne_i32(v_i32 a, v_i32 b) {
      v_i32 eq = _mm_cmpeq_epi32(a, b);
      return _mm_xor_si128(eq, _mm_set1_epi32(-1));
  }
  static inline v_i32 uv_cmple_i32(v_i32 a, v_i32 b) {
      v_i32 gt = _mm_cmpgt_epi32(a, b);
      return _mm_xor_si128(gt, _mm_set1_epi32(-1));
  }
  static inline v_i32 uv_cmpge_i32(v_i32 a, v_i32 b) {
      v_i32 gt = _mm_cmpgt_epi32(b, a);
      return _mm_xor_si128(gt, _mm_set1_epi32(-1));
  }
  static inline v_i32 uv_cmpnlt_i32(v_i32 a, v_i32 b) {
      return uv_cmpge_i32(a, b);
  }
  static inline v_i32 uv_cmpnle_i32(v_i32 a, v_i32 b) {
      return _mm_cmpgt_epi32(a, b);
  }

  #if defined(__SSE4_1__)
    #define uv_select_i32(m, t, f) _mm_blendv_epi8(f, t, m)
  #else
    #define uv_select_i32(m, t, f) _mm_or_si128(_mm_andnot_si128(m, f), _mm_and_si128(m, t))
  #endif

  #define uv_shl_i32(v, imm)     _mm_slli_epi32(v, imm)
  #define uv_shr_i32(v, imm)     _mm_srai_epi32(v, imm)

  static inline v_i32 uv_cvt_i16_i32(v_i32 v) {
    __m128i low_sign = _mm_unpacklo_epi16(v, v);
    return _mm_srai_epi32(low_sign, 16);
  }

  static inline v_i32 uv_cvt_i32_i16(v_i32 v) {
    return _mm_packs_epi32(v, _mm_setzero_si128());
  }

  static inline v_i32 uv_cvt_i8_i32(v_i32 v) {
    __m128i low_bytes = _mm_unpacklo_epi8(v, v);
    __m128i low_words = _mm_unpacklo_epi16(low_bytes, low_bytes);
    return _mm_srai_epi32(low_words, 24);
  }

  static inline v_i32 uv_cvt_i32_i8(v_i32 v) {
    __m128i packed16 = _mm_packs_epi32(v, _mm_setzero_si128());
    return _mm_packs_epi16(packed16, _mm_setzero_si128());
  }

#else
  #define uv_load_i32(ptr)       (*(const int32_t*)(ptr))
  #define uv_loadu_i32(ptr)      (*(const int32_t*)(ptr))
  #define uv_store_i32(ptr, v)   (*(int32_t*)(ptr) = (v))
  #define uv_storeu_i32(ptr, v)  (*(int32_t*)(ptr) = (v))
  #define uv_setzero_i32()       (0)
  #define uv_dup_i32(v)          (v)

  #define uv_add_i32(a, b)       ((a) + (b))
  #define uv_sub_i32(a, b)       ((a) - (b))
  #define uv_mul_i32(a, b)       ((a) * (b))
  #define uv_max_i32(a, b)       ((a) > (b) ? (a) : (b))
  #define uv_min_i32(a, b)       ((a) < (b) ? (a) : (b))
  #define uv_abs_i32(v)          ((v) < 0 ? -(v) : (v))
  #define uv_neg_i32(v)          (-(v))

  #define uv_not_i32(v)          (~(v))
  #define uv_and_i32(a, b)       ((a) & (b))
  #define uv_andnot_i32(a, b)    (~(a) & (b))
  #define uv_or_i32(a, b)        ((a) | (b))
  #define uv_xor_i32(a, b)       ((a) ^ (b))

  #define uv_cmpeq_i32(a, b)     ((a) == (b))
  #define uv_cmplt_i32(a, b)     ((a) < (b))
  #define uv_cmple_i32(a, b)     ((a) <= (b))
  #define uv_cmpne_i32(a, b)     ((a) != (b))
  #define uv_cmpnlt_i32(a, b)    (!((a) < (b)))
  #define uv_cmpge_i32(a, b)     ((a) >= (b))
  #define uv_cmpnle_i32(a, b)    (!((a) <= (b)))
  #define uv_cmpgt_i32(a, b)     ((a) > (b))

  #define uv_select_i32(m, t, f) ((m) ? (t) : (f))

  #define uv_shl_i32(v, imm)     ((v) << (imm))
  #define uv_shr_i32(v, imm)     ((v) >> (imm))

  static inline int32_t uv_cvt_i16_i32(int32_t v) {
    return (int32_t)((int16_t)v);
  }

  static inline int32_t uv_cvt_i32_i16(int32_t v) {
    return (int32_t)((int16_t)v);
  }

  static inline int32_t uv_cvt_i8_i32(int32_t v) {
    return (int32_t)((int8_t)v);
  }

  static inline int32_t uv_cvt_i32_i8(int32_t v) {
    return (int32_t)((int8_t)v);
  }

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
    #define uv_permutexvar_i8(idx, v)  _mm512_permutexvar_epi8(idx, v)

    #define uv_select_i8(m, t, f)  _mm512_mask_blend_epi8(m, f, t)
    #define uv_compress_i8(m, v)   _mm512_maskz_compress_epi8(m, v)

  #else
    #define uv_load_i8(ptr)      _mm256_load_si256((const __m256i*)(ptr))
    #define uv_loadu_i8(ptr)     _mm256_loadu_si256((const __m256i*)(ptr))
    #define uv_store_i8(ptr, v)  _mm256_store_si256((__m256i*)(ptr), (v))
    #define uv_storeu_i8(ptr, v) _mm256_storeu_si256((__m256i*)(ptr), (v))

    #define uv_cmpeq_i8(a, b)    _mm256_cmpeq_epi8_mask(a, b)
    #define uv_cmplt_i8(a, b)    _mm256_cmplt_epi8_mask(a, b)

    #define uv_shuffle_i8(a, b)  _mm256_shuffle_epi8(a, b)
    #define uv_permutexvar_i8(idx, v)  _mm256_permutexvar_epi8(idx, v)

    #define uv_select_i8(m, t, f)  _mm256_mask_blend_epi8(m, f, t)
    #define uv_compress_i8(m, v)   _mm256_maskz_compress_epi8(m, v)
  #endif

#elif defined(__AVX10__) || defined(__AVX512F__)
  #define uv_load_i8(ptr)        _mm512_load_si512((__m512i const*)(ptr))
  #define uv_loadu_i8(ptr)       _mm512_loadu_si512((__m512i const*)(ptr))
  #define uv_store_i8(ptr, v)    _mm512_store_si512((__m512i*)(ptr), (v))
  #define uv_storeu_i8(ptr, v)   _mm512_storeu_si512((__m512i*)(ptr), (v))

  #if defined(__AVX512BW__) || defined(__AVX10__) || defined(_MSC_VER)
    #define uv_cmpeq_i8(a, b)    _mm512_cmpeq_epi8_mask(a, b)
    #define uv_cmplt_i8(a, b)    _mm512_cmplt_epi8_mask(a, b)

    #define uv_shuffle_i8(a, b)  _mm512_s nlooks like it can be made smaller by lettshuffle_epi8(a, b)
    #if defined(__AVX512VBMI__)
      #define uv_permutexvar_i8(idx, v)  _mm512_permutexvar_epi8(idx, v)
    #else
      static inline v_i8 uv_permutexvar_i8(v_i8 idx, v_i8 a) {
        __m512i a_lane0 = a;
        __m512i a_lane1 = _mm512_permutex_epi64(a, _MM_SHUFFLE(0, 3, 2, 1));
        __m512i a_lane2 = _mm512_permutex_epi64(a, _MM_SHUFFLE(1, 0, 3, 2));
        __m512i a_lane3 = _mm512_permutex_epi64(a, _MM_SHUFFLE(2, 1, 0, 3));
        __m512i mask_0 = _mm512_and_si512(idx, _mm512_set1_epi8(0x0F));
        __mmask64 k0   = _mm512_cmpge_epu8_mask(idx, _mm512_set1_epi8(16));
        __m512i idx_0  = _mm512_mask_set1_epi8(mask_0, k0, 0x80);
        __m512i idx_sub16 = _mm512_sub_epi8(idx, _mm512_set1_epi8(16));
        __m512i mask_1    = _mm512_and_si512(idx_sub16, _mm512_set1_epi8(0x0F));
        __mmask64 k1      = _mm512_cmpge_epu8_mask(idx_sub16, _mm512_set1_epi8(16));
        __m512i idx_1     = _mm512_mask_set1_epi8(mask_1, k1, 0x80);
        __m512i idx_sub32 = _mm512_sub_epi8(idx, _mm512_set1_epi8(32));
        __m512i mask_2    = _mm512_and_si512(idx_sub32, _mm512_set1_epi8(0x0F));
        __mmask64 k2      = _mm512_cmpge_epu8_mask(idx_sub32, _mm512_set1_epi8(16));
        __m512i idx_2     = _mm512_mask_set1_epi8(mask_2, k2, 0x80);
        __m512i idx_sub48 = _mm512_sub_epi8(idx, _mm512_set1_epi8(48));
        __m512i idx_3     = _mm512_and_si512(idx_sub48, _mm512_set1_epi8(0x0F));
        __m512i res0 = _mm512_shuffle_epi8(a_lane0, idx_0);
        __m512i res1 = _mm512_shuffle_epi8(a_lane1, idx_1);
        __m512i res2 = _mm512_shuffle_epi8(a_lane2, idx_2);
        __m512i res3 = _mm512_shuffle_epi8(a_lane3, idx_3);
        __m512i out_half1 = _mm512_or_si512(res0, res1);
        __m512i out_half2 = _mm512_or_si512(res2, res3);
        return _mm512_or_si512(out_half1, out_half2);
      }
    #endif

    #define uv_select_i8(m, t, f) _mm512_mask_blend_epi8(m, f, t)
    #define uv_compress_i8(m, v)  _mm512_maskz_compress_epi8(m, v)
  #else
    static inline v_i32 uv_shuffle_i8(v_i32 a, v_i32 b) {
     __m256i a_low  = _mm512_castsi512_si256(a);
      __m256i a_high = _mm512_extracti64x4_epi64(a, 1);
      __m256i b_low  = _mm512_castsi512_si256(b);
      __m256i b_high = _mm512_extracti64x4_epi64(b, 1);

      __m256i r_low  = _mm256_shuffle_epi8(a_low, b_low);
      __m256i r_high = _mm256_shuffle_epi8(a_high, b_high);

      return _mm512_inserti64x4(_mm512_castsi256_si512(r_low), r_high, 1);
    }

    #define uv_permutexvar_i8(idx, v) uv_shuffle_i8(v, idx)

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

    static inline __m512i uv_select_i8(__mmask64 m, __m512i t, __m512i f) {
      return _mm512_ternarylogic_epi32(_mm512_movm_epi32((__mmask16)m), t, f, 0xD8);
    }

    static inline __m512i uv_compress_i8(__mmask64 mask, __m512i v) {
      alignas(64) int8_t src[64];
      alignas(64) int8_t dst[64] = {0};
      _mm512_storeu_si512((void*)src, v);
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

  static inline v_i64 uv_permutexvar_i8(v_i64 idx, v_i64 v) {
    __m256i a_low  = _mm256_permute2x128_si256(v, v, 0x00);
    __m256i a_high = _mm256_permute2x128_si256(v, v, 0x11);
    __m256i lane_select_mask = _mm256_cmpgt_epi8(_mm256_setzero_si256(),
                               _mm256_slli_epi32(idx, 3));
    __m256i shuffled_low  = _mm256_shuffle_epi8(a_low, idx);
    __m256i shuffled_high = _mm256_shuffle_epi8(a_high, idx);
    return _mm256_blendv_epi8(shuffled_low, shuffled_high, lane_select_mask);
  }

  static inline v_i64 uv_compress_i8(v_i64 mask, v_i64 v) {
    alignas(32) int8_t src[32];
    alignas(32) int8_t m_bytes[32];
    alignas(32) int8_t dst[32] = {0};
    _mm256_storeu_si256((__m256i*)src, v);
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
  #define uv_cmplt_i8(a, b)      _mm_cmplt_epi8(a, b)

  #if defined(__SSE4_1__)
    #define uv_select_i8(m, t, f) _mm_blendv_epi8(f, t, m)
  #else
    static inline v_i8 uv_select_i8(v_mask_i8 m, v_i8 t, v_i8 f) {
      return _mm_or_si128(_mm_and_si128(m, t), _mm_andnot_si128(m, f));
    }
  #endif

  #if defined(__SSSE3__)
    #define uv_shuffle_i8(a, b)      _mm_shuffle_epi8(a, b)
  #else
    static inline __m128i uv_shuffle_i8(__m128i a, __m128i b) {
      union { __m128i vec; int8_t bytes[16]; } src, ctrl, dst;
      src.vec = a;
      ctrl.vec = b;
      for (int i = 0; i < 16; ++i) {
          dst.bytes[i] = (ctrl.bytes[i] & 0x80) ? 0 : src.bytes[ctrl.bytes[i] & 0x0F];
      }
      return dst.vec;
    }
  #endif
  static inline __m128i uv_permutexvar_i8(__m128i idx, __m128i v) {
    __m128i a_low  = _mm_unpacklo_epi64(v, v);
    __m128i a_high = _mm_unpackhi_epi64(v, v);
    __m128i shuffled_low  = uv_shuffle_i8(a_low, idx);
    __m128i shuffled_high = uv_shuffle_i8(a_high, idx);
    __m128i bit3_mask = _mm_set1_epi8(0x08);
    __m128i lane_select_mask = _mm_cmpeq_epi8(_mm_and_si128(idx, bit3_mask), bit3_mask);
    __m128i result = _mm_or_si128(
        _mm_andnot_si128(lane_select_mask, shuffled_low),
        _mm_and_si128(lane_select_mask, shuffled_high)
    );

    return result;
  }
  static inline __m128i uv_compress_i8(__m128i mask, __m128i v) {
    alignas(16) int8_t src[16];
    alignas(16) int8_t m_bytes[16];
    alignas(16) int8_t dst[16] = {0};
    _mm_storeu_si128((__m128i*)src, v);
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
  #define uv_cmple_i8(a, b)      ((a) <= (b))
  #define uv_cmpne_i8(a, b)      ((a) != (b))
  #define uv_cmpnlt_i8(a, b)     (!((a) < (b)))
  #define uv_cmpge_i8(a, b)      ((a) >= (b))
  #define uv_cmpnle_i8(a, b)     (!((a) > (b)))
  #define uv_cmpgt_i8(a, b)      ((a) > (b))

  #define uv_select_i8(mask, t, f)   ((mask) ? (t) : (f))

  #define uv_shuffle_i8(a, b)        (((b) & 0x80) ? 0 : (a))
  #define uv_permutexvar_i8(idx, v)  (((idx) & 0x80) ? 0 : (v))
  #define uv_compress_i8(mask, v)    ((mask) ? (v) : 0)
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
    static inline int uv_tzcnt_u64(unsigned long long v) {
      unsigned long index;
      return _BitScanForward64(&index, v) ? (int)index : 64;
    }
    static inline int uv_lzcnt_u64(unsigned long long v) {
      unsigned long index;
      return _BitScanReverse64(&index, v) ? (int)(63 - index) : 64;
    }
  #endif
#else
  // Software fallback for unknown compiler environments
  static inline int uv_popcnt_u64(uint64_t v) {
    v = v - ((v >> 1) & 0x5555555555555555ULL);
    v = (v & 0x3333333333333333ULL) + ((v >> 2) & 0x3333333333333333ULL);
    return (int)((((v + (v >> 4)) & 0xF0F0F0F0F0F0F0FULL) * 0x101010101010101ULL) >> 56);
  }
  static inline int uv_tzcnt_u64(uint64_t v) {
    if (v == 0) return 64;
    int n = 0;
    if ((v & 0xFFFFFFFFULL) == 0) { n += 32; v >>= 32; }
    if ((v & 0xFFFFULL) == 0)     { n += 16; v >>= 16; }
    if ((v & 0xFFULL) == 0)       { n += 8;  v >>= 8;  }
    if ((v & 0xFULL) == 0)        { n += 4;  v >>= 4;  }
    if ((v & 0x3ULL) == 0)        { n += 2;  v >>= 2;  }
    if ((v & 0x1ULL) == 0)        { n += 1; }
    return n;
  }
  static inline int uv_lzcnt_u64(uint64_t v) {
    if (v == 0) return 64;
    int n = 0;
    if ((v & 0xFFFFFFFF00000000ULL) == 0) { n += 32; v <<= 32; }
    if ((v & 0xFFFF000000000000ULL) == 0) { n += 16; v <<= 16; }
    if ((v & 0xFF00000000000000ULL) == 0) { n += 8;  v <<= 8;  }
    if ((v & 0xF000000000000000ULL) == 0) { n += 4;  v <<= 4;  }
    if ((v & 0xC000000000000000ULL) == 0) { n += 2;  v <<= 2;  }
    if ((v & 0x8000000000000000ULL) == 0) { n += 1; }
    return n;
  }
#endif

#if defined(__cplusplus)
}  /* extern "C" */
#endif

#endif // UVX_SIMD_H
