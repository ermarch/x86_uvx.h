#include "x86_uvx.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

#define TEST_epsilon 1e-5f

#define _MIN(a,b)    (((a)<(b)) ? (a) : (b))
#define _MAX(a,b)    (((a)>(b)) ? (a) : (b))

// Helper to print test results
void report(const char* test_name, bool success) {
    printf("[%s] %s\n", success ? "PASS" : "FAIL", test_name);
}

/*
 * Scalar bitwise helpers
 * These return the raw bit pattern as a uint32_t.
 */
// F32
static inline uint32_t scalar_and_f32_bits(float a, float b) {
    union { float f; uint32_t i; } va, vb, vr;
    va.f = a; vb.f = b; vr.i = va.i & vb.i;
    return vr.i;
}

static inline uint32_t scalar_andnot_f32_bits(float a, float b) {
    union { float f; uint32_t i; } va, vb, vr;
    va.f = a; vb.f = b; vr.i = ~va.i & vb.i;
    return vr.i;
}

static inline uint32_t scalar_or_f32_bits(float a, float b) {
    union { float f; uint32_t i; } va, vb, vr;
    va.f = a; vb.f = b; vr.i = va.i | vb.i;
    return vr.i;
}

static inline uint32_t scalar_xor_f32_bits(float a, float b) {
    union { float f; uint32_t i; } va, vb, vr;
    va.f = a; vb.f = b; vr.i = va.i ^ vb.i;
    return vr.i;
}

static inline v_f32 scalar_cmpgt_f32(float *a, float *b) {
   const v_f32 v_val_true = uv_dup_f32(1.0f);
   const v_f32 v_val_false = uv_dup_f32(0.0f);

   v_f32 v_a = uv_loadu_f32(a);
   v_f32 v_b = uv_loadu_f32(b);

   v_mask v_mask = uv_cmpgt_f32(v_a, v_b);
   return uv_select_f32(v_mask, v_val_true, v_val_false);
}

static inline v_f32 scalar_cmplt_f32(float *a, float *b) {
   const v_f32 v_val_true = uv_dup_f32(1.0f);
   const v_f32 v_val_false = uv_dup_f32(0.0f);

   v_f32 v_a = uv_loadu_f32(a);
   v_f32 v_b = uv_loadu_f32(b);

   v_mask v_mask = uv_cmplt_f32(v_a, v_b);
   return uv_select_f32(v_mask, v_val_true, v_val_false);
}

static inline float scalar_rint_f32(float x) {
    float magic = copysign(8388608.0f, x);
    return (int32_t)((x + magic) - magic);
}

// I32
static inline v_i32 scalar_cmpgt_i32(int32_t *a, int32_t *b) {
   const v_i32 v_val_true = uv_dup_i32(1);
   const v_i32 v_val_false = uv_dup_i32(0);

   v_i32 v_a = uv_loadu_i32(a);
   v_i32 v_b = uv_loadu_i32(b);

   v_mask_i8 v_mask = uv_cmpgt_i32(v_a, v_b);
   return uv_select_i32(v_mask, v_val_true, v_val_false);
}

static inline v_i32 scalar_cmplt_i32(int32_t *a, int32_t *b) {
   const v_i32 v_val_true = uv_dup_i32(1);
   const v_i32 v_val_false = uv_dup_i32(0);

   v_i32 v_a = uv_loadu_i32(a);
   v_i32 v_b = uv_loadu_i32(b);

   v_mask_i8 v_mask = uv_cmplt_i32(v_a, v_b);
   return uv_select_i32(v_mask, v_val_true, v_val_false);
}

// F64
static inline uint64_t scalar_and_f64_bits(double a, double b) {
    union { double f; uint64_t i; } va, vb, vr;
    va.f = a; vb.f = b; vr.i = va.i & vb.i;
    return vr.i;
}

static inline uint64_t scalar_andnot_f64_bits(double a, double b) {
    union { double f; uint64_t i; } va, vb, vr;
    va.f = a; vb.f = b; vr.i = ~va.i & vb.i;
    return vr.i;
}

static inline uint64_t scalar_or_f64_bits(double a, double b) {
    union { double f; uint64_t i; } va, vb, vr;
    va.f = a; vb.f = b; vr.i = va.i | vb.i;
    return vr.i;
}

static inline uint64_t scalar_xor_f64_bits(double a, double b) {
    union { double f; uint64_t i; } va, vb, vr;
    va.f = a; vb.f = b; vr.i = va.i ^ vb.i;
    return vr.i;
}

static inline v_f64 scalar_cmpgt_f64(double *a, double *b) {
   const v_f64 v_val_true = uv_dup_f64(1.0f);
   const v_f64 v_val_false = uv_dup_f64(0.0f);

   v_f64 v_a = uv_loadu_f64(a);
   v_f64 v_b = uv_loadu_f64(b);

   v_mask_f64 v_mask = uv_cmpgt_f64(v_a, v_b);
   return uv_select_f64(v_mask, v_val_true, v_val_false);
}

static inline v_f64 scalar_cmplt_f64(double *a, double *b) {
   const v_f64 v_val_true = uv_dup_f64(1.0f);
   const v_f64 v_val_false = uv_dup_f64(0.0f);

   v_f64 v_a = uv_loadu_f64(a);
   v_f64 v_b = uv_loadu_f64(b);

   v_mask_f64 v_mask = uv_cmplt_f64(v_a, v_b);
   return uv_select_f64(v_mask, v_val_true, v_val_false);
}


/**
 * Test Arithmetic
 */
void test_f32_arithmetic() {
    const int lanes = uv_lanes(32);
    float src1[lanes], src2[lanes], dst[lanes];

    for (int i = 0; i < lanes; i++) {
        src1[i] = (float)(i + 1) * 1.5f;
        src2[i] = (float)(i + 1) * 2.0f;
    }

    v_f32 v_src1 = uv_loadu_f32(src1);
    v_f32 v_src2 = uv_loadu_f32(src2);

    // Test Add
    v_f32 v_res = uv_add_f32(v_src1, v_src2);
    uv_storeu_f32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(fabsf(dst[i] - (src1[i] + src2[i])) < TEST_epsilon);
    }

    // Test Sub
    v_res = uv_sub_f32(v_src1, v_src2);
    uv_storeu_f32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(fabsf(dst[i] - (src1[i] - src2[i])) < TEST_epsilon);
    }

    // Test Mul
    v_res = uv_mul_f32(v_src1, v_src2);
    uv_storeu_f32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(fabsf(dst[i] - (src1[i] * src2[i])) < TEST_epsilon);
    }

    // Test Div
    for (int i = 0; i < lanes; i++) src2[i] = 2.0f;
    v_src2 = uv_loadu_f32(src2);
    v_res = uv_div_f32(v_src1, v_src2);
    uv_storeu_f32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(fabsf(dst[i] - (src1[i] / src2[i])) < TEST_epsilon);
    }

    // Test Abs
    v_res = uv_abs_f32(v_src1);
    uv_storeu_f32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == fabsf(src1[i]));
    }

    // Test Neg
    v_res = uv_neg_f32(v_src1);
    uv_storeu_f32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == -src1[i]);
    }

    report("f32 Arithmetic", true);
}

void test_f64_arithmetic() {
    const int lanes = uv_lanes(64);
    double src1[lanes], src2[lanes], dst[lanes];

    for (int i = 0; i < lanes; i++) {
        src1[i] = (double)(i + 1) * 1.5;
        src2[i] = (double)(i + 1) * 2.0;
    }

    v_f64 v_src1 = uv_loadu_f64(src1);
    v_f64 v_src2 = uv_loadu_f64(src2);

    // Test Add
    v_f64 v_res = uv_add_f64(v_src1, v_src2);
    uv_storeu_f64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(fabsf(dst[i] - (src1[i] + src2[i])) < TEST_epsilon);
    }

    // Test Sub
    v_res = uv_sub_f64(v_src1, v_src2);
    uv_storeu_f64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(fabsf(dst[i] - (src1[i] - src2[i])) < TEST_epsilon);
    }

    // Test Mul
    v_res = uv_mul_f64(v_src1, v_src2);
    uv_storeu_f64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(fabsf(dst[i] - (src1[i] * src2[i])) < TEST_epsilon);
    }

    // Test Div
    for (int i = 0; i < lanes; i++) src2[i] = 2.0f;
    v_src2 = uv_loadu_f64(src2);
    v_res = uv_div_f64(v_src1, v_src2);
    uv_storeu_f64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(fabsf(dst[i] - (src1[i] / src2[i])) < TEST_epsilon);
    }

    // Test Abs
    v_res = uv_abs_f64(v_src1);
    uv_storeu_f64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == fabsf(src1[i]));
    }

    // Test Neg
    v_res = uv_neg_f64(v_src1);
    uv_storeu_f64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == -src1[i]);
    }

    report("f64 Arithmetic", true);
}

void test_i32_arithmetic() {
    const int lanes = uv_lanes(32);
    int32_t src1[lanes], src2[lanes], dst[lanes];

    for (int i = 0; i < lanes; i++) {
        src1[i] = i * 10;
        src2[i] = i + 5;
    }

    v_i32 v_src1 = uv_loadu_i32(src1);
    v_i32 v_src2 = uv_loadu_i32(src2);

    // Test Add
    v_i32 v_res = uv_add_i32(v_src1, v_src2);
    uv_storeu_i32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == (src1[i] + src2[i]));
    }

    // Test Sub
    v_res = uv_sub_i32(v_src1, v_src2);
    uv_storeu_i32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(fabsf(dst[i] - (src1[i] - src2[i])) < TEST_epsilon);
    }

    // Test Mul (mullo)
    v_res = uv_mul_i32(v_src1, v_src2);
    uv_storeu_i32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == (int32_t)((int64_t)src1[i] * src2[i]));
    }

    // Test Abs
    v_res = uv_abs_i32(v_src1);
    uv_storeu_i32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == fabsf(src1[i]));
    }

    // Test Neg
    v_res = uv_neg_i32(v_src1);
    uv_storeu_i32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == -src1[i]);
    }

    // Test Shift Left
    v_res = uv_shl_i32(v_src1, 2);
    uv_storeu_i32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == (src1[i] << 2));
    }

    // Test Shift Right
    v_res = uv_shr_i32(v_src1, 2);
    uv_storeu_i32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == (src1[i] >> 2));
    }

    report("i32 Arithmetic", true);
}

void test_i64_arithmetic() {
    const int lanes = uv_lanes(64);
    int64_t src1[lanes], src2[lanes], dst[lanes];

    for (int i = 0; i < lanes; i++) {
        src1[i] = i * 10;
        src2[i] = i + 5;
    }

    v_i64 v_src1 = uv_loadu_i64(src1);
    v_i64 v_src2 = uv_loadu_i64(src2);

    // Test Add
    v_i64 v_res = uv_add_i64(v_src1, v_src2);
    uv_storeu_i64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == (src1[i] + src2[i]));
    }

    // Test Sub
    v_res = uv_sub_i64(v_src1, v_src2);
    uv_storeu_i64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(fabsf(dst[i] - (src1[i] - src2[i])) < TEST_epsilon);
    }

    // Test Mul (mullo)
    v_res = uv_mul_i64(v_src1, v_src2);
    uv_storeu_i64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == (int64_t)((int64_t)src1[i] * src2[i]));
    }

    // Test Abs
    v_res = uv_abs_i64(v_src1);
    uv_storeu_i64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == fabsf(src1[i]));
    }

    // Test Neg
    v_res = uv_neg_i64(v_src1);
    uv_storeu_i64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == -src1[i]);
    }

    // Test Shift Left
    v_res = uv_shl_i64(v_src1, 2);
    uv_storeu_i64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == (src1[i] << 2));
    }

    // Test Shift Right
    v_res = uv_shr_i64(v_src1, 2);
    uv_storeu_i64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == (src1[i] >> 2));
    }

    report("i64 Arithmetic", true);
}

/**
 * Test Multiply-Add (FMA)
 * Tests: (a * b) + c
 */
void test_f32_fmadd() {
    const int lanes = uv_lanes(32);
    float a[lanes], b[lanes], c[lanes], dst[lanes];

    // Initialize with a variety of values to stress the hardware
    // 1. Standard values
    // 2. Multiplication by zero
    // 3. Multiplication by one (identity)
    // 4. Large values to check for overflow/rounding
    for (int i = 0; i < lanes; i++) {
        if (i < 4) {
            a[i] = 1.5f;  b[i] = 2.0f;  c[i] = 0.5f; // (1.5 * 2.0) + 0.5 = 3.5
        } else if (i < 8) {
            a[i] = 10.0f; b[i] = 0.0f;  c[i] = 5.0f; // (10.0 * 0.0) + 5.0 = 5.0
        } else if (i < 12) {
            a[i] = 2.0f;  b[i] = 1.0f;  c[i] = 3.0f; // (2.0 * 1.0) + 3.0 = 5.0
        } else {
            a[i] = 100.0f; b[i] = 50.0f; c[i] = 25.0f; // (100 * 50) + 25 = 5025
        }
    }

    v_f32 v_a = uv_loadu_f32(a);
    v_f32 v_b = uv_loadu_f32(b);
    v_f32 v_c = uv_loadu_f32(c);

    // Test FMADD
    v_f32 v_res = uv_fmadd_f32(v_a, v_b, v_c);
    uv_storeu_f32(dst, v_res);

    // Use a slightly larger epsilon (1e-4) because
    // hardware FMA might differ slightly from two-step scalar rounding.
    for (int i = 0; i < lanes; i++) {
        float expected = (a[i] * b[i]) + c[i];

        // If we are on a target with hardware FMA (AVX2/AVX512),
        // the result is more precise than the scalar equivalent.
        // If we are on SSE2/Scalar, they will be nearly identical.
        assert(fabsf(dst[i] - expected) < 1e-4f);
    }

    // Test FMSUB
    v_res = uv_fmsub_f32(v_a, v_b, v_c);
    uv_storeu_f32(dst, v_res);

    for (int i = 0; i < lanes; i++) {
        float expected = (a[i] * b[i]) - c[i];
        assert(fabsf(dst[i] - expected) < 1e-4f);
    }

    report("f32 Multiply-Add", true);
}

void test_f64_fmadd() {
    const int lanes = uv_lanes(64);
    double a[lanes], b[lanes], c[lanes], dst[lanes];

    // Initialize with a variety of values to stress the hardware
    // 1. Standard values
    // 2. Multiplication by zero
    // 3. Multiplication by one (identity)
    // 4. Large values to check for overflow/rounding
    for (int i = 0; i < lanes; i++) {
        if (i < 4) {
            a[i] = 1.5;  b[i] = 2.0;  c[i] = 0.5; // (1.5 * 2.0) + 0.5 = 3.5
        } else if (i < 8) {
            a[i] = 10.0; b[i] = 0.0;  c[i] = 5.0; // (10.0 * 0.0) + 5.0 = 5.0
        } else if (i < 12) {
            a[i] = 2.0;  b[i] = 1.0;  c[i] = 3.0; // (2.0 * 1.0) + 3.0 = 5.0
        } else {
            a[i] = 100.0; b[i] = 50.0; c[i] = 25.0; // (100 * 50) + 25 = 5025
        }
    }

    v_f64 v_a = uv_loadu_f64(a);
    v_f64 v_b = uv_loadu_f64(b);
    v_f64 v_c = uv_loadu_f64(c);

    // Test FMADD
    v_f64 v_res = uv_fmadd_f64(v_a, v_b, v_c);
    uv_storeu_f64(dst, v_res);

    // Use a slightly larger epsilon (1e-4) because
    // hardware FMA might differ slightly from two-step scalar rounding.
    for (int i = 0; i < lanes; i++) {
        double expected = (a[i] * b[i]) + c[i];

        // If we are on a target with hardware FMA (AVX2/AVX512),
        // the result is more precise than the scalar equivalent.
        // If we are on SSE2/Scalar, they will be nearly identical.
        assert(fabsf(dst[i] - expected) < 1e-4f);
    }

    // Test FMSUB
    v_res = uv_fmsub_f64(v_a, v_b, v_c);
    uv_storeu_f64(dst, v_res);

    for (int i = 0; i < lanes; i++) {
        double expected = (a[i] * b[i]) - c[i];
        assert(fabsf(dst[i] - expected) < 1e-4f);
    }

    report("f64 Multiply-Add", true);
}

/**
 * Test Comparison
 */
void test_f32_comparison() {
    const int lanes = uv_lanes(32);
    float src1[lanes], src2[lanes], dst[lanes];

    for (int i = 0; i < lanes; i++) {
        src1[i] = (float)(i + 1) * 1.5f;
        src2[i] = (float)(i + 1) * 2.0f;
    }

    v_f32 v_src1 = uv_loadu_f32(src1);
    v_f32 v_src2 = uv_loadu_f32(src2);

    // Test Min
    v_f32 v_res = uv_min_f32(v_src1, v_src2);
    uv_storeu_f32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == _MIN(src1[i], src2[i]));
    }

    // Test Max
    v_res = uv_max_f32(v_src1, v_src2);
    uv_storeu_f32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == _MAX(src1[i], src2[i]));
    }

    // Test Greater-than and Select
    float a_src[UV_MAX_LANES] = {
            1.0f,     -1.0f,     0.0f,  5.0f,
            1.0f, -INFINITY, INFINITY,  1.0f,
             NAN,       NAN,    1e30f,  0.0f,
             NAN,       NAN,    1e30f,  0.0f
    };
    float b_src[UV_MAX_LANES] = {
            0.5f,     -2.0f,     1.0f,  5.0f,
        INFINITY,      0.0f, INFINITY,   NAN,
            1.0f,       NAN,    1e30f,  0.0f,
            1.0f,       NAN,    1e30f,  0.0f
    };
    for (int i = 0; i < UV_MAX_LANES; i += lanes) {
        union { v_f32 ps; float f[UV_MAX_LANES]; } res;
        res.ps = scalar_cmpgt_f32(a_src + i, b_src + i);
        for (int r = 0; r < lanes; r++) {
            assert((a_src[i + r] > b_src[i + r]) == res.f[r]);
        }
    }

    // Test Less-than
    for (int i = 0; i < UV_MAX_LANES; i += lanes) {
        union { v_f32 ps; float f[UV_MAX_LANES]; } res;
        res.ps = scalar_cmplt_f32(a_src + i, b_src + i);
        for (int r = 0; r < lanes; r++) {
            assert((a_src[i + r] < b_src[i + r]) == res.f[r]);
        }
    }

    report("f32 Comparison", true);
}

void test_f64_comparison() {
    const int lanes = uv_lanes(64);
    double src1[lanes], src2[lanes], dst[lanes];

    for (int i = 0; i < lanes; i++) {
        src1[i] = (double)(i + 1) * 1.5;
        src2[i] = (double)(i + 1) * 2.0;
    }

    v_f64 v_src1 = uv_loadu_f64(src1);
    v_f64 v_src2 = uv_loadu_f64(src2);

    // Test Min
    v_f64 v_res = uv_min_f64(v_src1, v_src2);
    uv_storeu_f64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == _MIN(src1[i], src2[i]));
    }

    // Test Max
    v_res = uv_max_f64(v_src1, v_src2);
    uv_storeu_f64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == _MAX(src1[i], src2[i]));
    }

    // Test Greater-than and Select
    double a_src[UV_MAX_LANES] = {
             1.0,      -1.0,      0.0,   5.0,
             1.0, -INFINITY, INFINITY,   1.0,
             NAN,       NAN,     1e30,   0.0,
             NAN,       NAN,     1e30,   0.0
    };
    double b_src[UV_MAX_LANES] = {
             0.5,      -2.0,      1.0,   5.0,
        INFINITY,        0.0, INFINITY,  NAN,
             1.0,       NAN,      1e30,  0.0,
             1.0,       NAN,      1e30,  0.0
    };
    for (int i = 0; i < UV_MAX_LANES; i += lanes) {
        union { v_f64 ps; double f[UV_MAX_LANES]; } res;
        res.ps = scalar_cmpgt_f64(a_src + i, b_src + i);
        for (int r = 0; r < lanes; r++) {
            assert((a_src[i + r] > b_src[i + r]) == res.f[r]);
        }
    }

    // Test Less-than
    for (int i = 0; i < UV_MAX_LANES; i += lanes) {
        union { v_f64 ps; double f[UV_MAX_LANES]; } res;
        res.ps = scalar_cmplt_f64(a_src + i, b_src + i);
        for (int r = 0; r < lanes; r++) {
            assert((a_src[i + r] < b_src[i + r]) == res.f[r]);
        }
    }

    report("f64 Comparison", true);
}

void test_i32_comparison() {
    const int lanes = uv_lanes(32);
    int32_t src1[lanes], src2[lanes], dst[lanes];

    for (int i = 0; i < lanes; i++) {
        src1[i] = (int32_t)(i + 1) * 1.5f;
        src2[i] = (int32_t)(i + 1) * 2.0f;
    }

    v_i32 v_src1 = uv_loadu_i32(src1);
    v_i32 v_src2 = uv_loadu_i32(src2);

    // Test Min
    v_i32 v_res = uv_min_i32(v_src1, v_src2);
    uv_storeu_i32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == _MIN(src1[i], src2[i]));
    }

    // Test Max
    v_res = uv_max_i32(v_src1, v_src2);
    uv_storeu_i32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == _MAX(src1[i], src2[i]));
    }

    // Test Greater-than and Select
    int32_t a_src[UV_MAX_LANES] = {
               1,         -1,         0,         5,
               1, -INT32_MAX, INT32_MAX,         1,
       INT32_MIN,  INT32_MIN,     1e30f,         0,
       INT32_MIN,  INT32_MIN,     1e30f,         0
    };
    int32_t b_src[UV_MAX_LANES] = {
            0.5f,         -2,         1,         5,
       INT32_MAX,          0, INT32_MAX, INT32_MIN,
               1,  INT32_MIN,     1e30f,         0,
               1,  INT32_MIN,     1e30f,         0
    };
    for (int i = 0; i < UV_MAX_LANES; i += lanes) {
        union { v_i32 epi32; int32_t i[UV_MAX_LANES]; } res;
        res.epi32 = scalar_cmpgt_i32(a_src + i, b_src + i);
        for (int r = 0; r < lanes; r++) {
            assert((a_src[i + r] > b_src[i + r]) == res.i[r]);
        }
    }

    // Test Less-than
    for (int i = 0; i < UV_MAX_LANES; i += lanes) {
        union { v_i32 epi32; int32_t i[UV_MAX_LANES]; } res;
        res.epi32 = scalar_cmplt_i32(a_src + i, b_src + i);
        for (int r = 0; r < lanes; r++) {
            assert((a_src[i + r] < b_src[i + r]) == res.i[r]);
        }
    }

    report("i32 Comparison", true);
}

/**
 * Test Bitwise/Logic
 * We compare bit patterns (uint32_t) to avoid NaN comparison issues.
 */
void test_f32_logic() {
    const int lanes = uv_lanes(32);
    float src1[lanes], src2[lanes], dst[lanes];

    for (int i = 0; i < lanes; i++) {
        src1[i] = 1.0f;
        src2[i] = 2.0f;
    }

    v_f32 v1 = uv_loadu_f32(src1);
    v_f32 v2 = uv_loadu_f32(src2);

    // Test AND
    v_f32 v_res = uv_and_f32(v1, v2);
    uv_storeu_f32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        union { float f; uint32_t i; } u_res, u_s1, u_s2;
        u_res.f = dst[i];
        u_s1.f = src1[i];
        u_s2.f = src2[i];
        assert(u_res.i == scalar_and_f32_bits(u_s1.f, u_s2.f));
    }

    // Test ANDNOT
    v_res = uv_andnot_f32(v1, v2);
    uv_storeu_f32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        union { float f; uint32_t i; } u_res, u_s1, u_s2;
        u_res.f = dst[i];
        u_s1.f = src1[i];
        u_s2.f = src2[i];
        assert(u_res.i == scalar_andnot_f32_bits(u_s1.f, u_s2.f));
    }

    // Test OR
    v_res = uv_or_f32(v1, v2);
    uv_storeu_f32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        union { float f; uint32_t i; } u_res, u_s1, u_s2;
        u_res.f = dst[i];
        u_s1.f = src1[i];
        u_s2.f = src2[i];
        assert(u_res.i == scalar_or_f32_bits(u_s1.f, u_s2.f));
    }

    // Test XOR
    v_res = uv_xor_f32(v1, v2);
    uv_storeu_f32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        union { float f; uint32_t i; } u_res, u_s1, u_s2;
        u_res.f = dst[i];
        u_s1.f = src1[i];
        u_s2.f = src2[i];
        assert(u_res.i == scalar_xor_f32_bits(u_s1.f, u_s2.f));
    }

    report("f32 Bitwise Logic", true);
}

void test_f64_logic() {
    const int lanes = uv_lanes(64);
    double src1[lanes], src2[lanes], dst[lanes];

    for (int i = 0; i < lanes; i++) {
        src1[i] = 1.0f;
        src2[i] = 2.0f;
    }

    v_f64 v1 = uv_loadu_f64(src1);
    v_f64 v2 = uv_loadu_f64(src2);

    // Test AND
    v_f64 v_res = uv_and_f64(v1, v2);
    uv_storeu_f64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        union { double f; uint64_t i; } u_res, u_s1, u_s2;
        u_res.f = dst[i];
        u_s1.f = src1[i];
        u_s2.f = src2[i];
        assert(u_res.i == scalar_and_f64_bits(u_s1.f, u_s2.f));
    }

    // Test ANDNOT
    v_res = uv_andnot_f64(v1, v2);
    uv_storeu_f64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        union { double f; uint64_t i; } u_res, u_s1, u_s2;
        u_res.f = dst[i];
        u_s1.f = src1[i];
        u_s2.f = src2[i];
        assert(u_res.i == scalar_andnot_f64_bits(u_s1.f, u_s2.f));
    }

    // Test OR
    v_res = uv_or_f64(v1, v2);
    uv_storeu_f64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        union { double f; uint64_t i; } u_res, u_s1, u_s2;
        u_res.f = dst[i];
        u_s1.f = src1[i];
        u_s2.f = src2[i];
        assert(u_res.i == scalar_or_f64_bits(u_s1.f, u_s2.f));
    }

    // Test XOR
    v_res = uv_xor_f64(v1, v2);
    uv_storeu_f64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        union { double f; uint64_t i; } u_res, u_s1, u_s2;
        u_res.f = dst[i];
        u_s1.f = src1[i];
        u_s2.f = src2[i];
        assert(u_res.i == scalar_xor_f64_bits(u_s1.f, u_s2.f));
    }

    report("f64 Bitwise Logic", true);
}

void test_i32_logic() {
    const int lanes = uv_lanes(32);
    int32_t src1[lanes], src2[lanes], dst[lanes];

    for (int i = 0; i < lanes; i++) {
        src1[i] = 1;
        src2[i] = 2;
    }

    v_i32 v1 = uv_loadu_i32(src1);
    v_i32 v2 = uv_loadu_i32(src2);

    // Test AND
    v_i32 v_res = uv_and_i32(v1, v2);
    uv_storeu_i32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == (src1[i] & src2[i]));
    }

    // Test ANDNOT
    v_res = uv_andnot_i32(v1, v2);
    uv_storeu_i32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == (~src1[i] & src2[i]));
    }

    // Test OR
    v_res = uv_or_i32(v1, v2);
    uv_storeu_i32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == (src1[i] | src2[i]));
    }

    // Test XOR
    v_res = uv_xor_i32(v1, v2);
    uv_storeu_i32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == (src1[i] ^ src2[i]));
    }

    report("i32 Bitwise Logic", true);
}

void test_f32_convert() {
    const int lanes = uv_lanes(32);
    int32_t i_src[lanes], i_dst[lanes];
    float f_src[lanes], f_dst[lanes];

    for (int i = 0; i < lanes; i++) {
        f_src[i] = (float)(i + 1) * ((i % 2) ? 1.5f : -1.5f);
        i_src[i] = -lanes/2 + i;
    }

    v_i32 i_v = uv_loadu_i32(i_src);
    v_f32 v_f_res = uv_cvt_i32_f32(i_v);
    uv_storeu_f32(f_dst, v_f_res);
    for (int i = 0; i < lanes; i++) {
        assert(f_dst[i] == (float)i_src[i]);
    }

    v_f32 f_v = uv_loadu_f32(f_src);
    v_i32 v_i_res = uv_cvt_f32_i32(f_v);
    uv_storeu_i32(i_dst, v_i_res);
    for (int i = 0; i < lanes; i++) {
        assert(i_dst[i] == (int32_t)scalar_rint_f32(f_src[i]));
    }

    report("f32 Convert", true);
}

void test_f64_convert() {
    const int lanes = uv_lanes(64);
    double f_src[lanes], f_dst[lanes];
    int64_t i_src[lanes], i_dst[lanes];

    for (int i = 0; i < lanes; i++) {
        f_src[i] = (double)(i + 1) * ((i % 2) ? 1.5f : -1.5f);
        i_src[i] = ((i % 2) ? -1 : 1)<<(64-i*lanes);
    }

    v_i64 i_v = uv_loadu_i64(i_src);
    v_f64 v_f_res = uv_cvt_i64_f64(i_v);
    uv_storeu_f64(f_dst, v_f_res);
    for (int i = 0; i < lanes; i++) {
        assert(f_dst[i] == (double)i_src[i]);
    }

    v_f64 f_v = uv_loadu_f64(f_src);
    v_i64 v_i_res = uv_cvt_f64_i64(f_v);
    uv_storeu_i64(i_dst, v_i_res);
    for (int i = 0; i < lanes; i++) {
        assert(i_dst[i] == (int64_t)f_src[i]);
    }

    report("f64 Convert", true);
}

/**
 * Test Integer 8 (Complex Shuffles)
 */
void test_i8_complex() {
    const int bytes = UV_LANES_8;
    int8_t src[bytes];
    int8_t idx[bytes];
    int8_t dst[bytes];
    int8_t expected[bytes];

    for (int i = 0; i < bytes; i++) {
        src[i] = (int8_t)(i + 10);
        idx[i] = (int8_t)((i * 17 + 5) % bytes);
    }

    for (int i = 0; i < bytes; i++) {
        expected[i] = src[idx[i]];
    }

    v_i32 v_src = uv_loadu_i8(src);
    v_i32 v_idx = uv_loadu_i8(idx); // Loaded the cross-lane idx pattern here

    v_i32 v_res = uv_permutexvar_i8(v_idx, v_src);
    uv_storeu_i8(dst, v_res);

    for (int i = 0; i < bytes; i++) {
        assert(dst[i] == expected[i]);
    }

    report("i8 Complex (Shuffle/Permute)", true);
}

/**
 * Test Reductions
 */
void test_reductions() {
    const int lanes = uv_lanes(32);
    float src[lanes];
    for (int i = 0; i < lanes; i++) src[i] = (float)i;

    v_f32 v_src = uv_loadu_f32(src);

    float res_add = uv_reduce_add_f32(v_src);
    float expected_add = 0;
    for (int i = 0; i < lanes; i++) expected_add += src[i];
    assert(fabsf(res_add - expected_add) < TEST_epsilon);

    float res_min = uv_reduce_min_f32(v_src);
    float res_max = uv_reduce_max_f32(v_src);

    assert(res_min <= res_max);

    report("Reductions", true);
}

/**
 * Test Bit Manipulation
 */
void test_bit_ops() {
    uint64_t val = 0x000000000000000F;
    assert(uv_popcnt_u64(val) == 4);
    assert(uv_tzcnt_u64(val) == 0);
    assert(uv_lzcnt_u64(val) == 60);

    report("Bit Manipulation", true);
}

int main() {
    printf("Starting UVX SIMD Test Suite...\n");
    printf("Target: UV_REGISTERS=%d, UV_REGISTER_WIDTH=%d\n\n", UV_REGISTERS, UV_REGISTER_WIDTH);

    test_f64_arithmetic();
    test_f64_fmadd();
    test_f64_comparison();
    test_f64_logic();
    test_f64_convert();

    test_f32_arithmetic();
    test_f32_fmadd();
    test_f32_comparison();
    test_f32_logic();
    test_f32_convert();

    test_i64_arithmetic();
//  test_i64_comparison();
//  test_i64_logic();

    test_i32_arithmetic();
    test_i32_comparison();
    test_i32_logic();

    test_i8_complex();
    test_reductions();
    test_bit_ops();

    printf("\nAll tests passed successfully.\n");
    return 0;
}

