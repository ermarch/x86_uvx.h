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

template <typename T> struct uvx_traits;

template <> struct uvx_traits<float> {
    using vector_type = v_f32;
    using mask_type = v_mask;
    static inline v_f32 dup(float v) { return uv_dup_f32(v); }
    static inline v_f32 load(const float* p) { return uv_loadu_f32(p); }
    static inline int testz(v_f32 v, v_f32 m) { return uv_testz_f32(v, m); }
    static inline v_mask cmpeq(v_f32 a, v_f32 b) { return uv_cmpeq_f32(a, b); }
    static inline v_mask cmplt(v_f32 a, v_f32 b) { return uv_cmplt_f32(a, b); }
    static inline v_mask cmple(v_f32 a, v_f32 b) { return uv_cmple_f32(a, b); }
    static inline v_mask cmpne(v_f32 a, v_f32 b) { return uv_cmpne_f32(a, b); }
    static inline v_mask cmpnlt(v_f32 a, v_f32 b) { return uv_cmpnlt_f32(a, b); }
    static inline v_mask cmpge(v_f32 a, v_f32 b) { return uv_cmpge_f32(a, b); }
    static inline v_mask cmpnle(v_f32 a, v_f32 b) { return uv_cmpnle_f32(a, b); }
    static inline v_mask cmpgt(v_f32 a, v_f32 b) { return uv_cmpgt_f32(a, b); }
    static inline v_f32 select(v_mask m, v_f32 t, v_f32 f) {
        return uv_select_f32(m, t, f);
    }
};

template <> struct uvx_traits<int32_t> {
    using vector_type = v_i32;
    using mask_type = v_mask_i8;
    static inline v_i32 dup(float v) { return uv_dup_i32(v); }
    static inline v_i32 load(const int32_t* p) { return uv_loadu_i32(p); }
    static inline int testz(v_i32 v, v_i32 m) { return uv_testz_i32(v, m); }
    static inline v_mask_i8 cmpeq(v_i32 a, v_i32 b) { return uv_cmpeq_i32(a, b); }
    static inline v_mask_i8 cmplt(v_i32 a, v_i32 b) { return uv_cmplt_i32(a, b); }
    static inline v_mask_i8 cmple(v_i32 a, v_i32 b) { return uv_cmple_i32(a, b); }
    static inline v_mask_i8 cmpne(v_i32 a, v_i32 b) { return uv_cmpne_i32(a, b); }
    static inline v_mask_i8 cmpnlt(v_i32 a, v_i32 b) { return uv_cmpnlt_i32(a, b); }
    static inline v_mask_i8 cmpge(v_i32 a, v_i32 b) { return uv_cmpge_i32(a, b); }
    static inline v_mask_i8 cmpnle(v_i32 a, v_i32 b) { return uv_cmpnle_i32(a, b); }
    static inline v_mask_i8 cmpgt(v_i32 a, v_i32 b) { return uv_cmpgt_i32(a, b); }
    static inline v_i32 select(v_mask_i8 m, v_i32 t, v_i32 f) {
        return uv_select_i32(m, t, f);
    }
};

template <> struct uvx_traits<double> {
    using vector_type = v_f64;
    using mask_type = v_mask_f64;
    static inline v_f64 dup(double v) { return uv_dup_f64(v); }
    static inline v_f64 load(const double* p) { return uv_loadu_f64(p); }
    static inline int testz(v_f64 v, v_f64 m) { return uv_testz_f64(v, m); }
    static inline v_mask_f64 cmpeq(v_f64 a, v_f64 b) { return uv_cmpeq_f64(a, b); }
    static inline v_mask_f64 cmplt(v_f64 a, v_f64 b) { return uv_cmplt_f64(a, b); }
    static inline v_mask_f64 cmple(v_f64 a, v_f64 b) { return uv_cmple_f64(a, b); }
    static inline v_mask_f64 cmpne(v_f64 a, v_f64 b) { return uv_cmpne_f64(a, b); }
    static inline v_mask_f64 cmpnlt(v_f64 a, v_f64 b) { return uv_cmpnlt_f64(a, b); }
    static inline v_mask_f64 cmpge(v_f64 a, v_f64 b) { return uv_cmpge_f64(a, b); }
    static inline v_mask_f64 cmpnle(v_f64 a, v_f64 b) { return uv_cmpnle_f64(a, b); }
    static inline v_mask_f64 cmpgt(v_f64 a, v_f64 b) { return uv_cmpgt_f64(a, b); }
    static inline v_f64 select(v_mask_f64 m, v_f64 t, v_f64 f) {
        return uv_select_f64(m, t, f);
    }
};

template <> struct uvx_traits<int64_t> {
    using vector_type = v_i64;
    using mask_type = v_mask_i8;
    static inline v_i64 dup(float v) { return uv_dup_i64(v); }
    static inline v_i64 load(const int64_t* p) { return uv_loadu_i64(p); }
    static inline int testz(v_i64 v, v_i64 m) { return uv_testz_i64(v, m); }
    static inline v_mask_i8 cmpeq(v_i64 a, v_i64 b) { return uv_cmpeq_i64(a, b); }
    static inline v_mask_i8 cmplt(v_i64 a, v_i64 b) { return uv_cmplt_i64(a, b); }
    static inline v_mask_i8 cmple(v_i64 a, v_i64 b) { return uv_cmple_i64(a, b); }
    static inline v_mask_i8 cmpne(v_i64 a, v_i64 b) { return uv_cmpne_i64(a, b); }
    static inline v_mask_i8 cmpnlt(v_i64 a, v_i64 b) { return uv_cmpnlt_i64(a, b); }
    static inline v_mask_i8 cmpge(v_i64 a, v_i64 b) { return uv_cmpge_i64(a, b); }
    static inline v_mask_i8 cmpnle(v_i64 a, v_i64 b) { return uv_cmpnle_i64(a, b); }
    static inline v_mask_i8 cmpgt(v_i64 a, v_i64 b) { return uv_cmpgt_i64(a, b); }
    static inline v_i64 select(v_mask_i8 m, v_i64 t, v_i64 f) {
        return uv_select_i64(m, t, f);
    }
};

/*
 * Scalar bitwise helpers
 * These return the raw bit pattern as a I.
 */
template<typename F, typename I>
static inline I scalar_and(F a, F b) {
    union { F f; I i; } va, vb, vr;
    va.f = a; vb.f = b; vr.i = va.i & vb.i;
    return vr.i;
}

template<typename F, typename I>
static inline I scalar_andnot(F a, F b) {
    union { F f; I i; } va, vb, vr;
    va.f = a; vb.f = b; vr.i = ~va.i & vb.i;
    return vr.i;
}

template<typename F, typename I>
static inline I scalar_or(F a, F b) {
    union { F f; I i; } va, vb, vr;
    va.f = a; vb.f = b; vr.i = va.i | vb.i;
    return vr.i;
}

template<typename F, typename I>
static inline I scalar_xor(F a, F b) {
    union { F f; I i; } va, vb, vr;
    va.f = a; vb.f = b; vr.i = va.i ^ vb.i;
    return vr.i;
}

template<typename F, typename I>
static inline F scalar_rint(F x) {
    F magic = copysign(8388608.0f, x);
    return (I)((x + magic) - magic);
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

    // Test Sqrt
    v_res = uv_sqrt_f32(v_src1);
    uv_storeu_f32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == sqrtf(src1[i]));
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

    // Test Sqrt
    v_res = uv_sqrt_f64(v_src1);
    uv_storeu_f64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == sqrt(src1[i]));
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
    float z_src[uv_max_lanes(32)] = {
               0,          0,         0,         0,
               0,          0,         0,         0,
               0,          0,         0,         0,
               0,          0,         0,         0
    };
    float a_src[uv_max_lanes(32)] = {
            1.0f,     -1.0f,     0.0f,  5.0f,
            1.0f, -INFINITY, INFINITY,  1.0f,
             NAN,       NAN,    1e30f,  0.0f,
             NAN,       NAN,    1e30f,  0.0f
    };
    float b_src[uv_max_lanes(32)] = {
            0.5f,     -2.0f,     1.0f,  5.0f,
        INFINITY,      0.0f, INFINITY,   NAN,
            1.0f,       NAN,    1e30f,  0.0f,
            1.0f,       NAN,    1e30f,  0.0f
    };

    union { v_f32 f32; float f[uv_lanes(32)]; } res;

    const v_f32 v_val_true = uvx_traits<float>::dup(1);
    const v_f32 v_val_false = uvx_traits<float>::dup(0);

    v_f32 z = uvx_traits<float>::load(z_src);
    v_f32 a = uvx_traits<float>::load(a_src);
    v_f32 b = uvx_traits<float>::load(b_src);

    // Test Zero
    int zero = (uvx_traits<float>::testz(z, z) == 1);
    res.f[0] = 0;
    for (int r = 0; r < lanes; r++) {
        res.f[0] = scalar_or<float, uint32_t>(res.f[0], z_src[r]);
    }
    assert(zero == (res.f[0] == 0));

    zero = (uvx_traits<float>::testz(a, a) == 1);
    res.f[0] = 0;
    for (int r = 0; r < lanes; r++) {
        res.f[0] = scalar_or<float, uint32_t>(res.f[0], a_src[r]);
    }
    assert(zero == (res.f[0] == 0));

    // Test Equal
    v_mask mask = uvx_traits<float>::cmpeq(a, b);
    res.f32 = uvx_traits<float>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert((a_src[r] == b_src[r]) == res.f[r]);
    }

    // Test Less-than
    mask = uvx_traits<float>::cmplt(a, b);
    res.f32 = uvx_traits<float>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert((a_src[r] < b_src[r]) == res.f[r]);
    }

    // Test Less-than or Equal
    mask = (uvx_traits<float>::cmple(a, b));
    res.f32 = uvx_traits<float>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert((a_src[r] <= b_src[r]) == res.f[r]);
    }

    // Test Not Equal
    mask = (uvx_traits<float>::cmpne(a, b));
    res.f32 = uvx_traits<float>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert((a_src[r] != b_src[r]) == res.f[r]);
    }

    // Test Not Less-than
    mask = (uvx_traits<float>::cmpnlt(a, b));
    res.f32 = uvx_traits<float>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert(!(a_src[r] < b_src[r]) == res.f[r]);
    }

    // Test Greater-than or Equal
    mask = (uvx_traits<float>::cmpge(a, b));
    res.f32 = uvx_traits<float>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert((a_src[r] >= b_src[r]) == res.f[r]);
    }

    // Test Not Less-than or Equal
    mask = (uvx_traits<float>::cmpnle(a, b));
    res.f32 = uvx_traits<float>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert(!(a_src[r] <= b_src[r]) == res.f[r]);
    }

    // Test Greater-than
    mask = (uvx_traits<float>::cmpgt(a, b));
    res.f32 = uvx_traits<float>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert((a_src[r] > b_src[r]) == res.f[r]);
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
    double z_src[uv_max_lanes(32)] = {
               0,          0,         0,         0,
               0,          0,         0,         0,
               0,          0,         0,         0,
               0,          0,         0,         0
    };
    double a_src[uv_max_lanes(32)] = {
             1.0,      -1.0,      0.0,   5.0,
             1.0, -INFINITY, INFINITY,   1.0,
             NAN,       NAN,     1e30,   0.0,
             NAN,       NAN,     1e30,   0.0
    };
    double b_src[uv_max_lanes(32)] = {
             0.5,      -2.0,      1.0,   5.0,
        INFINITY,        0.0, INFINITY,  NAN,
             1.0,       NAN,      1e30,  0.0,
             1.0,       NAN,      1e30,  0.0
    };

    union { v_f64 f64; double f[uv_lanes(32)]; } res;

    const v_f64 v_val_true = uvx_traits<double>::dup(1);
    const v_f64 v_val_false = uvx_traits<double>::dup(0);

    v_f64 z = uvx_traits<double>::load(z_src);
    v_f64 a = uvx_traits<double>::load(a_src);
    v_f64 b = uvx_traits<double>::load(b_src);

    // Test Zero
    int zero = (uvx_traits<double>::testz(z, z) == 1);
    res.f[0] = 0;
    for (int r = 0; r < lanes; r++) {
        res.f[0] = scalar_or<double, uint64_t>(res.f[0], z_src[r]);
    }
    assert(zero == (res.f[0] == 0));

    zero = (uvx_traits<double>::testz(a, a) == 1);
    res.f[0] = 0;
    for (int r = 0; r < lanes; r++) {
        res.f[0] = scalar_or<double, uint64_t>(res.f[0], a_src[r]);
    }
    assert(zero == (res.f[0] == 0));

    // Test Equal
    v_mask_f64 mask = uvx_traits<double>::cmpeq(a, b);
    res.f64 = uvx_traits<double>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert((a_src[r] == b_src[r]) == res.f[r]);
    }

    // Test Less-than
    mask = uvx_traits<double>::cmplt(a, b);
    res.f64 = uvx_traits<double>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert((a_src[r] < b_src[r]) == res.f[r]);
    }

    // Test Less-than or Equal
    mask = (uvx_traits<double>::cmple(a, b));
    res.f64 = uvx_traits<double>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert((a_src[r] <= b_src[r]) == res.f[r]);
    }

    // Test Not Equal
    mask = (uvx_traits<double>::cmpne(a, b));
    res.f64 = uvx_traits<double>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert((a_src[r] != b_src[r]) == res.f[r]);
    }

    // Test Not Less-than
    mask = (uvx_traits<double>::cmpnlt(a, b));
    res.f64 = uvx_traits<double>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert(!(a_src[r] < b_src[r]) == res.f[r]);
    }

    // Test Greater-than or Equal
    mask = (uvx_traits<double>::cmpge(a, b));
    res.f64 = uvx_traits<double>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert((a_src[r] >= b_src[r]) == res.f[r]);
    }

    // Test Not Less-than or Equal
    mask = (uvx_traits<double>::cmpnle(a, b));
    res.f64 = uvx_traits<double>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert(!(a_src[r] <= b_src[r]) == res.f[r]);
    }

    // Test Greater-than
    mask = (uvx_traits<double>::cmpgt(a, b));
    res.f64 = uvx_traits<double>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert((a_src[r] > b_src[r]) == res.f[r]);
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
    int32_t z_src[uv_max_lanes(32)] = {
               0,          0,         0,         0,
               0,          0,         0,         0,
               0,          0,         0,         0,
               0,          0,         0,         0
    };
    int32_t a_src[uv_max_lanes(32)] = {
               1,         -1,         0,         5,
               1, -INT32_MAX, INT32_MAX,         1,
       INT32_MIN,  INT32_MIN, 100000030,         0,
       INT32_MIN,  INT32_MIN, -10000030,         0
    };
    int32_t b_src[uv_max_lanes(32)] = {
               5,         -2,         1,         5,
       INT32_MAX,          0, INT32_MAX, INT32_MIN,
               1,  INT32_MIN, 100000030,         0,
               1,  INT32_MIN, -10000030,         0
    };

    union { v_i32 epi32; int32_t i[uv_lanes(32)]; } res;

    const v_i32 v_val_true = uvx_traits<int32_t>::dup(1);
    const v_i32 v_val_false = uvx_traits<int32_t>::dup(0);

    v_i32 z = uvx_traits<int32_t>::load(z_src);
    v_i32 a = uvx_traits<int32_t>::load(a_src);
    v_i32 b = uvx_traits<int32_t>::load(b_src);

    // Test Zero
    int zero = (uvx_traits<int32_t>::testz(z, z) == 1);
    res.i[0] = 0;
    for (int r = 0; r < lanes; r++) {
        res.i[0] |= z_src[r];
    }
    assert(zero == (res.i[0] == 0));

    zero = (uvx_traits<int32_t>::testz(a, a) == 1);
    res.i[0] = 0;
    for (int r = 0; r < lanes; r++) {
        res.i[0] |= a_src[r];
    }
    assert(zero == (res.i[0] == 0));

    // Test Equal
    v_mask_i8 mask = uvx_traits<int32_t>::cmpeq(a, b);
    res.epi32 = uvx_traits<int32_t>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert((a_src[r] == b_src[r]) == res.i[r]);
    }

    // Test Less-than
    mask = uvx_traits<int32_t>::cmplt(a, b);
    res.epi32 = uvx_traits<int32_t>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert((a_src[r] < b_src[r]) == res.i[r]);
    }

    // Test Less-than or Equal
    mask = (uvx_traits<int32_t>::cmple(a, b));
    res.epi32 = uvx_traits<int32_t>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert((a_src[r] <= b_src[r]) == res.i[r]);
    }

    // Test Not Equal
    mask = (uvx_traits<int32_t>::cmpne(a, b));
    res.epi32 = uvx_traits<int32_t>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert((a_src[r] != b_src[r]) == res.i[r]);
    }

    // Test Not Less-than
    mask = (uvx_traits<int32_t>::cmpnlt(a, b));
    res.epi32 = uvx_traits<int32_t>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert((!(a_src[r] < b_src[r])) == res.i[r]);
    }

    // Test Greater-than or Equal
    mask = (uvx_traits<int32_t>::cmpge(a, b));
    res.epi32 = uvx_traits<int32_t>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert((a_src[r] >= b_src[r]) == res.i[r]);
    }

    // Test Not Less-than or Equal
    mask = (uvx_traits<int32_t>::cmpnle(a, b));
    res.epi32 = uvx_traits<int32_t>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert((!(a_src[r] <= b_src[r])) == res.i[r]);
    }

    // Test Greater-than
    mask = (uvx_traits<int32_t>::cmpgt(a, b));
    res.epi32 = uvx_traits<int32_t>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert((a_src[r] > b_src[r]) == res.i[r]);
    }

    report("i32 Comparison", true);
}

void test_i64_comparison() {
    const int lanes = uv_lanes(64);
    int64_t src1[lanes], src2[lanes], dst[lanes];

    for (int i = 0; i < lanes; i++) {
        src1[i] = (int64_t)(i + 1) * 1.5f;
        src2[i] = (int64_t)(i + 1) * 2.0f;
    }

    v_i64 v_src1 = uv_loadu_i64(src1);
    v_i64 v_src2 = uv_loadu_i64(src2);

    // Test Min
    v_i64 v_res = uv_min_i64(v_src1, v_src2);
    uv_storeu_i64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == _MIN(src1[i], src2[i]));
    }

    // Test Max
    v_res = uv_max_i64(v_src1, v_src2);
    uv_storeu_i64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == _MAX(src1[i], src2[i]));
    }

    // Comparison and and Select
    assert(8 == uv_max_lanes(64));
    int64_t z_src[uv_max_lanes(64)] = {
               0,          0,         0,         0,
               0,          0,         0,         0
    };
    int64_t a_src[uv_max_lanes(64)] = {
               1,         -1,         0,         5,
               1, -INT64_MAX, INT64_MAX,         1
    };
    int64_t b_src[uv_max_lanes(64)] = {
            5000,         -2,         1,         5,
       INT64_MAX,          0, INT64_MAX, INT64_MIN
    };

    union { v_i64 epi64; int64_t i[uv_lanes(64)]; } res;

    const v_i64 v_val_true = uvx_traits<int64_t>::dup(1);
    const v_i64 v_val_false = uvx_traits<int64_t>::dup(0);

    v_i64 z = uvx_traits<int64_t>::load(z_src);
    v_i64 a = uvx_traits<int64_t>::load(a_src);
    v_i64 b = uvx_traits<int64_t>::load(b_src);

    // Test Zero
    int zero = (uvx_traits<int64_t>::testz(z, z) == 1);
    res.i[0] = 0;
    for (int r = 0; r < lanes; r++) {
        res.i[0] |= z_src[r];
    }
    assert(zero == (res.i[0] == 0));

    zero = (uvx_traits<int64_t>::testz(a, a) == 1);
    res.i[0] = 0;
    for (int r = 0; r < lanes; r++) {
        res.i[0] |= a_src[r];
    }
    assert(zero == (res.i[0] == 0));

    // Test Equal
    v_mask_i8 mask = uvx_traits<int64_t>::cmpeq(a, b);
    res.epi64 = uvx_traits<int64_t>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert((a_src[r] == b_src[r]) == res.i[r]);
    }

    // Test Less-than
    mask = uvx_traits<int64_t>::cmplt(a, b);
    res.epi64 = uvx_traits<int64_t>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert((a_src[r] < b_src[r]) == res.i[r]);
    }

    // Test Less-than or Equal
    mask = (uvx_traits<int64_t>::cmple(a, b));
    res.epi64 = uvx_traits<int64_t>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert((a_src[r] <= b_src[r]) == res.i[r]);
    }

    // Test Not Equal
    mask = (uvx_traits<int64_t>::cmpne(a, b));
    res.epi64 = uvx_traits<int64_t>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert((a_src[r] != b_src[r]) == res.i[r]);
    }

    // Test Not Less-than
    mask = (uvx_traits<int64_t>::cmpnlt(a, b));
    res.epi64 = uvx_traits<int64_t>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert((!(a_src[r] < b_src[r])) == res.i[r]);
    }

    // Test Greater-than or Equal
    mask = (uvx_traits<int64_t>::cmpge(a, b));
    res.epi64 = uvx_traits<int64_t>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert((a_src[r] >= b_src[r]) == res.i[r]);
    }

    // Test Not Less-than or Equal
    mask = (uvx_traits<int64_t>::cmpnle(a, b));
    res.epi64 = uvx_traits<int64_t>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert((!(a_src[r] <= b_src[r])) == res.i[r]);
    }

    // Test Greater-than
    mask = (uvx_traits<int64_t>::cmpgt(a, b));
    res.epi64 = uvx_traits<int64_t>::select(mask, v_val_true, v_val_false);
    for (int r = 0; r < lanes; r++) {
        assert((a_src[r] > b_src[r]) == res.i[r]);
    }
    report("i64 Comparison", true);
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

    // Test NOT
    v_f32 v_res = uv_not_f32(v1);
    uv_storeu_f32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        union { float f; uint32_t i; } u_res, u_s1;
        u_res.f = dst[i];
        u_s1.f = src1[i];
        assert(u_res.i == ~u_s1.i);
    }

    // Test AND
    v_res = uv_and_f32(v1, v2);
    uv_storeu_f32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        union { float f; uint32_t i; } u_res, u_s1, u_s2;
        u_res.f = dst[i];
        u_s1.f = src1[i];
        u_s2.f = src2[i];
        assert(u_res.i == (scalar_and<float, uint32_t>(u_s1.f, u_s2.f)));
    }

    // Test ANDNOT
    v_res = uv_andnot_f32(v1, v2);
    uv_storeu_f32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        union { float f; uint32_t i; } u_res, u_s1, u_s2;
        u_res.f = dst[i];
        u_s1.f = src1[i];
        u_s2.f = src2[i];
        assert(u_res.i == (scalar_andnot<float, uint32_t>(u_s1.f, u_s2.f)));
    }

    // Test OR
    v_res = uv_or_f32(v1, v2);
    uv_storeu_f32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        union { float f; uint32_t i; } u_res, u_s1, u_s2;
        u_res.f = dst[i];
        u_s1.f = src1[i];
        u_s2.f = src2[i];
        assert(u_res.i == (scalar_or<float, uint32_t>(u_s1.f, u_s2.f)));
    }

    // Test XOR
    v_res = uv_xor_f32(v1, v2);
    uv_storeu_f32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        union { float f; uint32_t i; } u_res, u_s1, u_s2;
        u_res.f = dst[i];
        u_s1.f = src1[i];
        u_s2.f = src2[i];
        assert(u_res.i == (scalar_xor<float, uint32_t>(u_s1.f, u_s2.f)));
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

    // Test NOT
    v_f64 v_res = uv_not_f64(v1);
    uv_storeu_f64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        union { double f; uint64_t i; } u_res, u_s1;
        u_res.f = dst[i];
        u_s1.f = src1[i];
        assert(u_res.i == ~u_s1.i);
    }

    // Test AND
    v_res = uv_and_f64(v1, v2);
    uv_storeu_f64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        union { double f; uint64_t i; } u_res, u_s1, u_s2;
        u_res.f = dst[i];
        u_s1.f = src1[i];
        u_s2.f = src2[i];
        assert(u_res.i == (scalar_and<double, uint64_t>(u_s1.f, u_s2.f)));
    }

    // Test ANDNOT
    v_res = uv_andnot_f64(v1, v2);
    uv_storeu_f64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        union { double f; uint64_t i; } u_res, u_s1, u_s2;
        u_res.f = dst[i];
        u_s1.f = src1[i];
        u_s2.f = src2[i];
        assert(u_res.i == (scalar_andnot<double, uint64_t>(u_s1.f, u_s2.f)));
    }

    // Test OR
    v_res = uv_or_f64(v1, v2);
    uv_storeu_f64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        union { double f; uint64_t i; } u_res, u_s1, u_s2;
        u_res.f = dst[i];
        u_s1.f = src1[i];
        u_s2.f = src2[i];
        assert(u_res.i == (scalar_or<double, uint64_t>(u_s1.f, u_s2.f)));
    }

    // Test XOR
    v_res = uv_xor_f64(v1, v2);
    uv_storeu_f64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        union { double f; uint64_t i; } u_res, u_s1, u_s2;
        u_res.f = dst[i];
        u_s1.f = src1[i];
        u_s2.f = src2[i];
        assert(u_res.i == (scalar_xor<double, uint64_t>(u_s1.f, u_s2.f)));
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

    // Test NOT
    v_i32 v_res = uv_not_i32(v1);
    uv_storeu_i32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == ~src1[i]);
    }

    // Test AND
    v_res = uv_and_i32(v1, v2);
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

void test_i64_logic() {
    const int lanes = uv_lanes(64);
    int64_t src1[lanes], src2[lanes], dst[lanes];

    for (int i = 0; i < lanes; i++) {
        src1[i] = 1;
        src2[i] = 2;
    }

    v_i64 v1 = uv_loadu_i64(src1);
    v_i64 v2 = uv_loadu_i64(src2);

    // Test NOT
    v_i64 v_res = uv_not_i64(v1);
    uv_storeu_i64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == ~src1[i]);
    }

    // Test AND
    v_res = uv_and_i64(v1, v2);
    uv_storeu_i64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == (src1[i] & src2[i]));
    }

    // Test ANDNOT
    v_res = uv_andnot_i64(v1, v2);
    uv_storeu_i64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == (~src1[i] & src2[i]));
    }

    // Test OR
    v_res = uv_or_i64(v1, v2);
    uv_storeu_i64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == (src1[i] | src2[i]));
    }

    // Test XOR
    v_res = uv_xor_i64(v1, v2);
    uv_storeu_i64(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == (src1[i] ^ src2[i]));
    }

    report("i64 Bitwise Logic", true);
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
        assert(i_dst[i] == (scalar_rint<float, int32_t>(f_src[i])));
    }

    report("f32 Convert", true);
}

void test_f64_convert() {
    const int lanes = uv_lanes(64);
    double f_src[lanes], f_dst[lanes];
    int64_t i_src[lanes], i_dst[lanes];

    // Values chosen within the exactly-representable double range (|v| < 2^53).
    // The original expression ((i % 2) ? -1 : 1) << (64 - i*lanes) is UB for
    // shift counts >= 64 and would silently pass or crash depending on ISA.
    static const int64_t safe_vals[8] = {
        0LL, 1LL, -1LL, 4503599627370495LL,          // 2^52 - 1
        -4503599627370495LL, 1LL << 52, -(1LL << 52), 1234567890123LL
    };
    for (int i = 0; i < lanes; i++) {
        f_src[i] = (double)(i + 1) * ((i % 2) ? 1.5f : -1.5f);
        i_src[i] = safe_vals[i % 8];
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
 * The original version only exercised the calls without verifying results.
 */
void test_i8_complex() {
    const int lanes = uv_lanes(8);
    assert(lanes >= 1 && lanes <= 64);

    int8_t src[64], dst[64];
    for (int i = 0; i < 64; i++) src[i] = (int8_t)i;

    // Permutexvar with an identity index must reproduce src exactly on every
    // ISA level (this catches lane-grouping bugs in the 256/512-bit emulations).
    int8_t idx_id[64];
    for (int i = 0; i < 64; i++) idx_id[i] = (int8_t)(i < lanes ? i : 0);

    v_i8 v_src = uv_loadu_i8(src);
    v_i8 v_idx = uv_loadu_i8(idx_id);
    v_i8 v_res = uv_permutexvar_i8(v_idx, v_src);
    uv_storeu_i8(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == src[i]);
    }

    // Reverse permutation within the vector length.
    int8_t idx_rev[64];
    for (int i = 0; i < 64; i++) {
        idx_rev[i] = (int8_t)(i < lanes ? (lanes - 1 - i) : 0);
    }
    v_idx = uv_loadu_i8(idx_rev);
    v_res = uv_permutexvar_i8(v_idx, v_src);
    uv_storeu_i8(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == src[lanes - 1 - i]);
    }

    // Byte shuffle with identity control must reproduce src on every level:
    // the in-lane masking (idx & 0x0F) maps each byte back onto itself.
    v_idx = uv_loadu_i8(idx_id);
    v_res = uv_shuffle_i8(v_src, v_idx);
    uv_storeu_i8(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == src[i]);
    }

    report("i8 Complex (Shuffle/Permute)", true);
}

/**
 * Test i8 compress and select with a sparse mask.
 * Mask representation differs per level (__mmask64 on AVX-512, byte vector
 * with the high bit set per lane on AVX2/SSE2), so build it per level.
 */
void test_i8_compress_select() {
    const int lanes = uv_lanes(8);
    int8_t src[64], dst[64], exp[64];
    for (int i = 0; i < 64; i++) { src[i] = (int8_t)(i + 1); exp[i] = 0; }

    uint64_t maskbits = 0;
    for (int i = 0; i < lanes; i += 2) maskbits |= (1ULL << i);

    int d = 0;
    for (int i = 0; i < lanes; i++) {
        if ((maskbits >> i) & 1) exp[d++] = src[i];
    }

#if defined(__AVX10__) || defined(__AVX512F__)
    v_i8 v_res = uv_compress_i8((__mmask64)maskbits, uv_loadu_i8(src));
#else
    int8_t mbytes[64];
    for (int i = 0; i < 64; i++) mbytes[i] = (int8_t)(((maskbits >> i) & 1) ? 0x80 : 0x00);
    v_i8 v_res = uv_compress_i8(uv_loadu_i8(mbytes), uv_loadu_i8(src));
#endif
    uv_storeu_i8(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == exp[i]);
    }

    // Select with the same sparse mask.
    int8_t tsrc[64], fsrc[64];
    for (int i = 0; i < 64; i++) { tsrc[i] = (int8_t)(0x40 + i); fsrc[i] = (int8_t)(0x80 + i); }
#if defined(__AVX10__) || defined(__AVX512F__)
    v_res = uv_select_i8((__mmask64)maskbits, uv_loadu_i8(tsrc), uv_loadu_i8(fsrc));
#else
    v_res = uv_select_i8(uv_loadu_i8(mbytes), uv_loadu_i8(tsrc), uv_loadu_i8(fsrc));
#endif
    uv_storeu_i8(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        int8_t expected = ((maskbits >> i) & 1) ? tsrc[i] : fsrc[i];
        assert(dst[i] == expected);
    }

    report("i8 Compress/Select", true);
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

/**
 * Regression: uv_testz_* must implement (v AND m) == 0 on the raw bits,
 * not "v equals m everywhere" and not a value conversion of the floats.
 */
void test_testz_semantics() {
    // f32: disjoint bit patterns that are non-zero and unequal.
    {
        const int lanes = uv_max_lanes(32);
        float vbuf[lanes], mbuf[lanes];
        uint32_t vbits = 0xF0F0F0F0u, mbits = 0x0F0F0F0Fu;
        for (int i = 0; i < lanes; i++) {
            memcpy(&vbuf[i], &vbits, 4);
            memcpy(&mbuf[i], &mbits, 4);
        }
        v_f32 v = uv_loadu_f32(vbuf);
        v_f32 m = uv_loadu_f32(mbuf);
        assert(uv_testz_f32(v, m) == 1);   // disjoint, but v != m
        assert(uv_testz_f32(v, v) == 0);   // v & v != 0
        assert(uv_testz_f32(uv_setzero_f32(), v) == 1);
        assert(uv_testz_f32(v, uv_setzero_f32()) == 1);
    }
    // f64
    {
        const int lanes = uv_max_lanes(64);
        double vbuf[lanes], mbuf[lanes];
        uint64_t vbits = 0xF0F0F0F0F0F0F0F0ull, mbits = 0x0F0F0F0F0F0F0F0Full;
        for (int i = 0; i < lanes; i++) {
            memcpy(&vbuf[i], &vbits, 8);
            memcpy(&mbuf[i], &mbits, 8);
        }
        v_f64 v = uv_loadu_f64(vbuf);
        v_f64 m = uv_loadu_f64(mbuf);
        assert(uv_testz_f64(v, m) == 1);
        assert(uv_testz_f64(v, v) == 0);
        assert(uv_testz_f64(uv_setzero_f64(), v) == 1);
    }
    // i32 / i64
    {
        v_i32 v32 = uv_dup_i32((int32_t)0xF0F0F0F0u);
        v_i32 m32 = uv_dup_i32((int32_t)0x0F0F0F0Fu);
        assert(uv_testz_i32(v32, m32) == 1);
        assert(uv_testz_i32(v32, v32) == 0);
    }
    {
        v_i64 v64 = uv_dup_i64((int64_t)0xF0F0F0F0F0F0F0F0ull);
        v_i64 m64 = uv_dup_i64((int64_t)0x0F0F0F0F0F0F0F0Full);
        assert(uv_testz_i64(v64, m64) == 1);
        assert(uv_testz_i64(v64, v64) == 0);
    }

    report("testz bit semantics", true);
}

/**
 * Regression: uv_abs_i64 must handle negative values whose bit 31 is clear
 * (the old shl/srai-by-31 sign trick produced wrong results for these).
 */
void test_abs_i64_edge() {
    static const int64_t vals[] = {
        0, 1, -1, -2, -0x7FFFFFFFll, -0x100000000ll,
        0x100000000ll, -123456789012345ll, INT64_MAX, INT64_MIN + 1
    };
    const int n = (int)(sizeof(vals) / sizeof(vals[0]));
    for (int k = 0; k < n; k++) {
        v_i64 v = uv_dup_i64(vals[k]);
        v_i64 r = uv_abs_i64(v);
        int64_t dst[uv_lanes(64)];
        uv_storeu_i64(dst, r);
        int64_t expected = (vals[k] < 0) ? -vals[k] : vals[k];
        for (int i = 0; i < uv_lanes(64); i++) {
            assert(dst[i] == expected);
        }
    }

    report("i64 Abs edge cases", true);
}

/**
 * Regression: uv_shr_i64 (arithmetic) for shift counts 0, 32/33, 63, >= 64,
 * including negative values (the old code clamped imm to 64, which
 * srli_epi64 masks to 0, and had UB in the fill-mask shift for imm == 0).
 */
void test_shr_i64_edges() {
    static const int64_t vals[] = {
        -1, -2, INT64_MIN, -0x100000000ll, 12345, -1234567890123ll, INT64_MAX
    };
    static const unsigned shifts[] = { 0, 1, 7, 31, 32, 33, 47, 62, 63, 64, 70 };
    const int nv = (int)(sizeof(vals) / sizeof(vals[0]));
    const int ns = (int)(sizeof(shifts) / sizeof(shifts[0]));

    for (int k = 0; k < nv; k++) {
        for (int s = 0; s < ns; s++) {
            unsigned imm = shifts[s];
            int64_t expected;
            if (imm >= 64) {
                expected = (vals[k] < 0) ? -1 : 0;   // arithmetic fill
            } else {
                expected = vals[k] >> imm;           // arithmetic, defined
            }
            v_i64 r = uv_shr_i64(uv_dup_i64(vals[k]), imm);
            int64_t dst[uv_lanes(64)];
            uv_storeu_i64(dst, r);
            for (int i = 0; i < uv_lanes(64); i++) {
                assert(dst[i] == expected);
            }

            // Left shift: bits shifted out are lost, >= 64 must give 0.
            // Compute the reference with an unsigned shift (signed << is UB).
            int64_t lexpected =
                (imm >= 64) ? 0 : (int64_t)((uint64_t)vals[k] << imm);
            v_i64 l = uv_shl_i64(uv_dup_i64(vals[k]), imm);
            int64_t ldst[uv_lanes(64)];
            uv_storeu_i64(ldst, l);
            for (int i = 0; i < uv_lanes(64); i++) {
                assert(ldst[i] == lexpected);
            }
        }
    }

    report("i64 Shift edge cases", true);
}

/**
 * uv_shl_i32/uv_shr_i32 for shift counts at and beyond the lane width,
 * where the hardware instructions mask the count to 5 bits.
 */
void test_shift_i32_edges() {
    static const int32_t vals[] = {
        -1, -2, INT32_MIN, -0x10000, 12345, -123456789, INT32_MAX
    };
    static const unsigned shifts[] = { 0, 1, 15, 30, 31, 32, 40 };
    const int nv = (int)(sizeof(vals) / sizeof(vals[0]));
    const int ns = (int)(sizeof(shifts) / sizeof(shifts[0]));

    for (int k = 0; k < nv; k++) {
        for (int s = 0; s < ns; s++) {
            unsigned imm = shifts[s];
            int32_t expected =
                (imm >= 32) ? ((vals[k] < 0) ? -1 : 0) : (vals[k] >> imm);
            v_i32 r = uv_shr_i32(uv_dup_i32(vals[k]), imm);
            int32_t dst[uv_lanes(32)];
            uv_storeu_i32(dst, r);
            for (int i = 0; i < uv_lanes(32); i++) {
                assert(dst[i] == expected);
            }

            int32_t lexpected =
                (imm >= 32) ? 0 : (int32_t)((uint32_t)vals[k] << imm);
            v_i32 l = uv_shl_i32(uv_dup_i32(vals[k]), imm);
            int32_t ldst[uv_lanes(32)];
            uv_storeu_i32(ldst, l);
            for (int i = 0; i < uv_lanes(32); i++) {
                assert(ldst[i] == lexpected);
            }
        }
    }

    report("i32 Shift edge cases", true);
}

/**
 * uv_floor_* and uv_ceil_* had no coverage at all, including negative values
 * (which is where the SSE2 truncation-based fallbacks can go wrong).
 */
void test_floor_ceil() {
#if UV_REGISTERS > 1   // the header defines no scalar uv_floor_*/uv_ceil_*
    static const float fvals[] = {
        1.5f, -1.5f, 2.999f, -2.999f, 0.5f, -0.5f,
        42.0f, -42.0f, 0.0f, 1e6f, -1e6f, 1234.0001f, -1234.0001f
    };
    const int fn = (int)(sizeof(fvals) / sizeof(fvals[0]));
    for (int k = 0; k < fn; k++) {
        float fdst[uv_lanes(32)];
        uv_storeu_f32(fdst, uv_floor_f32(uv_dup_f32(fvals[k])));
        for (int i = 0; i < uv_lanes(32); i++) {
            assert(fdst[i] == floorf(fvals[k]));
        }
        uv_storeu_f32(fdst, uv_ceil_f32(uv_dup_f32(fvals[k])));
        for (int i = 0; i < uv_lanes(32); i++) {
            assert(fdst[i] == ceilf(fvals[k]));
        }
    }

    static const double dvals[] = {
        1.5, -1.5, 2.999, -2.999, 0.5, -0.5,
        42.0, -42.0, 0.0, 1e6, -1e6, 1234.0001, -1234.0001
    };
    const int dn = (int)(sizeof(dvals) / sizeof(dvals[0]));
    for (int k = 0; k < dn; k++) {
        double ddst[uv_lanes(64)];
        uv_storeu_f64(ddst, uv_floor_f64(uv_dup_f64(dvals[k])));
        for (int i = 0; i < uv_lanes(64); i++) {
            assert(ddst[i] == floor(dvals[k]));
        }
        uv_storeu_f64(ddst, uv_ceil_f64(uv_dup_f64(dvals[k])));
        for (int i = 0; i < uv_lanes(64); i++) {
            assert(ddst[i] == ceil(dvals[k]));
        }
    }

    report("floor/ceil f32/f64", true);
#else
    report("floor/ceil f32/f64 (skipped: undefined at scalar level)", true);
#endif
}

/**
 * Regression: uv_not_f32/f64 must be a bitwise complement. The old
 * andnot(v, cmpeq(v,v)) construction returned 0 for NaN inputs.
 */
void test_not_nan() {
    // Expect the bitwise complement of the input pattern. (Note: ~NAN is
    // 0x803FFFFF, not 0xFFFFFFFF; the old andnot(v, cmpeq(v,v)) bug is
    // caught because it produced 0 for NaN inputs.)
    float fnan = NAN;
    uint32_t fnan_bits;
    memcpy(&fnan_bits, &fnan, 4);
    float fdst[uv_lanes(32)];
    uv_storeu_f32(fdst, uv_not_f32(uv_dup_f32(fnan)));
    for (int i = 0; i < uv_lanes(32); i++) {
        uint32_t bits;
        memcpy(&bits, &fdst[i], 4);
        assert(bits == ~fnan_bits);
    }

    double dnan = NAN;
    uint64_t dnan_bits;
    memcpy(&dnan_bits, &dnan, 8);
    double ddst[uv_lanes(64)];
    uv_storeu_f64(ddst, uv_not_f64(uv_dup_f64(dnan)));
    for (int i = 0; i < uv_lanes(64); i++) {
        uint64_t bits;
        memcpy(&bits, &ddst[i], 8);
        assert(bits == ~dnan_bits);
    }

    // And a normal value must still be complemented exactly.
    float one = 1.0f;
    uint32_t obits, rbits;
    memcpy(&obits, &one, 4);
    uv_storeu_f32(fdst, uv_not_f32(uv_dup_f32(one)));
    memcpy(&rbits, &fdst[0], 4);
    assert(rbits == ~obits);

    report("not() with NaN", true);
}

/**
 * Regression: the scalar uv_cvt_i32_f32 was bit-casting the integer *before*
 * subtracting the magic constant, producing garbage. Verify both conversion
 * directions over a range that all ISA paths must handle.
 */
void test_cvt_roundtrip() {
    // f32 <-> i32, well within the magic-number range (|v| < 2^22).
    static const int32_t ivals[] = {
        -1000000, -8388608/2, -3, -2, -1, 0, 1, 2, 3, 8388608/2, 1000000
    };
    const int n = (int)(sizeof(ivals) / sizeof(ivals[0]));
    for (int k = 0; k < n; k++) {
        float fdst[uv_lanes(32)];
        uv_storeu_f32(fdst, uv_cvt_i32_f32(uv_dup_i32(ivals[k])));
        for (int i = 0; i < uv_lanes(32); i++) {
            assert(fdst[i] == (float)ivals[k]);
        }
    }
    // f32 -> i32 rounds to nearest (current rounding mode), like lrintf.
    static const float fvals[] = {
        -1000000.5f, -2.5f, -1.5f, -0.5f, 0.5f, 1.5f, 2.5f, 1000000.5f,
        -3.7f, 3.7f, 42.0f, -42.0f
    };
    const int fn = (int)(sizeof(fvals) / sizeof(fvals[0]));
    for (int k = 0; k < fn; k++) {
        int32_t idst[uv_lanes(32)];
        uv_storeu_i32(idst, uv_cvt_f32_i32(uv_dup_f32(fvals[k])));
        for (int i = 0; i < uv_lanes(32); i++) {
            assert(idst[i] == (int32_t)lrintf(fvals[k]));
        }
    }

    // f64 <-> i64 within the exactly-representable range.
    static const int64_t jvals[] = {
        0, 1, -1, 1LL << 40, -(1LL << 40), 1LL << 52, -(1LL << 52),
        1234567890123ll, -987654321098ll
    };
    const int jn = (int)(sizeof(jvals) / sizeof(jvals[0]));
    for (int k = 0; k < jn; k++) {
        double ddst[uv_lanes(64)];
        uv_storeu_f64(ddst, uv_cvt_i64_f64(uv_dup_i64(jvals[k])));
        for (int i = 0; i < uv_lanes(64); i++) {
            assert(ddst[i] == (double)jvals[k]);
        }
    }
    static const double dvals[] = {
        -1000000.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 1000000.5, 42.0, -42.0
    };
    const int dn = (int)(sizeof(dvals) / sizeof(dvals[0]));
    for (int k = 0; k < dn; k++) {
        int64_t jdst[uv_lanes(64)];
        uv_storeu_i64(jdst, uv_cvt_f64_i64(uv_dup_f64(dvals[k])));
        for (int i = 0; i < uv_lanes(64); i++) {
            assert(jdst[i] == (int64_t)dvals[k]);   // truncation
        }
    }

    report("cvt round trips", true);
}

/**
 * Regression: uv_mul_i64 (mullo) with operands that exercise the cross-term
 * path (bits above 32). Reference computed with unsigned wrap-around to
 * avoid signed overflow UB in the test itself.
 */
void test_mul_i64_large() {
    static const int64_t avals[] = {
        0x123456789ll, -0x987654321ll, (1LL << 33) + 1, (1LL << 40),
        INT32_MAX + 1ll, -1, 0, 1
    };
    static const int64_t bvals[] = {
        0x9ABCDEF01ll, 0xABCDEF123ll, 3, -5, 0x111111111ll, 0x7FFFFFFFll, 7, -9
    };
    const int n = (int)(sizeof(avals) / sizeof(avals[0]));
    for (int k = 0; k < n; k++) {
        int64_t expected = (int64_t)((uint64_t)avals[k] * (uint64_t)bvals[k]);
        v_i64 r = uv_mul_i64(uv_dup_i64(avals[k]), uv_dup_i64(bvals[k]));
        int64_t dst[uv_lanes(64)];
        uv_storeu_i64(dst, r);
        for (int i = 0; i < uv_lanes(64); i++) {
            assert(dst[i] == expected);
        }
    }

    report("i64 Mul large operands", true);
}

/**
 * uv_min/uv_max on i64 with negative operands (earlier tests used
 * non-negative data only, which cannot catch sign-handling bugs).
 */
void test_minmax_i64_negative() {
    static const int64_t vals[] = {
        -5, 3, INT64_MIN + 1, INT64_MAX, -1, 0, 7, -7
    };
    const int n = (int)(sizeof(vals) / sizeof(vals[0]));
    for (int k = 0; k < n; k++) {
        for (int j = 0; j < n; j++) {
            v_i64 mn = uv_min_i64(uv_dup_i64(vals[k]), uv_dup_i64(vals[j]));
            v_i64 mx = uv_max_i64(uv_dup_i64(vals[k]), uv_dup_i64(vals[j]));
            int64_t dmn[uv_lanes(64)], dmx[uv_lanes(64)];
            uv_storeu_i64(dmn, mn);
            uv_storeu_i64(dmx, mx);
            for (int i = 0; i < uv_lanes(64); i++) {
                assert(dmn[i] == _MIN(vals[k], vals[j]));
                assert(dmx[i] == _MAX(vals[k], vals[j]));
            }
        }
    }

    report("i64 Min/Max negatives", true);
}

/**
 * Aligned load/store paths (uv_load_ and uv_store_) with properly aligned
 * buffers, since only the unaligned variants were covered.
 */
void test_aligned_access() {
    alignas(uv_alignment()) float af[uv_lanes(32)];
    alignas(uv_alignment()) double ad[uv_lanes(64)];
    // Destinations must be aligned too: uv_store_* are aligned stores on
    // SSE2/AVX2 (storing to an unaligned array would fault).
    alignas(uv_alignment()) float fdst[uv_lanes(32)];
    alignas(uv_alignment()) double ddst[uv_lanes(64)];
    for (int i = 0; i < uv_lanes(32); i++) af[i] = (float)i + 0.25f;
    for (int i = 0; i < uv_lanes(64); i++) ad[i] = (double)i + 0.5;

    uv_store_f32(fdst, uv_load_f32(af));
    for (int i = 0; i < uv_lanes(32); i++) assert(fdst[i] == af[i]);

    uv_store_f64(ddst, uv_load_f64(ad));
    for (int i = 0; i < uv_lanes(64); i++) assert(ddst[i] == ad[i]);

    report("Aligned load/store", true);
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
    test_i64_comparison();
    test_i64_logic();

    test_i32_arithmetic();
    test_i32_comparison();
    test_i32_logic();

    test_i8_complex();
    test_i8_compress_select();
    test_reductions();
    test_bit_ops();

    // Regression tests for issues found in review
    test_testz_semantics();
    test_abs_i64_edge();
    test_shr_i64_edges();
    test_shift_i32_edges();
    test_floor_ceil();
    test_not_nan();
    test_cvt_roundtrip();
    test_mul_i64_large();
    test_minmax_i64_negative();
    test_aligned_access();

    printf("\nAll tests passed successfully.\n");
    return 0;
}
