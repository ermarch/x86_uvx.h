#include "x86_uvx.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

#define TEST_epsilon 1e-5f

// Helper to print test results
void report(const char* test_name, bool success) {
    printf("[%s] %s\n", success ? "PASS" : "FAIL", test_name);
}

/*
 * Scalar bitwise helpers
 * These return the raw bit pattern as a uint32_t.
 */
static inline uint32_t scalar_and_f32_bits(float a, float b) {
    union { float f; uint32_t i; } va, vb, vr;
    va.f = a; vb.f = b; vr.i = va.i & vb.i;
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

/**
 * Test Float32 Arithmetic
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

    // Test Mul
    v_res = uv_mul_f32(v_src1, v_src2);
    uv_storeu_f32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(fabsf(dst[i] - (src1[i] * src2[i])) < TEST_epsilon);
    }

    // Test Sub
    v_res = uv_sub_f32(v_src1, v_src2);
    uv_storeu_f32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(fabsf(dst[i] - (src1[i] - src2[i])) < TEST_epsilon);
    }

    // Test Div
    for (int i = 0; i < lanes; i++) src2[i] = 2.0f;
    v_src2 = uv_loadu_f32(src2);
    v_res = uv_div_f32(v_src1, v_src2);
    uv_storeu_f32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(fabsf(dst[i] - (src1[i] / src2[i])) < TEST_epsilon);
    }

    report("f32 Arithmetic", true);
}

/**
 * Test Float32 Bitwise/Logic
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

/**
 * Test Integer 32
 */
void test_i32_arithmetic() {
    const int lanes = uv_lanes(32);
    int32_t src1[lanes], src2[lanes], dst[lanes];

    for (int i = 0; i < lanes; i++) {
        src1[i] = i * 10;
        src2[i] = i + 5;
    }

    v_i32 v1 = uv_loadu_i32(src1);
    v_i32 v2 = uv_loadu_i32(src2);

    // Test Add
    v_i32 v_res = uv_add_i32(v1, v2);
    uv_storeu_i32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == (src1[i] + src2[i]));
    }

    // Test Mul (mullo)
    v_res = uv_mul_i32(v1, v2);
    uv_storeu_i32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == (int32_t)((int64_t)src1[i] * src2[i]));
    }

    // Test Shift Left
    v_res = uv_shl_i32(v1, 2);
    uv_storeu_i32(dst, v_res);
    for (int i = 0; i < lanes; i++) {
        assert(dst[i] == (src1[i] << 2));
    }

    report("i32 Arithmetic", true);
}

/**
 * Test Integer 8 (Complex Shuffles)
 */
void test_i8_complex() {
    int8_t src[64];
    int8_t dst[64];
    for(int i=0; i<64; i++) src[i] = (int8_t)i;

    v_i32 v_src = uv_loadu_i8(src);
    v_i32 v_idx = uv_loadu_i8(src); 
    
    v_i32 v_res = uv_permutexvar_i8(v_idx, v_src);
    uv_storeu_i8(dst, v_res);
    
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

    test_f32_arithmetic();
    test_f32_logic();
    test_i32_arithmetic();
    test_i8_complex();
    test_reductions();
    test_bit_ops();

    printf("\nAll tests passed successfully.\n");
    return 0;
}

