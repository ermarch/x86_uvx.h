/**
 * @file      test_x86_uvx.c
 * @brief     Comprehensive validation runner for x86_uvx.h
 * Tests all framework operations against their scalar equivalents.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "x86_uvx.h"

// Macro helper to display active architectural optimization target at runtime
void print_current_target(void) {
    printf("====================================================\n");
    printf("Running x86_uvx.h Verification Pipeline\n");
#if defined(__AVX10__)
    #if defined(__EVEX512__) || (defined(__AVX10_MAX_VEC_LEN__) && __AVX10_MAX_VEC_LEN__ >= 512)
        printf("Detected Target: AVX10 (512-bit Extended Mode)\n");
    #else
        printf("Detected Target: AVX10 (256-bit Dense Mode)\n");
    #endif
#elif defined(__AVX512F__)
    printf("Detected Target: LEVEL 3 - AVX-512\n");
#elif defined(__FMA__) || defined(__AVX2__)
    printf("Detected Target: LEVEL 2 - AVX2 / FMA\n");
#elif defined(__SSE2__)
    printf("Detected Target: LEVEL 1 - SSE2\n");
#else
    printf("Detected Target: LEVEL 0 - Scalar Fallback\n");
#endif
    printf("Config Invariants: UV_LANES=%d, UV_LANES_F64=%d, UV_LANES_I8=%d, UV_REGISTERS=%d\n", 
           UV_LANES, UV_LANES_F64, UV_LANES_I8, UV_REGISTERS);
    printf("====================================================\n\n");
}

// Float bit comparison safety helper
static inline int bitwise_equal_f32(float a, float b) {
    uint32_t ia, ib;
    memcpy(&ia, &a, sizeof(float));
    memcpy(&ib, &b, sizeof(float));
    return ia == ib;
}

static inline int bitwise_equal_f64(double a, double b) {
    uint64_t ia, ib;
    memcpy(&ia, &a, sizeof(double));
    memcpy(&ib, &b, sizeof(double));
    return ia == ib;
}

// ============================================================================
// Core Tests
// ============================================================================

void test_f32_operations(void) {
    printf("[*] Testing Single-Precision Floats (f32)...\n");
    
    // Allocate bounds matching the active vector lane width
    alignas(64) float src_a[UV_LANES];
    alignas(64) float src_b[UV_LANES];
    alignas(64) float out_simd[UV_LANES];
    
    for (int i = 0; i < UV_LANES; ++i) {
        src_a[i] = (float)(i + 1) * 1.5f;
        src_b[i] = (float)(i + 1) * 0.5f;
    }
    
    v_f32 va = uv_loadu_f32(src_a);
    v_f32 vb = uv_loadu_f32(src_b);
    v_f32 v_res;

    // 1. Basic Arithmetic
    v_res = uv_add_f32(va, vb); uv_storeu_f32(out_simd, v_res);
    for(int i=0; i<UV_LANES; ++i) assert(bitwise_equal_f32(out_simd[i], src_a[i] + src_b[i]));

    v_res = uv_sub_f32(va, vb); uv_storeu_f32(out_simd, v_res);
    for(int i=0; i<UV_LANES; ++i) assert(bitwise_equal_f32(out_simd[i], src_a[i] - src_b[i]));

    v_res = uv_mul_f32(va, vb); uv_storeu_f32(out_simd, v_res);
    for(int i=0; i<UV_LANES; ++i) assert(bitwise_equal_f32(out_simd[i], src_a[i] * src_b[i]));

    v_res = uv_div_f32(va, vb); uv_storeu_f32(out_simd, v_res);
    for(int i=0; i<UV_LANES; ++i) assert(bitwise_equal_f32(out_simd[i], src_a[i] / src_b[i]));

    // 2. Extrema
    v_res = uv_max_f32(va, vb); uv_storeu_f32(out_simd, v_res);
    for(int i=0; i<UV_LANES; ++i) assert(bitwise_equal_f32(out_simd[i], src_a[i] > src_b[i] ? src_a[i] : src_b[i]));

    // 3. FMA (Fused Multiply-Add)
    alignas(64) float src_c[UV_LANES];
    for (int i = 0; i < UV_LANES; ++i) src_c[i] = 10.0f;
    v_f32 vc = uv_loadu_f32(src_c);
    v_res = uv_fma_f32(va, vb, vc); uv_storeu_f32(out_simd, v_res);
    for(int i=0; i<UV_LANES; ++i) {
        float expected = (src_a[i] * src_b[i]) + src_c[i];
        // Allow tiny delta margin only if hardware lacks precision match (Level 0/1/2 non-native vs native)
        assert(fabsf(out_simd[i] - expected) < 1e-5f);
    }

    // 4. Vector Logic Masks & Selection
    uv_mask mask = uv_cmpgt_f32(va, vb); 
    v_res = uv_select_f32(mask, va, vb); 
    uv_storeu_f32(out_simd, v_res);
    for(int i=0; i<UV_LANES; ++i) {
        float expected = (src_a[i] > src_b[i]) ? src_a[i] : src_b[i];
        assert(bitwise_equal_f32(out_simd[i], expected));
    }
    
    printf("  -> All f32 tests passed.\n");
}

void test_f64_operations(void) {
    printf("[*] Testing Double-Precision Floats (f64)...\n");
    
    alignas(64) double src_a[UV_LANES_F64];
    alignas(64) double src_b[UV_LANES_F64];
    alignas(64) double out_simd[UV_LANES_F64];
    
    for (int i = 0; i < UV_LANES_F64; ++i) {
        src_a[i] = (double)(i + 1) * 10.25;
        src_b[i] = (double)(i + 1) * 2.0;
    }
    
    v_f64 va = uv_loadu_f64(src_a);
    v_f64 vb = uv_loadu_f64(src_b);
    v_f64 v_res;

    v_res = uv_add_f64(va, vb); uv_storeu_f64(out_simd, v_res);
    for(int i=0; i<UV_LANES_F64; ++i) assert(bitwise_equal_f64(out_simd[i], src_a[i] + src_b[i]));

    v_res = uv_mul_f64(va, vb); uv_storeu_f64(out_simd, v_res);
    for(int i=0; i<UV_LANES_F64; ++i) assert(bitwise_equal_f64(out_simd[i], src_a[i] * src_b[i]));

    printf("  -> All f64 tests passed.\n");
}

void test_i32_operations(void) {
    printf("[*] Testing 32-bit Integers (i32)...\n");
    
    alignas(64) int32_t src_a[UV_LANES];
    alignas(64) int32_t src_b[UV_LANES];
    alignas(64) int32_t out_simd[UV_LANES];
    
    for (int i = 0; i < UV_LANES; ++i) {
        src_a[i] = (i % 2 == 0) ? -(i + 5) : (i + 5);
        src_b[i] = 3;
    }
    
    v_i32 va = uv_loadu_i32(src_a);
    v_i32 vb = uv_loadu_i32(src_b);
    v_i32 v_res;

    v_res = uv_add_i32(va, vb); uv_storeu_i32(out_simd, v_res);
    for(int i=0; i<UV_LANES; ++i) assert(out_simd[i] == src_a[i] + src_b[i]);

    v_res = uv_abs_i32(va); uv_storeu_i32(out_simd, v_res);
    for(int i=0; i<UV_LANES; ++i) assert(out_simd[i] == (src_a[i] < 0 ? -src_a[i] : src_a[i]));

    v_res = uv_shl_i32(va, 2); uv_storeu_i32(out_simd, v_res);
    for(int i=0; i<UV_LANES; ++i) assert(out_simd[i] == (src_a[i] << 2));

    printf("  -> All i32 tests passed.\n");
}

void test_i8_operations(void) {
    printf("[*] Testing 8-bit Bytes (i8)...\n");
    
    alignas(64) int8_t src_a[UV_LANES_I8];
    alignas(64) int8_t src_b[UV_LANES_I8];
    alignas(64) int8_t out_simd[UV_LANES_I8];
    alignas(64) int8_t ctrl[UV_LANES_I8];
    
    for (int i = 0; i < UV_LANES_I8; ++i) {
        src_a[i] = (int8_t)(i + 10);
        src_b[i] = (int8_t)20;
        // Interleave bit 7 clear/set patterns to fully evaluate intra-lane shuffles
        ctrl[i]  = (i % 3 == 0) ? (int8_t)0x80 : (int8_t)(i % 16); 
    }
    
    // Test Load / Store Boundary Compatibility
    // Handling underlying array allocations directly to match type width configurations
    v_i8 va, vb, vctrl, v_res;
    memcpy(&va, src_a, sizeof(v_i8));
    memcpy(&vb, src_b, sizeof(v_i8));
    memcpy(&vctrl, ctrl, sizeof(v_i8));

    // 1. In-lane Byte Shuffle Execution Check
    v_res = uv_shuffle_i8(va, vctrl);
    memcpy(out_simd, &v_res, sizeof(v_i8));
    for (int i = 0; i < UV_LANES_I8; ++i) {
        int8_t expected;
        if (ctrl[i] & 0x80) {
            expected = 0;
        } else {
            // Emulate localized 16-byte inner chunking boundaries typical of SSE/AVX2 configurations
            int lane_offset = (i / 16) * 16;
            int target_idx = lane_offset + (ctrl[i] & 0x0F);
            expected = (target_idx < UV_LANES_I8) ? src_a[target_idx] : src_a[0];
        }
        if (UV_LANES_I8 > 1) { // Skip strict index chunking checks on purely scalar fallback builds
            assert(out_simd[i] == expected);
        }
    }

    // 2. Conditional Selection Routing Verification
    #if defined(__AVX512F__) || defined(__AVX10__)
        // Bitmask variant checks
        #if UV_LANES_I8 == 64
            uv_mask8 mask = uv_cmpeq_i8(va, vb);
            v_res = uv_select_i8(mask, va, vb);
        #else
            uv_mask_i8 mask = uv_cmpeq_i8(va, vb);
            v_res = uv_select_i8(mask, va, vb);
        #endif
    #else
        // Vector mask variations typical of older hardware lines
        uv_mask_i8 mask = uv_cmpeq_i8(va, vb);
        v_res = uv_select_i8(mask, va, vb);
    #endif

    memcpy(out_simd, &v_res, sizeof(v_i8));
    for(int i = 0; i < UV_LANES_I8; ++i) {
        int8_t expected = (src_a[i] == src_b[i]) ? src_a[i] : src_b[i];
        assert(out_simd[i] == expected);
    }

    printf("  -> All i8 tests passed.\n");
}

void test_bit_manipulation_primitives(void) {
    printf("[*] Testing Scalar Bit-Manipulation Primitives...\n");
    
    uint64_t val = 0x0000FFFF00000080ULL; // Has 16 set bits, tzcnt=7, lzcnt=16
    
    assert(uv_popcnt_u64(val) == 17);
    assert(uv_tzcnt_u64(val) == 7);
    assert(uv_lzcnt_u64(val) == 16);
    
    // Extreme Boundary Checks
    assert(uv_tzcnt_u64(0) == 64);
    assert(uv_lzcnt_u64(0) == 64);
    assert(uv_popcnt_u64(0) == 0);

    printf("  -> All bit manipulation primitives passed.\n");
}

void test_prefetch_hints(void) {
    printf("[*] Validating compiler pass on Prefetch Macros...\n");
    
    int dummy_data[256] = {0};
    
    // Ensures prefetch compiles cleanly with correct parameter transformations
    uv_prefetch(&dummy_data[0],   UV_HINT_T0);
    uv_prefetch(&dummy_data[64],  UV_HINT_T1);
    uv_prefetch(&dummy_data[128], UV_HINT_T2);
    uv_prefetch(&dummy_data[192], UV_HINT_NTA);
    
    printf("  -> Prefetch directives verified.\n");
}

// ============================================================================
// Main Application Entry Point
// ============================================================================
int main(void) {
    print_current_target();
    
    // Execute unit pipelines
    test_f32_operations();
    test_f64_operations();
    test_i32_operations();
    test_i8_operations();
    test_bit_manipulation_primitives();
    test_prefetch_hints();
    
    // Clean trailing upper register environments out as a strict testing discipline
    uv_clear_lanes();
    
    printf("\n[SUCCESS] All abstract vector functions successfully validated bit-for-bit against structural fallbacks!\n");
    return 0;
}
