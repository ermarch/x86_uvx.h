cmake_minimum_required(VERSION 3.15...3.29)
project(x86_uvx.h)

FUNCTION(CREATE_UV_TEST TEST_NAME EXT NOFLAGS)
  set(EXEC_NAME "${TEST_NAME}-${EXT}")
  add_executable(${EXEC_NAME} ${TEST_NAME}.c)
  separate_arguments(FLAG_LIST NATIVE_COMMAND "${NOFLAGS}")
  target_link_libraries(${EXEC_NAME} "-lm")
  target_compile_options(${EXEC_NAME} PRIVATE
    ${FLAG_LIST}
  )
ENDFUNCTION()

CREATE_UV_TEST(test_x86_uvx scalar "-mno-sse2")
CREATE_UV_TEST(test_x86_uvx sse2 "-msse2 -mno-avx -mno-fma")
CREATE_UV_TEST(test_x86_uvx ssse3 "-mssse3 -mno-avx -mno-fma")
CREATE_UV_TEST(test_x86_uvx sse4.1 "-msse4.1 -mno-avx -mno-fma")
CREATE_UV_TEST(test_x86_uvx avx2 "-mavx2 -mno-fma -mno-avx512f")
CREATE_UV_TEST(test_x86_uvx fma "-mavx2 -mfma -mno-avx512f")
CREATE_UV_TEST(test_x86_uvx avx512f "-mavx512f")
CREATE_UV_TEST(test_x86_uvx avx512-all "-mavx512f -mavx512dq -mavx512bw -mavx512vbmi2")
CREATE_UV_TEST(test_x86_uvx avx10.1-evex512 "-mavx10.1 -mprefer-vector-width=512")
CREATE_UV_TEST(test_x86_uvx avx10.1-evex256 "-mavx10.1 -mprefer-vector-width=256")
