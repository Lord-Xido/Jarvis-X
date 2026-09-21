.intel_syntax noprefix
.text

/*
 * GNU/Clang preprocessed assembly (.S), x86-64.
 *
 * System V AMD64:
 *   rdi = src, rsi = dst, rdx = entropy, rcx = total_words
 *
 * Windows x64:
 *   rcx = src, rdx = dst, r8 = entropy, r9 = total_words
 *
 * The implementation intentionally uses unaligned 512-bit loads/stores so the
 * API does not make alignment a correctness requirement. total_words must be
 * a multiple of eight uint64_t values (one 512-bit block).
 */

#ifdef _WIN32
#define ARG_SRC   rcx
#define ARG_DST   rdx
#define ARG_ENT   r8
#define ARG_WORDS r9
#else
#define ARG_SRC   rdi
#define ARG_DST   rsi
#define ARG_ENT   rdx
#define ARG_WORDS rcx
#endif

.global jx_execute_3d_volumetric_fold_avx512
#ifndef _WIN32
.type jx_execute_3d_volumetric_fold_avx512, @function
#endif
jx_execute_3d_volumetric_fold_avx512:
    test ARG_SRC, ARG_SRC
    jz .L_invalid
    test ARG_DST, ARG_DST
    jz .L_invalid
    test ARG_ENT, ARG_ENT
    jz .L_invalid
    test ARG_WORDS, ARG_WORDS
    jz .L_ok
    test ARG_WORDS, 7
    jnz .L_invalid

    xor r10, r10
    mov r11, ARG_WORDS
    sub r11, 8
    vpxord zmm3, zmm3, zmm3

.p2align 6
.L_loop:
    vmovdqu64 zmm0, ZMMWORD PTR [ARG_SRC + r10*8]
    vmovdqu64 zmm1, ZMMWORD PTR [ARG_SRC + r11*8]
    vmovdqa64 zmm2, zmm0

    /*
     * Truth table 0xCF implements (~A) | B for:
     *   A = old destination zmm0 (src block)
     *   B = zmm1 (mirrored block)
     *   C = zmm3 (zero, ignored by the function)
     *
     * This is algebraically identical to:
     *   (src & mirror) ^ (~src)
     */
    vpternlogq zmm0, zmm1, zmm3, 0xCF

    vpxorq zmm2, zmm2, zmm0
    vmovdqu64 ZMMWORD PTR [ARG_DST + r10*8], zmm0
    vmovdqu64 ZMMWORD PTR [ARG_ENT + r10*8], zmm2

    add r10, 8
    sub r11, 8
    cmp r10, ARG_WORDS
    jb .L_loop

.L_ok:
    vzeroupper
    xor eax, eax
    ret

.L_invalid:
    mov eax, -1
    ret

#ifndef _WIN32
.size jx_execute_3d_volumetric_fold_avx512, .-jx_execute_3d_volumetric_fold_avx512
.section .note.GNU-stack,"",@progbits
#endif
