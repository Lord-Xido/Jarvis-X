#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define JX_8192_CUBE_BITS  UINT64_C(549755813888)
#define JX_8192_CUBE_BYTES UINT64_C(68719476736)
#define JX_8192_CUBE_WORDS UINT64_C(8589934592)

/*
 * All functions implement the same 512-bit blockwise mirror transform:
 *
 *   next = (src & mirror) ^ (~src) = (~src) | mirror
 *   entropy = next ^ src
 *
 * total_words must be a multiple of 8. Buffers must not overlap.
 * The public dispatcher selects AVX-512F+DQ when available and otherwise
 * uses the scalar reference implementation.
 */
int jx_execute_3d_volumetric_fold(
    const uint64_t *src,
    uint64_t *dst,
    uint64_t *entropy,
    uint64_t total_words);

int jx_execute_3d_volumetric_fold_scalar(
    const uint64_t *src,
    uint64_t *dst,
    uint64_t *entropy,
    uint64_t total_words);

int jx_execute_3d_volumetric_fold_avx512(
    const uint64_t *src,
    uint64_t *dst,
    uint64_t *entropy,
    uint64_t total_words);

#ifdef __cplusplus
}
#endif
