#include "volumetric_fold.h"

#include <stddef.h>

static int validate_args(
    const uint64_t *src,
    uint64_t *dst,
    uint64_t *entropy,
    uint64_t total_words)
{
    if (src == NULL || dst == NULL || entropy == NULL) {
        return -1;
    }
    if ((total_words & UINT64_C(7)) != 0) {
        return -1;
    }
    if (src == dst || src == entropy || dst == entropy) {
        return -1;
    }
    return 0;
}

int jx_execute_3d_volumetric_fold_scalar(
    const uint64_t *src,
    uint64_t *dst,
    uint64_t *entropy,
    uint64_t total_words)
{
    uint64_t block;

    if (validate_args(src, dst, entropy, total_words) != 0) {
        return -1;
    }

    for (block = 0; block < total_words; block += UINT64_C(8)) {
        const uint64_t mirror_base = total_words - UINT64_C(8) - block;
        uint64_t lane;

        for (lane = 0; lane < UINT64_C(8); ++lane) {
            const uint64_t i = block + lane;
            const uint64_t m = mirror_base + lane;
            const uint64_t next = (~src[i]) | src[m];
            dst[i] = next;
            entropy[i] = next ^ src[i];
        }
    }

    return 0;
}

static int have_avx512_fold_isa(void)
{
#if (defined(__x86_64__) || defined(_M_X64)) &&     (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512dq");
#else
    return 0;
#endif
}

int jx_execute_3d_volumetric_fold(
    const uint64_t *src,
    uint64_t *dst,
    uint64_t *entropy,
    uint64_t total_words)
{
    if (validate_args(src, dst, entropy, total_words) != 0) {
        return -1;
    }

    if (have_avx512_fold_isa()) {
        return jx_execute_3d_volumetric_fold_avx512(
            src, dst, entropy, total_words);
    }

    return jx_execute_3d_volumetric_fold_scalar(
        src, dst, entropy, total_words);
}
