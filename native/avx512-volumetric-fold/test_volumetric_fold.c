#include "volumetric_fold.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

static uint64_t reference_word(uint64_t src, uint64_t mirror)
{
    return (~src) | mirror;
}

static int check_case(const uint64_t *src, uint64_t words)
{
    uint64_t scalar_dst[32] = {0};
    uint64_t scalar_entropy[32] = {0};
    uint64_t dispatch_dst[32] = {0};
    uint64_t dispatch_entropy[32] = {0};
    uint64_t block;

    if (words > UINT64_C(32)) {
        return 99;
    }

    if (jx_execute_3d_volumetric_fold_scalar(
            src, scalar_dst, scalar_entropy, words) != 0) {
        return 1;
    }

    if (jx_execute_3d_volumetric_fold(
            src, dispatch_dst, dispatch_entropy, words) != 0) {
        return 2;
    }

    for (block = 0; block < words; block += UINT64_C(8)) {
        const uint64_t mirror_base = words - UINT64_C(8) - block;
        uint64_t lane;

        for (lane = 0; lane < UINT64_C(8); ++lane) {
            const uint64_t i = block + lane;
            const uint64_t m = mirror_base + lane;
            const uint64_t expected = reference_word(src[i], src[m]);
            const uint64_t expected_entropy = expected ^ src[i];

            if (scalar_dst[i] != expected ||
                scalar_entropy[i] != expected_entropy ||
                dispatch_dst[i] != expected ||
                dispatch_entropy[i] != expected_entropy) {
                fprintf(stderr,
                    "mismatch i=%" PRIu64 " mirror=%" PRIu64
                    " got=%016" PRIx64 " expected=%016" PRIx64 "\n",
                    i, m, dispatch_dst[i], expected);
                return 3;
            }
        }
    }

    return 0;
}

int main(void)
{
    uint64_t patterned[16];
    uint64_t zeros[8] = {0};
    uint64_t i;
    int rc;

    for (i = 0; i < UINT64_C(16); ++i) {
        patterned[i] =
            (UINT64_C(0x0102030405060708) * (i + UINT64_C(1))) ^
            (UINT64_C(0xF0F0F0F0F0F0F0F0) >> (i & UINT64_C(7)));
    }

    rc = check_case(patterned, UINT64_C(16));
    if (rc != 0) {
        return rc;
    }

    rc = check_case(zeros, UINT64_C(8));
    if (rc != 0) {
        return rc;
    }

    {
        uint64_t dst[8] = {0};
        uint64_t entropy[8] = {0};
        if (jx_execute_3d_volumetric_fold_scalar(
                zeros, dst, entropy, UINT64_C(8)) != 0) {
            return 4;
        }
        for (i = 0; i < UINT64_C(8); ++i) {
            if (dst[i] != UINT64_MAX || entropy[i] != UINT64_MAX) {
                return 5;
            }
        }
    }

    if (jx_execute_3d_volumetric_fold_scalar(
            patterned, patterned + 1, patterned + 2, UINT64_C(7)) != -1) {
        return 6;
    }

    puts("volumetric-fold: ok");
    return 0;
}
