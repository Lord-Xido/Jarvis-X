#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Native CPU lattice engine. 0=healthy, 1=faulted.
typedef struct JXStats {
    uint64_t ticks;
    uint32_t dimension, voxels, active_faults, repaired_faults, total_faults;
    uint32_t routes_found, routes_failed, last_route_expansions;
    double last_step_ms, total_step_ms, last_codec_ratio, last_max_delta;
    uint32_t last_encoded_bytes;
    uint32_t codec_verified;
    uint64_t switching_bits_total, last_switching_bits, byte_updates_total;
    uint64_t ann_epochs;
    double ann_initial_mse, ann_last_mse, last_ann_ratio;
    uint32_t last_ann_encoded_bytes;
} JXStats;
void* jx_create(uint32_t dimension, uint64_t seed);
void  jx_destroy(void* engine);
int   jx_reset(void* engine, uint64_t seed);
int   jx_step(void* engine, uint32_t iterations);
int   jx_fail(void* engine, uint32_t index);
int   jx_route(void* engine, uint32_t from, uint32_t to, uint32_t* out_path, uint32_t capacity);
int   jx_copy_field(void* engine, uint8_t* out_values, uint8_t* out_faults, uint32_t capacity);
int   jx_set_field(void* engine, const uint8_t* values, uint32_t length);
int   jx_stats(void* engine, JXStats* out);
uint32_t jx_codec_bound(uint32_t raw_length);
int   jx_encode(const uint8_t* data, uint32_t length, uint8_t* out, uint32_t capacity);
int   jx_decode(const uint8_t* encoded, uint32_t length, uint8_t* out, uint32_t capacity);
int   jx_encode_field(void* engine, uint8_t* out, uint32_t capacity);
int   jx_import_field(void* engine, const uint8_t* encoded, uint32_t length);
int   jx_verify_field(void* engine);
// Electrical-logic observability: 8 packed bitplanes (LSB to MSB) of actual Q8 memory.
uint32_t jx_bitplane_bound(uint32_t voxels);
int   jx_bitplanes(void* engine, uint8_t* out, uint32_t capacity);
// Shared neural 8->4->8 tile predictor. Lossy preview; residual path is byte-exact.
int   jx_ann_train(void* engine, uint32_t epochs, double learning_rate);
int   jx_ann_preview(void* engine, uint8_t* out_raster, uint32_t capacity);
int   jx_ann_latents(void* engine, float* out, uint32_t capacity);
uint32_t jx_ann_bound(uint32_t voxels);
int   jx_ann_export(void* engine, uint8_t* out, uint32_t capacity);
int   jx_ann_import(void* engine, const uint8_t* stream, uint32_t length);

#ifdef __cplusplus
}
#endif
