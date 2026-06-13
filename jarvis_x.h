#ifndef JARVIS_X_H
#define JARVIS_X_H

#include <stdint.h>
#include <stddef.h>
#include <time.h>

/* ============================================================================
   JARVIS-X: Unified State Evolution Engine
   X_{t+1} = X_t + P_t - E_t + Ω_t
   ============================================================================ */

/* Bit-State Form: B_i = (a_i, m_i, p_i, r_i) */
typedef struct {
    uint8_t activation;      /* a_i: activation state */
    uint8_t memory;          /* m_i: memory state */
    uint8_t prediction;      /* p_i: prediction state */
    uint8_t reconstruction;  /* r_i: reconstruction state */
} BitState;

/* Voxel: 3D spatial bit aggregation */
typedef struct {
    BitState bits[8];        /* 8 sub-bits per voxel (2x2x2) */
    uint32_t hash;           /* Fractal hash anchor */
    float telemetry_energy;  /* Distance from fixed point */
} Voxel;

/* 3D Voxel Grid */
typedef struct {
    Voxel ***grid;           /* 3D array of voxels */
    uint32_t dims[3];        /* X, Y, Z dimensions */
    uint64_t total_voxels;
} VoxelGrid;

/* Episodic Memory: Ω-Ledger (tamper-evident trace) */
typedef struct {
    uint64_t timestamp;
    uint32_t state_hash;
    uint32_t prediction_hash;
    uint32_t error_hash;
    uint32_t prev_hash;      /* Link to previous entry (chain) */
} OmegaEntry;

typedef struct {
    OmegaEntry *ledger;
    size_t capacity;
    size_t head;             /* Current write position */
    uint32_t ip_lock;        /* IP_lock: recursive hash anchor (PUF-bound) */
} OmegaLedger;

/* Global State Vector: X_t */
typedef struct {
    VoxelGrid *voxels;
    OmegaLedger *omega;
    float *state_vector;     /* Flattened state */
    size_t state_dim;
    uint64_t iteration;
    float telemetry_energy;  /* System-wide telemetry */
} SystemState;

/* Prediction Engine: P_t */
typedef struct {
    float **weights;         /* Prediction weights */
    size_t input_dim;
    size_t output_dim;
    float (*forward)(struct PredictionEngine *self, float *x, size_t len);
} PredictionEngine;

/* Error Accumulator: E_t = Z ⊕ Ẑ */
typedef struct {
    float *error_vector;
    size_t dim;
    float accumulated_error;
    float (*xor_error)(struct ErrorAccumulator *self, float *predicted, float *actual, size_t len);
} ErrorAccumulator;

/* JARVIS-X Core System */
typedef struct {
    SystemState *state;
    PredictionEngine *predictor;
    ErrorAccumulator *error;
    
    /* Π_Λ Projection (Green Outer Cage): law constraint */
    float (*lambda_projection)(struct JarvisX *self, float *x, size_t len);
    
    /* Configuration */
    uint32_t max_iterations;
    float convergence_threshold;
    float learning_rate;
    
    /* Status */
    int is_saturated;        /* At Ω fixed point? */
    int is_in_deterministic_idle;
} JarvisX;

/* ============================================================================
   API Functions
   ============================================================================ */

/* Initialize JARVIS-X system */
JarvisX* jarvis_x_init(uint32_t grid_dims[3], size_t state_dim, uint32_t omega_capacity);

/* Execute single unified state update: X_{t+1} = X_t + P_t - E_t + Ω_t */
void jarvis_x_step(JarvisX *sys);

/* Fractal voxel aggregation */
void jarvis_x_fractal_aggregate(JarvisX *sys, uint32_t depth);

/* Compress voxel grid through fractal recursion */
void jarvis_x_compress(JarvisX *sys);

/* Record state to Ω-Ledger (tamper-evident) */
void jarvis_x_ledger_record(JarvisX *sys);

/* Apply Π_Λ projection (law constraint) */
void jarvis_x_apply_lambda_constraint(JarvisX *sys);

/* Check for saturation at Ω fixed point */
int jarvis_x_check_saturation(JarvisX *sys);

/* Main loop: iterate until saturation or max_iterations */
void jarvis_x_run(JarvisX *sys);

/* Cleanup */
void jarvis_x_destroy(JarvisX *sys);

/* Utility: Hash state for ledger */
uint32_t jarvis_x_hash_state(float *state, size_t len);

#endif /* JARVIS_X_H */
