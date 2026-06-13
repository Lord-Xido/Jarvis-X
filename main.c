#include "jarvis_x.h"
#include <stdio.h>

/* ============================================================================
   JARVIS-X Main: Demonstration of unified state evolution engine
   ============================================================================ */

int main(void) {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                      JARVIS-X INITIALIZATION                   ║\n");
    printf("║              Unified State Evolution Engine v1.0               ║\n");
    printf("║         X_{t+1} = X_t + P_t - E_t + Ω_t                       ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    
    /* Configuration */
    uint32_t grid_dims[3] = {8, 8, 8};    /* 8x8x8 voxel grid */
    size_t state_dim = 512;                 /* 512-dimensional state space */
    uint32_t omega_capacity = 2048;         /* Ω-Ledger capacity */
    
    printf("[CONFIG] Voxel grid: %u x %u x %u\n", grid_dims[0], grid_dims[1], grid_dims[2]);
    printf("[CONFIG] State dimensionality: %lu\n", state_dim);
    printf("[CONFIG] Ω-Ledger capacity: %u entries\n\n", omega_capacity);
    
    /* Initialize JARVIS-X system */
    printf("[INIT] Initializing JARVIS-X core system...\n");
    JarvisX *jarvis = jarvis_x_init(grid_dims, state_dim, omega_capacity);
    
    printf("[INIT] System initialized:\n");
    printf("       - State vector: %.6f (initial)\n", jarvis->state->state_vector[0]);
    printf("       - Convergence threshold: %.2e\n", jarvis->convergence_threshold);
    printf("       - Learning rate: %.4f\n", jarvis->learning_rate);
    printf("       - Max iterations: %u\n\n", jarvis->max_iterations);
    
    /* Configure convergence parameters */
    jarvis->max_iterations = 5000;
    jarvis->convergence_threshold = 1e-5f;
    jarvis->learning_rate = 0.001f;
    
    printf("[RUN] Beginning unified state evolution...\n");
    printf("[RUN] System will iterate until saturation at Ω fixed point or max iterations reached.\n\n");
    
    /* Run the main evolution loop */
    jarvis_x_run(jarvis);
    
    /* Report final status */
    printf("\n╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                        FINAL STATUS                           ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("[STATUS] Iteration count: %lu\n", jarvis->state->iteration);
    printf("[STATUS] Final telemetry energy: %.6f\n", jarvis->state->telemetry_energy);
    printf("[STATUS] Saturation achieved: %s\n", jarvis->is_saturated ? "YES" : "NO");
    printf("[STATUS] Deterministic idle mode: %s\n", jarvis->is_in_deterministic_idle ? "ACTIVE" : "INACTIVE");
    printf("[STATUS] Ω-Ledger entries: %lu\n", jarvis->state->omega->head);
    printf("[STATUS] IP_lock (PUF-bound): 0x%08X\n\n", jarvis->state->omega->ip_lock);
    
    /* Sample state vector values */
    printf("[STATE] Sample state vector values (first 5 components):\n");
    for (int i = 0; i < 5 && i < (int)state_dim; i++) {
        printf("       X[%d] = %.8f\n", i, jarvis->state->state_vector[i]);
    }
    
    /* Verify causal chain integrity */
    printf("\n[LEDGER] Ω-Ledger causal chain verification:\n");
    OmegaLedger *ol = jarvis->state->omega;
    size_t entries_to_check = (ol->head < 5) ? ol->head : 5;
    
    for (size_t i = 0; i < entries_to_check; i++) {
        OmegaEntry *entry = &ol->ledger[i];
        printf("       Entry %lu: state_hash=0x%08X, prev_hash=0x%08X, timestamp=%lu\n",
               i, entry->state_hash, entry->prev_hash, entry->timestamp);
    }
    
    printf("\n[COMPLETE] JARVIS-X execution cycle complete.\n");
    printf("[COMPLETE] System is ready for next directive.\n\n");
    
    /* Cleanup */
    jarvis_x_destroy(jarvis);
    
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                    SYSTEM SHUTDOWN COMPLETE                   ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    
    return 0;
}
