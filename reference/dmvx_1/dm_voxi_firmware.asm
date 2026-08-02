; ==============================================================================
; DM-vOmegaXi+ TRANSACTIONAL AUTO-ENCODING ROM FIRMWARE
; Architecture : Psi-Phi-Lambda-Omega-Theta stack
; Status       : Normative source for the bounded reference ISA in ISA.md
; Capability   : Propose -> validate -> stage -> verify -> atomically commit
; Non-claim    : Virtual manifold size is symbolic; physical work is budgeted.
; ==============================================================================

.ARCH       DMVX-1
.ENDIAN     LITTLE
.WORD       32
.ADDR       32
.STACK      0x00008000, 0x00001000
.ENTRY      BOOT

; ------------------------------------------------------------------------------
; Numeric and execution constants
; Q16.16 values are encoded as integer(value * 65536).
; ------------------------------------------------------------------------------
.EQU STATUS_OK,                 0x00000000
.EQU STATUS_READY,              0x00000001
.EQU STATUS_COMMITTED,          0x00000002
.EQU STATUS_REJECTED,           0x00000003
.EQU STATUS_DRIFT,              0x00000101
.EQU STATUS_VERIFY_FAILED,      0x00000102
.EQU STATUS_BOUNDS_FAILED,      0x00000103
.EQU STATUS_BUDGET_FAILED,      0x00000104
.EQU STATUS_NONFINITE,          0x00000105
.EQU STATUS_STACK_FAULT,        0x00000106
.EQU STATUS_ROM_FAULT,          0x00000107

.EQU BUS_EMPTY,                 0x00000000
.EQU MAX_RETRIES,               0x00000004
.EQU MAX_ACTIVE_PAGES,          0x00000100
.EQU FREE_ENERGY_LIMIT_Q16,     0x00004000      ; 0.25
.EQU RECON_TOLERANCE_Q16,       0x00000800      ; 0.03125
.EQU LATENT_MIN_Q16,            0xFFFF0000      ; -1.0
.EQU LATENT_MAX_Q16,            0x00010000      ; +1.0
.EQU MODULATION_GAIN_Q16,       0x0000E666      ; 0.90

; ------------------------------------------------------------------------------
; Memory map
; ------------------------------------------------------------------------------
.ORG 0x00000000
.SECTION "ROM_BOOT", RX

BOOT:
    INIT_SYS          #0x00, R0
    INIT_STACK        STACK_BASE, STACK_LIMIT
    VERIFY_ROM        ROM_BOOT_BEGIN, ROM_END, R0
    CMP               R0, STATUS_OK
    JNZ               FATAL_ROM_FAULT

    INIT_VMANIFOLD    MANIFOLD_TABLE, MAX_ACTIVE_PAGES, R0
    CMP               R0, STATUS_OK
    JNZ               FATAL_BOOT_FAULT

    CLEAR_MEM         OMEGA_CANDIDATE_BANK
    CLEAR_MEM         RECEIPT_WORKSPACE
    MOVI              R7, #0x00
    SIGNAL_STATUS     STATUS_READY
    JMP               MAIN_LOOP

ROM_BOOT_BEGIN:
    NOP

; ------------------------------------------------------------------------------
; Core transaction loop
; R0 : status/result
; R1 : bus payload handle
; R2 : free energy Q16.16
; R3 : scratch
; R4 : reconstruction distance Q16.16
; R5 : transaction id
; R6 : candidate digest
; R7 : retry counter
; ------------------------------------------------------------------------------
.SECTION "CORE_EXECUTION_LOOP", RX

MAIN_LOOP:
    WAIT_BUS          R1
    CMP               R1, BUS_EMPTY
    JZ                MAIN_LOOP

    BEGIN_TX          R5
    STORE_HANDLE      REG_INJECTION_BUFFER, R1
    MOVI              R7, #0x00
    CLEAR_MEM         OMEGA_CANDIDATE_BANK

ENCODE_ATTEMPT:
    JSR               EXEC_AUTO_ENCODE
    CMP               R0, STATUS_OK
    JNZ               HANDLE_ENCODE_FAILURE

    JSR               EXEC_LATENT_STAGE
    CMP               R0, STATUS_OK
    JNZ               REJECT_TRANSACTION

    JSR               EXEC_DECODE_VERIFY
    CMP               R0, STATUS_OK
    JNZ               HANDLE_VERIFY_FAILURE

    JSR               EXEC_POLICY_VALIDATE
    CMP               R0, STATUS_OK
    JNZ               REJECT_TRANSACTION

    HASH_STATE        OMEGA_CANDIDATE_BANK, R6
    ATOMIC_COMMIT     OMEGA_CANDIDATE_BANK, OMEGA_T_BANK, R0
    CMP               R0, STATUS_OK
    JNZ               FATAL_COMMIT_FAULT

    APPEND_RECEIPT    RECEIPT_LOG, R5, R6, R2, R4, STATUS_COMMITTED
    SIGNAL_STATUS     STATUS_COMMITTED
    END_TX            R5
    JMP               MAIN_LOOP

HANDLE_ENCODE_FAILURE:
    CMP               R0, STATUS_DRIFT
    JNZ               REJECT_TRANSACTION
    JMP               BOUNDED_RETRY

HANDLE_VERIFY_FAILURE:
    CMP               R0, STATUS_VERIFY_FAILED
    JNZ               REJECT_TRANSACTION

BOUNDED_RETRY:
    INC               R7
    CMP               R7, MAX_RETRIES
    JGE               REJECT_TRANSACTION

    APPLY_MODULATION  R_THETA, MODULATION_GAIN_Q16, Z_LATENT_VECTOR, R0
    CMP               R0, STATUS_OK
    JNZ               REJECT_TRANSACTION
    JMP               ENCODE_ATTEMPT

REJECT_TRANSACTION:
    HASH_STATE        OMEGA_CANDIDATE_BANK, R6
    CLEAR_MEM         OMEGA_CANDIDATE_BANK
    APPEND_RECEIPT    RECEIPT_LOG, R5, R6, R2, R4, STATUS_REJECTED
    APPEND_FAULT      FAULT_LOG, R5, PC, R0, R4
    SIGNAL_STATUS     STATUS_REJECTED
    ABORT_TX          R5
    JMP               MAIN_LOOP

; ------------------------------------------------------------------------------
; Encoder: computes a candidate latent vector and measured free energy.
; Every exit restores the stack frame.
; ------------------------------------------------------------------------------
.SECTION "NEURAL_CODEC_SUBSYSTEM", RX

EXEC_AUTO_ENCODE:
    ENTER             #0x10
    PUSH              R2
    PUSH              R3

    ENCODE_LATENT     REG_INJECTION_BUFFER, R_THETA, Z_LATENT_VECTOR, R0
    CMP               R0, STATUS_OK
    JNZ               AUTO_ENCODE_RETURN

    CHECK_FINITE      Z_LATENT_VECTOR, R0
    CMP               R0, STATUS_OK
    JNZ               AUTO_ENCODE_RETURN

    CALC_FREE_ENERGY  Z_LATENT_VECTOR, R2, R0
    CMP               R0, STATUS_OK
    JNZ               AUTO_ENCODE_RETURN

    CMP               R2, FREE_ENERGY_LIMIT_Q16
    JLE               AUTO_ENCODE_ACCEPT

    MOVI              R0, STATUS_DRIFT
    JMP               AUTO_ENCODE_RETURN

AUTO_ENCODE_ACCEPT:
    MOVI              R0, STATUS_OK

AUTO_ENCODE_RETURN:
    POP               R3
    POP               R2
    LEAVE
    RTS

; ---------------------------------------------------------------------------------
; Quantize into a staging bank. Committed Omega state is never mutated here.
; ------------------------------------------------------------------------------
EXEC_LATENT_STAGE:
    ENTER             #0x08

    QUANTIZE_Q16      Z_LATENT_VECTOR, Z_QUANTIZED, R0
    CMP               R0, STATUS_OK
    JNZ               LATENT_STAGE_RETURN

    CHECK_BOUNDS      Z_QUANTIZED, LATENT_MIN_Q16, LATENT_MAX_Q16, R0
    CMP               R0, STATUS_OK
    JNZ               LATENT_STAGE_RETURN

    CHECK_BUDGET      Z_QUANTIZED, MAX_ACTIVE_PAGES, R0
    CMP               R0, STATUS_OK
    JNZ               LATENT_STAGE_RETURN

    WRITE_MEM         OMEGA_CANDIDATE_BANK, Z_QUANTIZED, R0

LATENT_STAGE_RETURN:
    LEAVE
    RTS

; ------------------------------------------------------------------------------
; Decode from staged state and compare with a tolerance, not exact equality.
; ------------------------------------------------------------------------------
EXEC_DECODE_VERIFY:
    ENTER             #0x08

    READ_MEM          OMEGA_CANDIDATE_BANK, Z_RETRIEVED, R0
    CMP               R0, STATUS_OK
    JNZ               DECODE_VERIFY_RETURN

    DECODE_LATENT     Z_RETRIEVED, R_THETA, REG_RECONSTRUCTION_OUTPUT, R0
    CMP               R0, STATUS_OK
    JNZ               DECODE_VERIFY_RETURN

    CHECK_FINITE      REG_RECONSTRUCTION_OUTPUT, R0
    CMP               R0, STATUS_OK
    JNZ               DECODE_VERIFY_RETURN

    CALC_DISTANCE     REG_RECONSTRUCTION_OUTPUT, REG_INJECTION_BUFFER, R4, R0
    CMP               R0, STATUS_OK
    JNZ               DECODE_VERIFY_RETURN

    CMP               R4, RECON_TOLERANCE_Q16
    JLE               DECODE_VERIFY_ACCEPT

    MOVI              R0, STATUS_VERIFY_FAILED
    JMP               DECODE_VERIFY_RETURN

DECODE_VERIFY_ACCEPT:
    MOVI              R0, STATUS_OK

DECODE_VERIFY_RETURN:
    LEAVE
    RTS

; ------------------------------------------------------------------------------
; Lambda gate: validates policy evidence, bounds, authorization, and receipts.
; ------------------------------------------------------------------------------
EXEC_POLICY_VALIDATE:
    ENTER             #0x08

    VERIFY_AUTH       ACTIVE_AUTHORITY, R0
    CMP               R0, STATUS_OK
    JNZ               POLICY_RETURN

    POLICY_CHECK      OMEGA_CANDIDATE_BANK, R_THETA, POLICY_EVIDENCE, R0
    CMP               R0, STATUS_OK
    JNZ               POLICY_RETURN

    VERIFY_PROVENANCE REG_INJECTION_BUFFER, POLICY_EVIDENCE, R0

POLICY_RETURN:
    LEAVE
    RTS

; ------------------------------------------------------------------------------
; Fatal faults: only integrity failures halt the runtime.
; ------------------------------------------------------------------------------
FATAL_ROM_FAULT:
    APPEND_FAULT      FAULT_LOG, #0x00, PC, STATUS_ROM_FAULT, #0x00
    SIGNAL_STATUS     STATUS_ROM_FAULT
    HALT

FATAL_BOOT_FAULT:
    APPEND_FAULT      FAULT_LOG, #0x00, PC, R0, #0x00
    SIGNAL_STATUS     R0
    HALT

FATAL_COMMIT_FAULT:
    APPEND_FAULT      FAULT_LOG, R5, PC, R0, R4
    SIGNAL_STATUS     R0
    HALT

; ------------------------------------------------------------------------------
; Data and persistent regions
; ------------------------------------------------------------------------------
.ORG 0x00010000
.SECTION "VOLATILE_WORK", RW
REG_INJECTION_BUFFER:       .RES 0x1000
REG_RECONSTRUCTION_OUTPUT:  .RES 0x1000
Z_LATENT_VECTOR:            .RES 0x1000
Z_QUANTIZED:                .RES 0x1000
Z_RETRIEVED:                .RES 0x1000
RECEIPT_WORKSPACE:          .RES 0x0100
R_THETA:                    .Q16 1.000000
ACTIVE_AUTHORITY:           .RES 0x0100
POLICY_EVIDENCE:            .RES 0x0400

.ORG 0x00020000
.SECTION "PERSISTENT_STATE", RWP
OMEGA_T_BANK:               .RES 0x2000
OMEGA_CANDIDATE_BANK:       .RES 0x2000
MANIFOLD_TABLE:             .RES 0x1000
RECEIPT_LOG:                .RES 0x4000
FAULT_LOG:                  .RES 0x2000

.ORG 0x0002FFFF
ROM_END:
    .DWORD 0x44564D58       ; "DVMX"
