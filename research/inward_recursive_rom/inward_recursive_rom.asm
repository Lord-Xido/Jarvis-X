; ==============================================================================
; SYSTEM: INWARD RECURSIVE 3D AUTO-OPTIMIZATION ROM ENGINE
; PROFILE: DM-vOmegaXi+ / ADR-009-compatible 16-register research macro-source
; ARITHMETIC: saturating signed Q16.16, widened multiply before >> 16
; CONTROL: no data-dependent branch; 192 compile-time-unrolled passes
; FIXED POINT: sticky masked lock when gamma <= EPS
; ==============================================================================
;
; This is macro-source, not a new canonical opcode table. QLOAD/QSTORE/MUL_FX,
; MAC_FX, ADD_SAT, SUB_SAT and ABS_SAT name semantic primitives that must lower
; to an implementation preserving ADR-009 field widths and Q16.16 behavior.
;
; Register allocation
;   R0-R2   current focused recurrent state Xk
;   R3      latent z
;   R4      typed pointer to W0
;   R5      typed pointer to state/scratch block
;   R6      scratch / signed residual
;   R7      eta = 1/16 = 0x00001000
;   R8-R10  reconstruction Xhat
;   R11     coupling C = sum(e_i * w_i)
;   R12     accumulator / gap / candidate scratch
;   R13     rho = 1/4 = 0x00004000
;   R14     sticky lock mask: 0x00000000 or 0xFFFFFFFF
;   R15     lens at init, scratch thereafter
;
; State offsets under R5
.equ W0,          0x000
.equ W1,          0x004
.equ W2,          0x008
.equ ANCHOR_X,    0x020
.equ ANCHOR_Y,    0x024
.equ ANCHOR_Z,    0x028
.equ ERR_X,       0x040
.equ ERR_Y,       0x044
.equ ERR_Z,       0x048
.equ GAP,         0x04C
.equ LOCK_MASK,   0x050
.equ RELU_MASK,   0x054
.equ UNLOCK_MASK, 0x058
.equ OUT_X,       0x100
.equ OUT_Y,       0x104
.equ OUT_Z,       0x108
.equ OUT_LATENT,  0x10C
.equ OUT_GAP,     0x110
.equ OUT_LOCK,    0x114

.equ EPS,         0x040       ; 64 raw Q16 LSB ~= 9.765625e-4
.equ EPS_PLUS_1,  0x041       ; used for inclusive <= comparison
.equ PASS_COUNT,  192

.SECTION .TEXT_INWARD_ROM
.ALIGN 32

ENTRY_INWARD_RECURSION:
    ; --------------------------------------------------------------------------
    ; One-time constants and typed base pointers
    ; --------------------------------------------------------------------------
    MOV_IMM   R7,  0x00001000       ; eta = 1/16
    MOV_IMM   R13, 0x00004000       ; rho = 1/4
    MOV_IMM   R14, 0x00000000       ; not locked
    MOV_IMM   R15, 0x0000C59A       ; lens = 0.771881103515625
    LEA        R4, R5, W0            ; typed weight base pointer

    ; --------------------------------------------------------------------------
    ; Focus exactly once. The focused coordinates become an immutable anchor.
    ; Reapplying the lens every pass would introduce an unintended contraction.
    ; --------------------------------------------------------------------------
    MUL_FX    R0, R0, R15
    MUL_FX    R1, R1, R15
    MUL_FX    R2, R2, R15
    QSTORE    R0, R5, ANCHOR_X
    QSTORE    R1, R5, ANCHOR_Y
    QSTORE    R2, R5, ANCHOR_Z
    STORE_REG R14, R5, LOCK_MASK

    ; ========================================================================== 
    ; Fixed-work recursive fold. .REPT is compile-time expansion, not runtime
    ; control flow. Convergence only changes masks/data; execution remains fixed.
    ; ========================================================================== 
    .REPT PASS_COUNT

        ; ----------------------------------------------------------------------
        ; 1. Encode: a = W^T X. Accumulator is explicitly reset every pass.
        ; ----------------------------------------------------------------------
        MOV_IMM R12, 0x00000000
        QLOAD   R15, R4, 0x000
        MAC_FX  R12, R0, R15
        QLOAD   R15, R4, 0x004
        MAC_FX  R12, R1, R15
        QLOAD   R15, R4, 0x008
        MAC_FX  R12, R2, R15

        ; z = ReLU(a), branchlessly.
        MOV      R3, R12
        SAR      R15, R12, 31
        NOT      R15, R15
        AND      R3, R3, R15

        ; ReLU'(a): -1 when z > 0, otherwise 0. Define derivative at zero = 0.
        MOV      R15, R3
        SUB_IMM  R15, R15, 0x001
        SAR      R15, R15, 31
        NOT      R15, R15
        STORE_REG R15, R5, RELU_MASK

        ; ----------------------------------------------------------------------
        ; 2. Decode with tied transpose pair: Xhat_i = w_i * z.
        ; ----------------------------------------------------------------------
        QLOAD    R15, R4, 0x000
        MUL_FX   R8, R3, R15
        QLOAD    R15, R4, 0x004
        MUL_FX   R9, R3, R15
        QLOAD    R15, R4, 0x008
        MUL_FX   R10, R3, R15

        ; ----------------------------------------------------------------------
        ; 3. Immutable-anchor reality gap.
        ; e = Xhat - A, gamma = |ex| + |ey| + |ez|.
        ; Preserve signed residuals for the gradient.
        ; ----------------------------------------------------------------------
        MOV_IMM  R12, 0x00000000

        QLOAD    R6, R5, ANCHOR_X
        SUB_SAT  R6, R8, R6
        QSTORE   R6, R5, ERR_X
        ABS_SAT  R15, R6
        ADD_SAT  R12, R12, R15

        QLOAD    R6, R5, ANCHOR_Y
        SUB_SAT  R6, R9, R6
        QSTORE   R6, R5, ERR_Y
        ABS_SAT  R15, R6
        ADD_SAT  R12, R12, R15

        QLOAD    R6, R5, ANCHOR_Z
        SUB_SAT  R6, R10, R6
        QSTORE   R6, R5, ERR_Z
        ABS_SAT  R15, R6
        ADD_SAT  R12, R12, R15

        QSTORE   R12, R5, GAP

        ; ----------------------------------------------------------------------
        ; 4. Inclusive branchless convergence test: gamma <= EPS.
        ; Since gamma >= 0, sign(gamma - (EPS + 1)) is all ones iff <= EPS.
        ; Sticky lock means convergence cannot subsequently unlock.
        ; ----------------------------------------------------------------------
        MOV       R11, R12
        SUB_IMM   R11, R11, EPS_PLUS_1
        SAR       R11, R11, 31
        OR        R14, R14, R11
        STORE_REG R14, R5, LOCK_MASK
        NOT       R15, R14
        STORE_REG R15, R5, UNLOCK_MASK

        ; ----------------------------------------------------------------------
        ; 5. Exact tied-weight gradient for active ReLU region.
        ; C = sum_i e_i w_i
        ; g_j = e_j*z + C*ReLU'(a)*x_j
        ; W_candidate = W - eta*g
        ; When locked, masked select retains the old weight exactly.
        ; ----------------------------------------------------------------------
        MOV_IMM   R11, 0x00000000
        QLOAD     R6, R5, ERR_X
        QLOAD     R15, R4, 0x000
        MAC_FX    R11, R6, R15
        QLOAD     R6, R5, ERR_Y
        QLOAD     R15, R4, 0x004
        MAC_FX    R11, R6, R15
        QLOAD     R6, R5, ERR_Z
        QLOAD     R15, R4, 0x008
        MAC_FX    R11, R6, R15

        ; ---- W0 ---------------------------------------------------------------
        QLOAD     R6, R5, ERR_X
        MUL_FX    R12, R6, R3           ; direct decoder term e0*z
        MUL_FX    R6, R11, R0           ; encoder coupling term C*x0
        LOAD_REG  R15, R5, RELU_MASK
        AND       R6, R6, R15
        ADD_SAT   R12, R12, R6
        MUL_FX    R12, R12, R7          ; eta*g0
        QLOAD     R6, R4, 0x000          ; old W0
        SUB_SAT   R12, R6, R12           ; candidate W0
        AND       R6, R6, R14            ; old when locked
        LOAD_REG  R15, R5, UNLOCK_MASK
        AND       R12, R12, R15          ; candidate when unlocked
        OR        R6, R6, R12
        QSTORE    R6, R4, 0x000

        ; ---- W1 ---------------------------------------------------------------
        QLOAD     R6, R5, ERR_Y
        MUL_FX    R12, R6, R3
        MUL_FX    R6, R11, R1
        LOAD_REG  R15, R5, RELU_MASK
        AND       R6, R6, R15
        ADD_SAT   R12, R12, R6
        MUL_FX    R12, R12, R7
        QLOAD     R6, R4, 0x004
        SUB_SAT   R12, R6, R12
        AND       R6, R6, R14
        LOAD_REG  R15, R5, UNLOCK_MASK
        AND       R12, R12, R15
        OR        R6, R6, R12
        QSTORE    R6, R4, 0x004

        ; ---- W2 ---------------------------------------------------------------
        QLOAD     R6, R5, ERR_Z
        MUL_FX    R12, R6, R3
        MUL_FX    R6, R11, R2
        LOAD_REG  R15, R5, RELU_MASK
        AND       R6, R6, R15
        ADD_SAT   R12, R12, R6
        MUL_FX    R12, R12, R7
        QLOAD     R6, R4, 0x008
        SUB_SAT   R12, R6, R12
        AND       R6, R6, R14
        LOAD_REG  R15, R5, UNLOCK_MASK
        AND       R12, R12, R15
        OR        R6, R6, R12
        QSTORE    R6, R4, 0x008

        ; ----------------------------------------------------------------------
        ; 6. Anchored inward recurrence.
        ; X_candidate = A + rho*(Xhat - A)
        ; Locked lanes preserve X exactly.
        ; ----------------------------------------------------------------------
        QLOAD     R6, R5, ANCHOR_X
        SUB_SAT   R12, R8, R6
        MUL_FX    R12, R12, R13
        ADD_SAT   R12, R6, R12
        AND       R0, R0, R14
        LOAD_REG  R15, R5, UNLOCK_MASK
        AND       R12, R12, R15
        OR        R0, R0, R12

        QLOAD     R6, R5, ANCHOR_Y
        SUB_SAT   R12, R9, R6
        MUL_FX    R12, R12, R13
        ADD_SAT   R12, R6, R12
        AND       R1, R1, R14
        LOAD_REG  R15, R5, UNLOCK_MASK
        AND       R12, R12, R15
        OR        R1, R1, R12

        QLOAD     R6, R5, ANCHOR_Z
        SUB_SAT   R12, R10, R6
        MUL_FX    R12, R12, R13
        ADD_SAT   R12, R6, R12
        AND       R2, R2, R14
        LOAD_REG  R15, R5, UNLOCK_MASK
        AND       R12, R12, R15
        OR        R2, R2, R12

    .ENDR

    ; --------------------------------------------------------------------------
    ; 7. Final bounded commit. OUT_LOCK tells the caller whether the tolerance
    ; was reached during the fixed pass budget. Non-convergence is observable;
    ; it is not silently relabeled as a fixed point.
    ; --------------------------------------------------------------------------
    QSTORE    R0, R5, OUT_X
    QSTORE    R1, R5, OUT_Y
    QSTORE    R2, R5, OUT_Z
    QSTORE    R3, R5, OUT_LATENT
    QLOAD     R12, R5, GAP
    QSTORE    R12, R5, OUT_GAP
    STORE_REG R14, R5, OUT_LOCK
    HALT
