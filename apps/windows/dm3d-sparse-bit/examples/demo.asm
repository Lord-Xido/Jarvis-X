# DM3D bit ISA. Branch addresses are zero based instruction numbers.
MOV R0 1
STORE 2 2 2 R0
LOAD R1 2 2 2
XOR R2 R0 R1
JNZ R2 8
STEP
HASH
HALT
MOV R3 1
JNZ R3 8
HALT
