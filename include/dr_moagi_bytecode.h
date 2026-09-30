#ifndef DR_MOAGI_BYTECODE_H
#define DR_MOAGI_BYTECODE_H

#include <stdint.h>

#define DR_MOAGI_ROM_WORDS 16u
#define DR_MOAGI_ROM_BYTES 64u
#define DR_MOAGI_ROM_BODY_BYTES 60u
#define DR_MOAGI_CANONICAL_CRC32 UINT32_C(0x96FDFC2F)
#define DR_MOAGI_SUPPLIED_CRC32 UINT32_C(0x7E3F91A2)

/*
 * Canonical raw ROM bytes. This byte array is endian-stable and can be written
 * directly to a binary file or memory-mapped ROM region.
 */
static const uint8_t DR_MOAGI_BYTECODE_ROM_BE[DR_MOAGI_ROM_BYTES] = {
    0x4D, 0x4F, 0x41, 0x47, 0x01, 0x00, 0x00, 0x10,
    0x04, 0x00, 0x00, 0x40, 0x04, 0x20, 0x00, 0x00,
    0x04, 0x40, 0x00, 0x01, 0x08, 0x61, 0x08, 0x00,
    0x0C, 0x83, 0x10, 0x04, 0x08, 0xA4, 0x18, 0x00,
    0x10, 0xC5, 0x20, 0x00, 0x10, 0xE6, 0x28, 0x00,
    0x14, 0xE7, 0x30, 0x00, 0x18, 0xE7, 0x00, 0x08,
    0x1C, 0xE7, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x96, 0xFD, 0xFC, 0x2F
};

/*
 * Logical 32-bit words. Do not write this array's in-memory representation as
 * a ROM image on little-endian hosts; use DR_MOAGI_BYTECODE_ROM_BE instead.
 */
static const uint32_t DR_MOAGI_BYTECODE_WORDS[DR_MOAGI_ROM_WORDS] = {
    UINT32_C(0x4D4F4147), /* [0x0000] Magic header "MOAG" */
    UINT32_C(0x01000010), /* [0x0004] Header: 16 total words */
    UINT32_C(0x04000040), /* [0x0008] INIT_GRID: side length 64 */
    UINT32_C(0x04200000), /* [0x000C] LOAD_COORD */
    UINT32_C(0x04400001), /* [0x0010] CLEAR_ACC */
    UINT32_C(0x08610800), /* [0x0014] ENC_PROJ layer 1 */
    UINT32_C(0x0C831004), /* [0x0018] QUANT_Z */
    UINT32_C(0x08A41800), /* [0x001C] ENC_PROJ layer 2 */
    UINT32_C(0x10C52000), /* [0x0020] DEC_DECONV layer 1 */
    UINT32_C(0x10E62800), /* [0x0024] DEC_DECONV layer 2 */
    UINT32_C(0x14E73000), /* [0x0028] SOL_CURL */
    UINT32_C(0x18E70008), /* [0x002C] DAMP_REG */
    UINT32_C(0x1CE70000), /* [0x0030] SYNC_OUT */
    UINT32_C(0x00000000), /* [0x0034] NOP by slot contract */
    UINT32_C(0x00000000), /* [0x0038] HALT by slot contract */
    DR_MOAGI_CANONICAL_CRC32 /* [0x003C] IEEE CRC-32 over bytes 0x0000-0x003B */
};

#endif /* DR_MOAGI_BYTECODE_H */
