"""Canonical Jarvis-X instruction registry."""

OPCODES = {
    "SET": 0x01,
    "ADD": 0x03,
    "SUB": 0x04,
    "HALT": 0x0A,
    "LOAD30": 0x20,
    "ENCODE30": 0x21,
    "PLACE30": 0x22,
    "FIELD30": 0x23,
    "PREDICT30": 0x24,
    "COMPARE30": 0x25,
    "UPDATE_MEMORY30": 0x26,
    "PROJECT30": 0x27,
    "DECODE30": 0x28,
    "HALT30": 0x29,
}

OPCODE_NAMES = {value: key for key, value in OPCODES.items()}
