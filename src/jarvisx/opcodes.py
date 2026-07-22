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
    "LOAD3D": 0x30,
    "ABSTRACT3D": 0x31,
    "ROUTE3D": 0x32,
    "ATTEND3D": 0x33,
    "PREDICT3D": 0x34,
    "COMPARE3D": 0x35,
    "LEARN3D": 0x36,
    "PROJECT3D": 0x37,
    "DECODE3D": 0x38,
    "HALT3D": 0x39,
}

OPCODE_NAMES = {value: key for key, value in OPCODES.items()}
