"""Minimal sparse JX-3D-1PB-BitVM demonstration."""

from jarvisx.bitvm_3d_1pb import (
    AddressClass,
    BitAddress,
    BitInstruction,
    BitOpcode,
    Sparse3DBitVM,
)


def main() -> None:
    vm = Sparse3DBitVM()
    start = BitAddress(
        asid=0,
        access_class=int(AddressClass.WRITE),
        x=10,
        y=20,
        z=30,
        byte_offset=0,
        bit_offset=0,
    )
    receipt = vm.execute(
        BitInstruction(BitOpcode.BSET, destination=start, length_bits=16)
    )

    print(receipt)
    print("virtual bytes:", vm.virtual_byte_count)
    print("resident bytes:", vm.resident_payload_bytes)


if __name__ == "__main__":
    main()
