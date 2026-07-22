from jarvisx.assembler import Assembler
from jarvisx.decoder import Decoder
from jarvisx.parser import Parser


def test_assembler_resolves_labels_and_signed_immediates():
    bytecode = Assembler().assemble(
        Parser().parse("SET A -1\nloop:\nJMP loop\nHALT")
    )
    decoder = Decoder()
    assert decoder.decode(bytecode[0]).imm == -1
    assert decoder.decode(bytecode[1]).imm == 1
