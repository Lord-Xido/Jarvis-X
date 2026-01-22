from jarvisx.parser import Parser

def test_parser():
    ast = Parser().parse("SET Ψ 5\nHALT")
    assert ast[0][0] == "SET"
