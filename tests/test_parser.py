from jarvisx.parser import Parser


def test_parser_ignores_comments_and_preserves_labels():
    ast = Parser().parse("SET Ψ 5 # input\nloop:\nHALT ; done")
    assert ast == [["SET", "Ψ", "5"], ["loop:"], ["HALT"]]
