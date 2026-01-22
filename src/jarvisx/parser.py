class Parser:
    def parse(self, text):
        ast = []
        for line in text.strip().splitlines():
            if not line.strip():
                continue
            ast.append(line.strip().split())
        return ast
