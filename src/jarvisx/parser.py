class Parser:
    def parse(self, text):
        if not isinstance(text, str):
            raise TypeError("source must be text")
        ast = []
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].split(";", 1)[0].strip()
            if line:
                ast.append(line.split())
        if not ast:
            raise ValueError("source contains no instructions")
        return ast
