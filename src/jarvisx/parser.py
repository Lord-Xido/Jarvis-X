class Parser:
    def parse(self, text):
        ast = []
        for raw_line in str(text).splitlines():
            line = raw_line
            for marker in ("#", ";"):
                line = line.split(marker, 1)[0]
            line = line.strip()
            if not line:
                continue
            ast.append(line.split())
        return ast
