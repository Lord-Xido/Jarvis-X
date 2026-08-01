from __future__ import annotations


class Parser:
    def parse(self, text: str) -> list[list[str]]:
        ast: list[list[str]] = []
        for line in text.strip().splitlines():
            if not line.strip():
                continue
            ast.append(line.strip().split())
        return ast
