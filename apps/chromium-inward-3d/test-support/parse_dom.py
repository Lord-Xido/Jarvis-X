import json
import sys
from html.parser import HTMLParser

class Parser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = {'tag': 'document', 'attrs': {}, 'children': []}
        self.stack = [self.root]
    def handle_starttag(self, tag, attrs):
        node = {'tag': tag, 'attrs': dict(attrs), 'children': []}
        self.stack[-1]['children'].append(node)
        if tag not in {'meta', 'input', 'br', 'hr', 'img', 'link'}:
            self.stack.append(node)
    def handle_endtag(self, tag):
        for i in range(len(self.stack)-1, 0, -1):
            if self.stack[i]['tag'] == tag:
                self.stack = self.stack[:i]
                return
    def handle_data(self, data):
        self.stack[-1]['children'].append({'text': data})

p = Parser()
p.feed(sys.stdin.read())
print(json.dumps(p.root))
