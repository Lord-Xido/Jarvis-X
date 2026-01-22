from flask import Flask, render_template_string, request
from .core import CodexVM
from .parser import Parser
from .assembler import Assembler

app = Flask(__name__)
vm = CodexVM()

HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Jarvis-X Dashboard</title>
</head>
<body>
  <h1>Jarvis-X Control Panel</h1>
  <form method="post">
    <textarea name="code" rows="10" cols="60"></textarea><br>
    <button type="submit">Run</button>
  </form>
  {% if result %}
    <h3>Registers</h3>
    <pre>{{ result }}</pre>
  {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET","POST"])
def home():
    result = None
    if request.method == "POST":
        source = request.form["code"]
        ast = Parser().parse(source)
        bytecode = Assembler().assemble(ast)
        vm.load(bytecode)
        vm.run()
        result = vm.regs.snapshot()
    return render_template_string(HTML, result=result)

def start_web():
    app.run(host="0.0.0.0", port=5000)
