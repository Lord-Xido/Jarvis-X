from flask import Flask, request, jsonify
from .core import CodexVM
from .parser import Parser
from .assembler import Assembler

app = Flask(__name__)
vm = CodexVM()

@app.route("/run", methods=["POST"])
def run_code():
    source = request.json.get("source", "")
    ast = Parser().parse(source)
    bytecode = Assembler().assemble(ast)
    vm.load(bytecode)
    vm.run()
    return jsonify({
        "registers": vm.regs.snapshot(),
        "ledger_entries": len(vm.ledger.chain)
    })

def start_api():
    app.run(host="0.0.0.0", port=8080)
