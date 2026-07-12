from __future__ import annotations

import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from .ann import TinyAutoencoder
from .compiler import Assembler
from .demo import DEMO_SOURCE, demo_input
from .vm import VANNVirtualMachine


class VirtualIDE(tk.Tk):
    """Lightweight local IDE for assembling and running VANN-ROM programs."""

    def __init__(self) -> None:
        super().__init__()
        self.title("VANN-ROM Ω³ Virtual IDE SDK")
        self.geometry("1180x760")
        self.minsize(900, 620)
        self._build_ui()
        self._load_demo()

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=6)
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="New", command=self._new).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Open", command=self._open).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Save", command=self._save).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Assemble", command=self._assemble).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Run", command=self._run).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Load Demo", command=self._load_demo).pack(side=tk.LEFT, padx=3)

        self.status = tk.StringVar(value="Ready")
        ttk.Label(toolbar, textvariable=self.status).pack(side=tk.RIGHT)

        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        left = ttk.Frame(main)
        right = ttk.Panedwindow(main, orient=tk.VERTICAL)
        main.add(left, weight=3)
        main.add(right, weight=2)

        ttk.Label(left, text="VANN Bytecode Source").pack(anchor=tk.W)
        self.editor = tk.Text(left, wrap=tk.NONE, undo=True, font=("Courier", 12))
        self.editor.pack(fill=tk.BOTH, expand=True)

        input_frame = ttk.LabelFrame(right, text="Input Tensor (JSON)")
        output_frame = ttk.LabelFrame(right, text="Runtime Output / Metrics")
        right.add(input_frame, weight=1)
        right.add(output_frame, weight=2)

        self.input_text = tk.Text(input_frame, height=8, wrap=tk.WORD, font=("Courier", 11))
        self.input_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.output_text = tk.Text(output_frame, wrap=tk.WORD, font=("Courier", 10), state=tk.DISABLED)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def _set_output(self, text: str) -> None:
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.configure(state=tk.DISABLED)

    def _new(self) -> None:
        self.editor.delete("1.0", tk.END)
        self.status.set("New program")

    def _open(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("VANN source", "*.vann"), ("Text", "*.txt"), ("All", "*")])
        if path:
            with open(path, "r", encoding="utf-8") as handle:
                self.editor.delete("1.0", tk.END)
                self.editor.insert(tk.END, handle.read())
            self.status.set(path)

    def _save(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".vann", filetypes=[("VANN source", "*.vann")])
        if path:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self.editor.get("1.0", tk.END).rstrip() + "\n")
            self.status.set(path)

    def _assemble(self) -> None:
        try:
            program = Assembler().assemble(self.editor.get("1.0", tk.END))
            listing = []
            for index, instruction in enumerate(program.instructions):
                listing.append(f"{index:04d} {instruction.opcode.name:<18} {instruction.encode().hex()}")
            self._set_output("\n".join(listing))
            self.status.set(f"Assembled {len(program.instructions)} instructions")
        except Exception as exc:
            messagebox.showerror("Assembly error", str(exc))

    def _run(self) -> None:
        try:
            source = self.editor.get("1.0", tk.END)
            data = np.asarray(json.loads(self.input_text.get("1.0", tk.END)), dtype=np.float32)
            if data.ndim == 1:
                data = data[None, :]
            if data.ndim != 2:
                raise ValueError("Input must be a vector or a batch of vectors")
            latent_dim = max(1, min(data.shape[1] - 1, data.shape[1] // 3))
            program = Assembler().assemble(source)
            rendered: list[str] = []
            vm = VANNVirtualMachine(
                TinyAutoencoder(data.shape[1], latent_dim),
                output_sink=rendered.append,
            )
            vm.load_program(program.instructions)
            vm.set_input(data)
            result = vm.run()
            report = {
                "rendered": rendered,
                "output": result.output,
                "latent": result.latent,
                "residual": result.residual,
                "metrics": result.metrics,
                "policy": result.policy,
                "rom": vm.rom.stats(),
                "journal": result.journal,
            }
            self._set_output(json.dumps(report, indent=2))
            self.status.set(f"Halted after {result.cycles} cycles")
        except Exception as exc:
            messagebox.showerror("Runtime error", str(exc))

    def _load_demo(self) -> None:
        self.editor.delete("1.0", tk.END)
        self.editor.insert(tk.END, DEMO_SOURCE)
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert(tk.END, json.dumps(demo_input().tolist()[0], indent=2))
        self.status.set("Demo loaded")


def main() -> None:
    VirtualIDE().mainloop()


if __name__ == "__main__":
    main()
