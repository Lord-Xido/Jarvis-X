"""
Dr Moagi Multimodal Multimedia Editor
=======================================
An interactive Python IDE powered by the geometric mind.
Features:
- 3D code manifold visualization (real-time)
- Audio feedback for evolution events
- Live code evolution with swarm visualization
- Auto-encoding/decoding of code
- Multimodal input (text, voice, gesture via keyboard)
- Video output of the geometric mind's evolution

Requires:
pip install pyqtgraph pyqt5 pyttsx3 pyaudio scipy opencv-python PyOpenGL astunparse

NOTE: This is an experimental desktop runtime. The evolution engine executes
candidate Python functions with exec(); run it only with trusted local inputs
until process isolation/timeouts are added.
"""

import sys
import os
import ast
import astunparse
import numpy as np
import time
import random
import copy
import threading
import queue
import json
from datetime import datetime
from collections import deque
from functools import lru_cache
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import pyttsx3
import pyaudio
import wave
import struct
import cv2
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# For audio synthesis
import math
import scipy.signal as signal

# -----------------------------------------------------------------
# 1. AUDIO ENGINE (Geometric Sound Synthesis)
# -----------------------------------------------------------------

class GeometricAudio:
    """Generates audio feedback based on geometric state."""

    def __init__(self):
        self.py_audio = pyaudio.PyAudio()
        self.stream = None
        self.volume = 0.5
        self.fs = 44100
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.8)

    def speak(self, text):
        """Text-to-speech output."""
        self.engine.say(text)
        self.engine.runAndWait()

    def play_tone(self, frequency, duration=0.1):
        """Generate a geometric tone based on frequency."""
        frames = int(self.fs * duration)
        t = np.linspace(0, duration, frames)
        wave = self.volume * np.sin(2 * np.pi * frequency * t)
        wave += 0.3 * self.volume * np.sin(2 * np.pi * frequency * 2 * t)
        wave += 0.1 * self.volume * np.sin(2 * np.pi * frequency * 3 * t)
        envelope = np.exp(-5 * t / duration)
        wave *= envelope
        wave_bytes = (wave * 32767).astype(np.int16).tobytes()

        if self.stream is None or not self.stream.is_active():
            self.stream = self.py_audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.fs,
                output=True
            )
        self.stream.write(wave_bytes)

    def play_evolution_sound(self, fitness):
        """Sound for evolution events."""
        base_freq = 440 + fitness * 100
        self.play_tone(base_freq, 0.15)
        time.sleep(0.05)
        self.play_tone(base_freq * 1.5, 0.1)

    def play_attractor_sound(self):
        """Sound when attractor is reached."""
        for freq in [440, 550, 660, 880, 1100]:
            self.play_tone(freq, 0.08)
            time.sleep(0.05)

    def play_error_sound(self):
        """Sound for errors."""
        self.play_tone(200, 0.3)

    def cleanup(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.py_audio.terminate()


# -----------------------------------------------------------------
# 2. 3D VISUALIZATION ENGINE (OpenGL + PyQtGraph)
# -----------------------------------------------------------------

class Geometry3DViewer(gl.GLViewWidget):
    """3D visualization of the code manifold and swarm."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Dr Moagi - 3D Code Manifold')
        self.setGeometry(0, 0, 800, 600)
        self.setCameraPosition(distance=20, elevation=30, azimuth=45)

        self.swarm_points = np.zeros((0, 3))
        self.manifold_surface = None
        self.code_cloud = None
        self.attractor_point = None
        self.history_trajectory = []

        self.colors = {
            'swarm': (0.2, 0.6, 1.0, 0.7),
            'attractor': (1.0, 0.2, 0.2, 1.0),
            'trajectory': (0.0, 1.0, 0.0, 0.3),
            'manifold': (0.8, 0.6, 0.2, 0.3)
        }

        self.init_geometry()

    def init_geometry(self):
        """Initialize 3D scene."""
        grid = gl.GLGridItem()
        grid.scale(0.5, 0.5, 0.5)
        self.addItem(grid)

        ax = gl.GLAxisItem()
        ax.scale(2, 2, 2)
        self.addItem(ax)

        self.add_item = self.addItem

        self.scatter_swarm = gl.GLScatterPlotItem()
        self.addItem(self.scatter_swarm)

        self.scatter_attractor = gl.GLScatterPlotItem()
        self.addItem(self.scatter_attractor)

        self.line_manifold = gl.GLLinePlotItem()
        self.addItem(self.line_manifold)

        self.mesh = gl.GLMeshItem()
        self.addItem(self.mesh)

    def update_swarm(self, points):
        """Update swarm particle positions."""
        if len(points) > 0:
            self.swarm_points = points
            colors = np.zeros((len(points), 4))
            colors[:, 3] = 0.7
            z_norm = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min() + 0.001)
            colors[:, 0] = 0.2 + 0.8 * z_norm
            colors[:, 1] = 0.6
            colors[:, 2] = 1.0 - 0.8 * z_norm

            self.scatter_swarm.setData(pos=points, color=colors, size=3)
        else:
            self.scatter_swarm.setData(pos=np.zeros((0, 3)), size=0)

    def update_attractor(self, point):
        """Update attractor position."""
        if point is not None:
            pos = np.array([point])
            colors = np.array([[1.0, 0.2, 0.2, 1.0]])
            self.scatter_attractor.setData(pos=pos, color=colors, size=8)

    def update_trajectory(self, trajectory):
        """Update swarm trajectory path."""
        if len(trajectory) > 1:
            self.line_manifold.setData(pos=np.array(trajectory), color=(0, 1, 0, 0.5), width=1)

    def update_manifold(self, surface_points):
        """Update the manifold surface."""
        if len(surface_points) > 0:
            pass

    def clear(self):
        """Clear all visual elements."""
        self.swarm_points = np.zeros((0, 3))
        self.history_trajectory = []
        self.scatter_swarm.setData(pos=np.zeros((0, 3)), size=0)
        self.scatter_attractor.setData(pos=np.zeros((0, 3)), size=0)
        self.line_manifold.setData(pos=np.zeros((0, 3)), color=(0, 0, 0, 0))


# -----------------------------------------------------------------
# 3. CODE EVOLUTION ENGINE (Dr Moagi Core)
# -----------------------------------------------------------------

class DrMoagiEvolutionEngine:
    """Core engine with geometric evolution."""

    def __init__(self):
        self.population = []
        self.fitnesses = []
        self.best_fitness = (0, float('inf'), float('inf'))
        self.best_src = ""
        self.best_ast = None
        self.generation = 0
        self.history = []
        self.latent_space = []

        self.population_size = 30
        self.mutation_rate = 0.3
        self.crossover_rate = 0.2

        self.initialize_population()

    def initialize_population(self):
        """Seed population with basic functions."""
        templates = [
            "def func(x):\n    return x",
            "def func(x):\n    if x > 0:\n        return x\n    return -x",
            "def func(x):\n    return x * x",
            "def func(n):\n    s = 0\n    for i in range(n):\n        s += i\n    return s",
        ]
        for src in templates:
            try:
                tree = ast.parse(src)
                self.population.append(tree.body[0])
            except Exception:
                pass

        while len(self.population) < self.population_size:
            src = self.generate_random_code()
            try:
                tree = ast.parse(src)
                self.population.append(tree.body[0])
            except Exception:
                pass

    def generate_random_code(self):
        """Generate random Python function."""
        patterns = [
            "def func(n):\n    return n",
            "def func(n):\n    if n <= 1: return 1\n    return n * func(n-1)",
            "def func(arr):\n    total = 0\n    for x in arr: total += x\n    return total",
            "def func(a,b):\n    return a + b",
        ]
        src = random.choice(patterns)
        import re
        nums = re.findall(r'\b\d+\b', src)
        if nums:
            old = random.choice(nums)
            new = str(int(old) + random.choice([-1, 1, 2]))
            if int(new) >= 0:
                src = src.replace(old, new, 1)
        return src

    def mutate_ast(self, node):
        """Mutate an AST node."""
        nodes = list(ast.walk(node))
        if not nodes:
            return
        target = random.choice(nodes)
        if isinstance(target, ast.Constant):
            if isinstance(target.value, (int, float)):
                delta = random.choice([-1, 1, 2, -2])
                if isinstance(target.value, int):
                    target.value = max(0, target.value + delta)
                else:
                    target.value += delta * 0.5
        elif isinstance(target, ast.BinOp):
            ops = [ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod]
            target.op = random.choice(ops)()
        elif isinstance(target, ast.Name):
            target.id = random.choice(['n', 'x', 'y', 'res', 'a', 'b'])

    def crossover_ast(self, a, b):
        """Crossover two ASTs."""
        body_a = None
        body_b = None
        for n in ast.walk(a):
            if isinstance(n, ast.FunctionDef):
                body_a = n.body
                break
        for n in ast.walk(b):
            if isinstance(n, ast.FunctionDef):
                body_b = n.body
                break
        if body_a and body_b and len(body_a) > 1 and len(body_b) > 1:
            idx_a = random.randint(0, len(body_a)-1)
            idx_b = random.randint(0, len(body_b)-1)
            body_a[idx_a], body_b[idx_b] = body_b[idx_b], body_a[idx_a]
        return a

    def evaluate_fitness(self, src, test_cases):
        """Evaluate code against test cases."""
        namespace = {}
        try:
            exec(src, namespace)
        except Exception:
            return (0, float('inf'), len(src))

        func = None
        for name, obj in namespace.items():
            if callable(obj) and name != 'builtins':
                func = obj
                break
        if func is None:
            return (0, float('inf'), len(src))

        passed = 0
        total_time = 0.0
        for args, expected in test_cases:
            try:
                start = time.perf_counter()
                result = func(*args)
                elapsed = time.perf_counter() - start
                total_time += elapsed
                if result == expected:
                    passed += 1
            except Exception:
                pass

        try:
            tree = ast.parse(src)
            size = sum(1 for _ in ast.walk(tree))
        except Exception:
            size = len(src)

        return (passed, total_time, size)

    def evolve(self, test_cases, audio=None):
        """Perform one generation of evolution."""
        self.generation += 1

        self.fitnesses = []
        for ast_node in self.population:
            src = astunparse.unparse(ast_node)
            fitness = self.evaluate_fitness(src, test_cases)
            self.fitnesses.append(fitness)

            if (fitness[0] > self.best_fitness[0] or
                (fitness[0] == self.best_fitness[0] and fitness[1] < self.best_fitness[1])):
                self.best_fitness = fitness
                self.best_src = src
                self.best_ast = ast_node

        new_pop = []
        for _ in range(self.population_size):
            parent = random.choice(self.population)
            child = copy.deepcopy(parent)
            if random.random() < self.mutation_rate:
                self.mutate_ast(child)
            if random.random() < self.crossover_rate:
                other = random.choice(self.population)
                child = self.crossover_ast(child, other)
            new_pop.append(child)

        new_fitnesses = []
        for ast_node in new_pop:
            src = astunparse.unparse(ast_node)
            fitness = self.evaluate_fitness(src, test_cases)
            new_fitnesses.append(fitness)

            if (fitness[0] > self.best_fitness[0] or
                (fitness[0] == self.best_fitness[0] and fitness[1] < self.best_fitness[1])):
                self.best_fitness = fitness
                self.best_src = src
                self.best_ast = ast_node

        combined = list(zip(self.population, self.fitnesses)) + list(zip(new_pop, new_fitnesses))
        combined.sort(key=lambda x: (x[1][0], -x[1][1], -x[1][2]), reverse=True)
        self.population = [x[0] for x in combined[:self.population_size]]
        self.fitnesses = [x[1] for x in combined[:self.population_size]]

        self.history.append({
            'generation': self.generation,
            'best_fitness': self.best_fitness[0],
            'best_time': self.best_fitness[1],
            'best_size': self.best_fitness[2]
        })

        if audio and self.best_fitness[0] > 0:
            audio.play_evolution_sound(self.best_fitness[0] / len(test_cases))

        return self.best_fitness, self.best_src

    def get_latent_points(self):
        """Extract 3D latent points for visualization."""
        points = []
        for ast_node in self.population:
            src = astunparse.unparse(ast_node)
            try:
                tree = ast.parse(src)
                x = sum(1 for _ in ast.walk(tree)) / 10

                def max_depth(node, d=0):
                    depths = [d]
                    for child in ast.iter_child_nodes(node):
                        depths.extend(max_depth(child, d+1))
                    return depths

                y = max(max_depth(tree)) if tree else 0
                z = random.random() * 5
                points.append([x, y, z])
            except Exception:
                points.append([0, 0, 0])
        return np.array(points)


# -----------------------------------------------------------------
# 4. MULTIMODAL EDITOR (Main Application)
# -----------------------------------------------------------------

class DrMoagiEditor(QMainWindow):
    """Main multimodal editor window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Dr Moagi - Multimodal Python Editor')
        self.setGeometry(100, 100, 1400, 900)

        self.audio = GeometricAudio()
        self.engine = DrMoagiEvolutionEngine()
        self.viewer = Geometry3DViewer()

        self.is_evolving = False
        self.is_recording = False
        self.current_fitness = 0
        self.test_cases = self.get_default_test_cases()

        self.evolution_thread = None
        self.visualization_timer = QTimer()
        self.visualization_timer.timeout.connect(self.update_visualization)
        self.visualization_timer.start(100)

        self.init_ui()
        self.audio.speak("Dr Moagi editor initialized. Ready for geometric coding.")

    def get_default_test_cases(self):
        """Get default test cases for evolution."""
        return [
            ((0,), 0),
            ((1,), 1),
            ((2,), 3),
            ((3,), 6),
            ((4,), 10),
            ((5,), 15),
        ]

    def init_ui(self):
        """Initialize the user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        viewer_frame = QFrame()
        viewer_frame.setFrameStyle(QFrame.StyledPanel)
        viewer_layout = QVBoxLayout(viewer_frame)
        viewer_layout.addWidget(self.viewer)
        viewer_frame.setMinimumWidth(600)
        layout.addWidget(viewer_frame, 2)

        right_panel = QFrame()
        right_panel.setFrameStyle(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_panel)

        editor_label = QLabel("Code Editor:")
        self.code_editor = QTextEdit()
        self.code_editor.setFont(QFont("Courier New", 12))
        self.code_editor.setPlainText(
            "def func(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return func(n-1) + func(n-2)"
        )
        right_layout.addWidget(editor_label)
        right_layout.addWidget(self.code_editor)

        control_group = QGroupBox("Evolution Control")
        control_layout = QVBoxLayout()

        btn_row1 = QHBoxLayout()
        self.btn_evolve = QPushButton("▶ Evolve")
        self.btn_evolve.clicked.connect(self.toggle_evolution)
        btn_row1.addWidget(self.btn_evolve)

        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.clicked.connect(self.stop_evolution)
        btn_row1.addWidget(self.btn_stop)
        control_layout.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        self.btn_reset = QPushButton("↺ Reset")
        self.btn_reset.clicked.connect(self.reset_evolution)
        btn_row2.addWidget(self.btn_reset)

        self.btn_speak = QPushButton("🔊 Speak Code")
        self.btn_speak.clicked.connect(self.speak_code)
        btn_row2.addWidget(self.btn_speak)
        control_layout.addLayout(btn_row2)

        control_group.setLayout(control_layout)
        right_layout.addWidget(control_group)

        param_group = QGroupBox("Parameters")
        param_layout = QGridLayout()

        param_layout.addWidget(QLabel("Pop Size:"), 0, 0)
        self.spin_pop = QSpinBox()
        self.spin_pop.setRange(10, 100)
        self.spin_pop.setValue(30)
        param_layout.addWidget(self.spin_pop, 0, 1)

        param_layout.addWidget(QLabel("Mutation Rate:"), 1, 0)
        self.slider_mutation = QSlider(Qt.Horizontal)
        self.slider_mutation.setRange(0, 100)
        self.slider_mutation.setValue(30)
        self.slider_mutation.valueChanged.connect(self.update_mutation_label)
        param_layout.addWidget(self.slider_mutation, 1, 1)
        self.lbl_mutation = QLabel("0.30")
        param_layout.addWidget(self.lbl_mutation, 1, 2)

        param_group.setLayout(param_layout)
        right_layout.addWidget(param_group)

        test_group = QGroupBox("Test Cases")
        test_layout = QVBoxLayout()
        self.txt_testcases = QTextEdit()
        self.txt_testcases.setMaximumHeight(100)
        self.txt_testcases.setPlainText(
            "((0,), 0)\n"
            "((1,), 1)\n"
            "((2,), 2)\n"
            "((3,), 6)"
        )
        test_layout.addWidget(self.txt_testcases)

        self.btn_update_tests = QPushButton("Update Tests")
        self.btn_update_tests.clicked.connect(self.update_test_cases)
        test_layout.addWidget(self.btn_update_tests)

        test_group.setLayout(test_layout)
        right_layout.addWidget(test_group)

        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()

        self.lbl_generation = QLabel("Generation: 0")
        status_layout.addWidget(self.lbl_generation)

        self.lbl_fitness = QLabel("Best Fitness: 0/0")
        status_layout.addWidget(self.lbl_fitness)

        self.lbl_time = QLabel("Time: 0.000s")
        status_layout.addWidget(self.lbl_time)

        self.progress_fitness = QProgressBar()
        self.progress_fitness.setRange(0, 100)
        status_layout.addWidget(self.progress_fitness)

        status_group.setLayout(status_layout)
        right_layout.addWidget(status_group)

        layout.addWidget(right_panel, 1)
        self.update_mutation_label()

    def update_mutation_label(self):
        """Update mutation rate label."""
        val = self.slider_mutation.value() / 100.0
        self.lbl_mutation.setText(f"{val:.2f}")
        self.engine.mutation_rate = val

    def toggle_evolution(self):
        """Toggle evolution on/off."""
        if self.is_evolving:
            self.stop_evolution()
        else:
            self.start_evolution()

    def start_evolution(self):
        """Start the evolution process."""
        if self.is_evolving:
            return

        self.is_evolving = True
        self.btn_evolve.setText("⏸ Pause")
        self.engine.population_size = self.spin_pop.value()

        self.evolution_thread = threading.Thread(target=self.evolution_loop)
        self.evolution_thread.daemon = True
        self.evolution_thread.start()

        self.audio.speak("Evolution started. Generating new code.")

    def evolution_loop(self):
        """Main evolution loop running in thread."""
        while self.is_evolving:
            try:
                fitness, src = self.engine.evolve(self.test_cases, self.audio)
                self.update_status(fitness, src)
                self.code_editor.setPlainText(src)

                if fitness[0] >= len(self.test_cases):
                    self.audio.play_attractor_sound()
                    self.audio.speak("Attractor reached! Perfect solution found.")
                    self.stop_evolution()
                    break

                QApplication.processEvents()

            except Exception as e:
                print(f"Evolution error: {e}")
                self.audio.play_error_sound()
                break

    def stop_evolution(self):
        """Stop the evolution process."""
        self.is_evolving = False
        self.btn_evolve.setText("▶ Evolve")
        if self.evolution_thread:
            self.evolution_thread.join(timeout=0.1)
        self.audio.speak("Evolution stopped.")

    def reset_evolution(self):
        """Reset the evolution engine."""
        self.stop_evolution()
        self.engine = DrMoagiEvolutionEngine()
        self.engine.population_size = self.spin_pop.value()
        self.viewer.clear()
        self.update_status((0, 0, 0), "")
        self.code_editor.clear()
        self.audio.speak("Reset complete.")

    def speak_code(self):
        """Read the current code aloud."""
        code = self.code_editor.toPlainText()
        if code.strip():
            self.audio.speak(code)

    def update_status(self, fitness, src):
        """Update status displays."""
        self.current_fitness = fitness[0]
        total_tests = len(self.test_cases)

        self.lbl_generation.setText(f"Generation: {self.engine.generation}")
        self.lbl_fitness.setText(f"Best Fitness: {fitness[0]}/{total_tests}")
        self.lbl_time.setText(f"Time: {fitness[1]:.4f}s")

        progress = (fitness[0] / total_tests) * 100 if total_tests > 0 else 0
        self.progress_fitness.setValue(int(progress))

    def update_visualization(self):
        """Update 3D visualization."""
        points = self.engine.get_latent_points()
        self.viewer.update_swarm(points)

        if self.engine.best_ast:
            attractor = np.mean(points, axis=0) if len(points) > 0 else [0, 0, 0]
            self.viewer.update_attractor(attractor)

    def update_test_cases(self):
        """Update test cases from text input."""
        try:
            text = self.txt_testcases.toPlainText()
            cases = []
            for line in text.strip().split('\n'):
                if line.strip():
                    case = ast.literal_eval(line.strip())
                    if not (isinstance(case, tuple) and len(case) == 2):
                        raise ValueError("Each line must be ((args,), expected)")
                    args, result = case
                    if not isinstance(args, tuple):
                        args = (args,)
                    cases.append((args, result))
            if cases:
                self.test_cases = cases
                self.audio.speak(f"Updated to {len(cases)} test cases")
        except Exception as e:
            self.audio.play_error_sound()
            QMessageBox.warning(self, "Error", f"Failed to parse test cases: {e}")

    def closeEvent(self, event):
        """Clean up on close."""
        self.stop_evolution()
        self.audio.cleanup()
        event.accept()


# -----------------------------------------------------------------
# 5. MAIN APPLICATION
# -----------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)

    window = DrMoagiEditor()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
