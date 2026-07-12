import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "neural_core_gpu.html"


class NeuralCoreHTMLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = EXAMPLE.read_text(encoding="utf-8")
        modules = re.findall(
            r'<script type="module">\s*(.*?)\s*</script>',
            cls.source,
            flags=re.DOTALL,
        )
        if len(modules) != 1:
            raise AssertionError("expected exactly one JavaScript module")
        cls.module = modules[0]

    def test_webgl2_and_glsl3_contract_is_explicit(self):
        self.assertIn('getContext("webgl2")', self.module)
        self.assertIn("glslVersion: THREE.GLSL3", self.module)
        self.assertIn("gl_VertexID", self.module)

    def test_shader_interface_is_glsl3_compatible(self):
        self.assertIn("out vec3 vColor;", self.module)
        self.assertIn("in vec3 vColor;", self.module)
        self.assertIn("out vec4 outColor;", self.module)
        self.assertNotIn("varying vec3", self.module)
        self.assertNotIn("attribute float size", self.module)
        self.assertNotIn("gl_FragColor", self.module)

    def test_point_size_and_camera_denominator_are_bounded(self):
        self.assertIn("max(-mvPosition.z, 0.25)", self.module)
        self.assertRegex(
            self.module,
            r"gl_PointSize\s*=\s*clamp\([^;]+,\s*1\.0,\s*22\.0\);",
        )

    def test_morph_adaptation_is_a_bounded_first_order_response(self):
        self.assertIn("1.0 - Math.exp(-adaptationRate * delta)", self.module)
        self.assertIn(
            "adaptedMorph += (morphTarget - adaptedMorph) * response",
            self.module,
        )
        self.assertIn("uAdaptedMorph", self.module)

    def test_particle_budget_is_device_adaptive_and_bounded(self):
        for count in (100000, 250000, 600000, 1000000):
            self.assertIn(f"return {count};", self.module)
        self.assertIn("navigator.deviceMemory", self.module)
        self.assertIn("navigator.hardwareConcurrency", self.module)

    def test_particle_sizes_are_deterministic(self):
        self.assertIn("deterministicSize", self.module)
        self.assertNotIn("Math.random", self.module)

    def test_procedural_geometry_is_not_cpu_frustum_culled(self):
        self.assertIn("particleSystem.frustumCulled = false", self.module)
        self.assertIn("particleGeometry.boundingSphere", self.module)

    def test_quality_governor_has_upper_and_lower_bounds(self):
        self.assertIn("function governQuality(frameMs)", self.module)
        self.assertIn("Math.max(0.65, activePixelRatio - 0.1)", self.module)
        self.assertIn("Math.min(maxPixelRatio, activePixelRatio + 0.05)", self.module)
        self.assertIn("qualityCooldown = 120", self.module)

    def test_telemetry_labels_match_measured_quantities(self):
        self.assertIn("VISUAL PULSE", self.source)
        self.assertIn("FRAME TIME", self.source)
        self.assertIn("GPU TIME", self.source)
        self.assertNotIn("SYNAPTIC FIRE", self.source)
        self.assertNotIn("LATENCY:", self.source)
        self.assertIn("angularFrequency / (2 * Math.PI)", self.module)
        self.assertIn("EXT_disjoint_timer_query_webgl2", self.module)

    def test_visibility_and_context_loss_are_handled(self):
        self.assertIn('document.addEventListener("visibilitychange"', self.module)
        self.assertIn('"webglcontextlost"', self.module)
        self.assertIn('"webglcontextrestored"', self.module)

    def test_animation_loop_avoids_per_frame_bounding_sphere_rebuild(self):
        animate_match = re.search(
            r"function animate\(\)\s*\{(.*?)\n\s*\}",
            self.module,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(animate_match)
        self.assertNotIn("computeBoundingSphere", animate_match.group(1))

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for syntax validation")
    def test_javascript_module_parses(self):
        with tempfile.TemporaryDirectory() as directory:
            module_path = Path(directory) / "neural_core_gpu.mjs"
            module_path.write_text(self.module, encoding="utf-8")
            completed = subprocess.run(
                ["node", "--check", str(module_path)],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
