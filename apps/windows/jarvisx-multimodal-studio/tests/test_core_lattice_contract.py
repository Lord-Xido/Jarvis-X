from __future__ import annotations

import re
import unittest
from pathlib import Path


APP = Path(__file__).resolve().parents[1]
FRAGMENTS = APP / "src" / "fragments" / "index.html"
CORE_RUNTIME = FRAGMENTS / "04a-core-runtime.part"


def assembled_html() -> str:
    return "".join(path.read_text(encoding="utf-8") for path in sorted(FRAGMENTS.glob("*.part")))


class CoreLatticeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = assembled_html()
        cls.core = CORE_RUNTIME.read_text(encoding="utf-8")

    def test_core_view_is_integrated_into_existing_multimodal_client(self) -> None:
        self.assertIn('coreNav.dataset.view="core"', self.core)
        self.assertIn('coreView.id="view-core"', self.core)
        self.assertIn('document.querySelector(".content").appendChild(coreView)', self.core)

    def test_telemetry_classes_are_explicit(self) -> None:
        for marker in ("AUTHORITATIVE:", "MEASURED:", "SIMULATED:", "SYMBOLIC:"):
            self.assertIn(marker, self.core)
        self.assertIn("SYMBOLIC TOPOLOGY", self.core)
        self.assertIn("not physically instantiated cores", self.core)

    def test_browser_does_not_embed_direct_provider_key_or_gemini_transport(self) -> None:
        forbidden = (
            "generativelanguage.googleapis.com",
            "gemini-2.5-flash",
            "const apiKey",
            "?key=${apiKey}",
            "cdnjs.cloudflare.com/ajax/libs/three.js",
        )
        for marker in forbidden:
            self.assertNotIn(marker, self.core)

    def test_core_uses_existing_loopback_chat_path(self) -> None:
        self.assertIn("/api/chat", self.core)
        self.assertIn('const api=(path,opts={})=>fetch(path', self.html)
        self.assertIn('"X-Jarvis-Token":launchToken', self.html)

    def test_q16_multiply_uses_widened_integer_intermediate(self) -> None:
        self.assertIn("BigInt(a)*BigInt(b)", self.core)
        self.assertIn(">>16n", self.core)
        self.assertIn("CORE_MIN=-2147483648", self.core)
        self.assertIn("CORE_MAX=2147483647", self.core)

    def test_recurrent_state_is_dual_bank_ping_pong(self) -> None:
        self.assertRegex(
            self.core,
            re.compile(r"this\.a=new Int32Array\(dim\);this\.b=new Int32Array\(dim\)"),
        )
        self.assertIn("this.active=this.a;this.next=this.b", self.core)
        self.assertIn("this.active=this.next;this.next=old", self.core)

    def test_local_stability_check_is_narrowly_described(self) -> None:
        self.assertIn('stable=r.stability.after<=r.stability.before+1e-12', self.core)
        self.assertIn("NON-EXPANSIVE ✓", self.core)
        self.assertIn("local non-expansion proxy", self.core)
        self.assertNotIn("Lyapunov Stability: CERTIFIED", self.core)

    def test_visual_geometry_is_bounded_and_reproducible(self) -> None:
        self.assertIn("for(let i=0;i<900;i++)", self.core)
        self.assertIn("let seed=0x51A7C0DE", self.core)
        self.assertIn("900 nodes", self.core)

    def test_unsupported_physical_claims_are_absent(self) -> None:
        forbidden = (
            "ALL CORES ACTIVE",
            "4.096 TB/s",
            "zero inductive ringing",
            "electron hardware register",
            "CERTIFIED ✓",
        )
        for marker in forbidden:
            self.assertNotIn(marker, self.core)

    def test_external_script_dependency_is_not_reintroduced(self) -> None:
        external_scripts = re.findall(r'<script\s+[^>]*src=["\']https?://', self.html, flags=re.I)
        self.assertEqual([], external_scripts)


if __name__ == "__main__":
    unittest.main()
