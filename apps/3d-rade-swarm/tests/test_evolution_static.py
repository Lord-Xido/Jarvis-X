from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "evolution.html"


class Parser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.scripts = 0

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.scripts += 1
        for key, value in attrs:
            if key == "id" and value:
                self.ids.add(value)


def main():
    text = HTML.read_text(encoding="utf-8")
    parser = Parser()
    parser.feed(text)

    required_ids = {
        "view", "toggle", "step", "reset", "grid", "pop", "mut", "eps",
        "mode", "gen", "fit", "fid", "spd", "nov", "val", "accepted",
    }
    missing = required_ids - parser.ids
    assert not missing, f"missing UI ids: {sorted(missing)}"
    assert parser.scripts >= 1, "runtime script missing"

    invariants = [
        "class Grid3D",
        "class Variant",
        "class MetaController",
        "targetDensity",
        "predictDensity",
        "validateScene",
        "microEvolve",
        "requestAnimationFrame",
        "F(P)^α",
        "Generate → Verify → Learn → Measure → Evolve → Deploy → Generate",
    ]
    for invariant in invariants:
        assert invariant in text, f"missing invariant: {invariant}"

    # Prevent accidental introduction of network dependencies into the standalone runtime.
    lowered = text.lower()
    assert "<script src=" not in lowered, "runtime must remain standalone"
    assert "fetch(" not in lowered, "runtime must not require network access"

    print("3D-RADE evolutionary runtime static checks passed")


if __name__ == "__main__":
    main()
