import json

from jarvisx.multiparallel import FramedArtifact
from jarvisx.multiparallel_cli import main


def test_run_cli_emits_strict_summary_and_verifiable_artifact(tmp_path, capsys) -> None:
    source = "def value(x):\n    return x + 1\n" * 8
    input_path = tmp_path / "input.py"
    output_path = tmp_path / "output.jxmp"
    input_path.write_text(source, encoding="utf-8")

    status = main(
        (
            "run",
            str(input_path),
            "--workers",
            "2",
            "--batch-size",
            "16",
            "--output",
            str(output_path),
        )
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert status == 0
    assert payload["success"] is True
    assert payload["packages"] == 2
    assert output_path.exists()
    assert FramedArtifact.from_bytes(output_path.read_bytes()).decode() == source


def test_map_code_cli_reports_observation_without_source_rewrite(tmp_path, capsys) -> None:
    input_path = tmp_path / "input.py"
    input_path.write_text("answer = input_value + 1\n", encoding="utf-8")

    status = main(("map-code", str(input_path), "--axis-order", "yzx", "--scale", "2"))
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert status == 0
    assert payload["source_rewritten"] is False
    assert payload["axis_order"] == "yzx"
    assert payload["points"][0]["x"] == 0.0
    assert payload["points"][0]["z"] == 2.0


def test_cli_rejects_invalid_python_source_for_spatial_mapping(tmp_path, capsys) -> None:
    input_path = tmp_path / "broken.py"
    input_path.write_text("def broken(:\n", encoding="utf-8")

    status = main(("map-code", str(input_path)))
    captured = capsys.readouterr()

    assert status == 2
    assert "syntactically valid" in captured.err

