import json
from pathlib import Path

from jarvisx.cli import main


def _write_fixture(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "shape": [2, 2, 2],
                "values": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            }
        ),
        encoding="utf-8",
    )


def test_codec_cli_roundtrip_and_decode(tmp_path: Path) -> None:
    source = tmp_path / "volume.json"
    bitstream = tmp_path / "volume.jx3d"
    reconstructed = tmp_path / "reconstructed.json"
    decoded = tmp_path / "decoded.json"
    _write_fixture(source)

    result = main(
        [
            "codec",
            str(source),
            "--bitstream",
            str(bitstream),
            "--reconstructed",
            str(reconstructed),
            "--quant-step",
            "0.25",
            "--virtual-depth",
            "1000000",
        ]
    )

    assert result == 0
    assert bitstream.read_bytes().startswith(b"JX3D")
    assert reconstructed.exists()

    decode_result = main(["codec-decode", str(bitstream), str(decoded)])
    assert decode_result == 0
    payload = json.loads(decoded.read_text(encoding="utf-8"))
    assert payload["shape"] == [2, 2, 2]
    assert len(payload["values"]) == 8


def test_cli_rejects_invalid_volume_json(tmp_path: Path) -> None:
    source = tmp_path / "invalid.json"
    source.write_text('{"shape": [2, 2], "values": []}', encoding="utf-8")

    assert main(["codec", str(source)]) == 1
