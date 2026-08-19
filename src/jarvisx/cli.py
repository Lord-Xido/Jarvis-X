from __future__ import annotations

import json
import sys
from typing import Any

from .api import start_api
from .assembler import Assembler
from .auto_codec_loop import AutoCodecLoop, AutoCodecLoopConfig, UniformQuantizedFieldCodec
from .core import CodexVM
from .dr_moagi_field_runtime import DrMoagiFieldConfig, DrMoagiFieldRuntime
from .node import CodexNode
from .parser import Parser
from .spatial_codec_3d import MortonQuantizedFieldCodec3D, SpatialAutoCodec3DSystem
from .web import start_web


def _usage() -> None:
    print("Usage: jarvisx [run|codec|codec3d|api|web|node] <file>")


def _load_json_object(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError("input must be a JSON object")
    return payload


def _field_from_payload(payload: dict[str, Any]) -> dict[tuple[int, int, int], float]:
    cells = payload.get("cells", [])
    if not isinstance(cells, list):
        raise ValueError("cells must be a JSON array")
    field: dict[tuple[int, int, int], float] = {}
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("each cell must be an object")
        coordinate = (int(cell["x"]), int(cell["y"]), int(cell["z"]))
        if coordinate in field:
            raise ValueError(f"duplicate sparse coordinate: {coordinate}")
        field[coordinate] = float(cell["value"])
    return field


def _configs(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    field_config = payload.get("field_config", {})
    loop_config = payload.get("loop_config", {})
    if not isinstance(field_config, dict) or not isinstance(loop_config, dict):
        raise ValueError("field_config and loop_config must be JSON objects")
    return field_config, loop_config


def _run_codec_file(path: str) -> None:
    payload = _load_json_object(path)
    field = _field_from_payload(payload)
    field_config, loop_config = _configs(payload)

    codec = UniformQuantizedFieldCodec(float(payload.get("quantization_step", 0.05)))
    runtime = DrMoagiFieldRuntime(codec, DrMoagiFieldConfig(**field_config))
    loop = AutoCodecLoop(runtime, AutoCodecLoopConfig(**loop_config))
    loop.load(field)
    print(json.dumps(loop.run().to_dict(), indent=2, sort_keys=True, allow_nan=False))


def _run_codec_3d_file(path: str) -> None:
    payload = _load_json_object(path)
    field = _field_from_payload(payload)
    field_config, loop_config = _configs(payload)
    spatial_config = payload.get("spatial_config", {})
    if not isinstance(spatial_config, dict):
        raise ValueError("spatial_config must be a JSON object")

    configured_field = DrMoagiFieldConfig(**field_config)
    codec = MortonQuantizedFieldCodec3D(
        float(payload.get("quantization_step", 0.05)),
        side=configured_field.side,
    )
    runtime = DrMoagiFieldRuntime(codec, configured_field)
    loop = AutoCodecLoop(runtime, AutoCodecLoopConfig(**loop_config))
    system = SpatialAutoCodec3DSystem(
        loop,
        codec,
        side=configured_field.side,
        **spatial_config,
    )
    summary = system.run(field)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True, allow_nan=False))


def main() -> None:
    if len(sys.argv) < 2:
        _usage()
        return

    cmd = sys.argv[1]

    if cmd == "run":
        if len(sys.argv) < 3:
            _usage()
            return
        with open(sys.argv[2], encoding="utf-8") as source:
            ast = Parser().parse(source.read())
        bytecode = Assembler().assemble(ast)
        vm = CodexVM()
        vm.load(bytecode)
        vm.run()
        print("Registers:", vm.regs.snapshot())

    elif cmd == "codec":
        if len(sys.argv) < 3:
            _usage()
            return
        _run_codec_file(sys.argv[2])

    elif cmd == "codec3d":
        if len(sys.argv) < 3:
            _usage()
            return
        _run_codec_3d_file(sys.argv[2])

    elif cmd == "api":
        start_api()

    elif cmd == "web":
        start_web()

    elif cmd == "node":
        CodexNode().start()

    else:
        _usage()


if __name__ == "__main__":
    main()
