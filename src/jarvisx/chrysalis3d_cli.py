"""CLI for the CHRYSALIS-Theta 3D runtime."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any, Dict

from .chrysalis3d import ChrysalisConfig, ChrysalisTheta3D, GridShape


def _load_json(value: str) -> Dict[str, Any]:
    if value.startswith("@"):
        with open(value[1:], "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    else:
        loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ValueError("input must be a JSON object")
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(prog="chrysalis-theta")
    parser.add_argument("input", help="JSON object or @file")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    payload = _load_json(args.input)
    raw_config = payload.get("config", {})
    if not isinstance(raw_config, dict):
        raise ValueError("config must be an object")
    grid = GridShape(
        width=int(raw_config.get("width", 4)),
        height=int(raw_config.get("height", 4)),
        depth=int(raw_config.get("depth", 4)),
    )
    config = ChrysalisConfig(
        grid=grid,
        d_model=int(raw_config.get("d_model", 512)),
        top_k=int(raw_config.get("top_k", 2)),
        expert_rank=int(raw_config.get("expert_rank", 8)),
        num_heads=int(raw_config.get("num_heads", 8)),
        seed=int(raw_config.get("seed", 0xC485A115)),
    )
    engine = ChrysalisTheta3D(config)

    sequence = payload.get("sequence")
    if sequence is None:
        sequence = [payload.get("modalities", {})]
    if not isinstance(sequence, list):
        raise ValueError("sequence must be an array")

    results = []
    for modalities in sequence:
        if not isinstance(modalities, dict):
            raise ValueError("each sequence item must be an object")
        result = engine.step(modalities)
        item = {
            "step": result.step,
            "state_hash": result.state_hash,
            "activations": [asdict(value) for value in result.activations],
            "arithmetic": asdict(result.arithmetic),
        }
        if not args.summary_only:
            item["output"] = list(result.output)
            item["modality_attention"] = [
                list(head) for head in result.modality_attention
            ]
        results.append(item)

    print(
        json.dumps(
            {"results": results, "snapshot": engine.snapshot()},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
