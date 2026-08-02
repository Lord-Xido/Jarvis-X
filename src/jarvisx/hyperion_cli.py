"""Command-line operations for the Hyperion audit engine and service."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence, cast

import uvicorn

from .hyperion import HyperionEngine, Observation
from .hyperion_service import (
    HyperionRuntimeSettings,
    build_evidence_bundle,
    create_hyperion_app,
    load_hyperion_config,
    load_score_model,
    observation_from_dict,
    verify_evidence_bundle,
)


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_observations(path: Path) -> list[Observation]:
    value = _read_json(path)
    if isinstance(value, dict):
        value = value.get("observations")
    if not isinstance(value, list) or not value:
        raise ValueError("input must be a non-empty JSON array or an observations object")
    observations: list[Observation] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("every observation must be a JSON object")
        observations.append(observation_from_dict(cast(dict[str, object], item)))
    return observations


def _engine(config: Path | None, model: Path | None) -> HyperionEngine:
    return HyperionEngine(
        config=load_hyperion_config(config),
        model=load_score_model(model),
    )


def _audit(args: argparse.Namespace) -> int:
    engine = _engine(args.config, args.model)
    observations = _load_observations(args.input)
    bundle = build_evidence_bundle(engine, observations)
    _write_json(args.output, bundle)
    report = cast(dict[str, object], bundle["report"])
    print(
        json.dumps(
            {
                "report_digest": bundle["report_digest"],
                "bundle_digest": bundle["bundle_digest"],
                "verified": True,
                "geometric_health_score": report["geometric_health_score"],
                "events": len(cast(list[object], report["points"])),
            },
            sort_keys=True,
        )
    )
    return 0


def _verify(args: argparse.Namespace) -> int:
    value = _read_json(args.bundle)
    if not isinstance(value, dict):
        raise ValueError("bundle must be a JSON object")
    verified, reason = verify_evidence_bundle(value, _engine(args.config, args.model))
    print(json.dumps({"verified": verified, "reason": reason}, sort_keys=True))
    return 0 if verified else 1


def _serve(args: argparse.Namespace) -> int:
    settings = HyperionRuntimeSettings(
        data_dir=args.data_dir,
        max_observations=args.max_observations,
        api_key=args.api_key,
        require_api_key=args.require_api_key,
    )
    application = create_hyperion_app(
        settings=settings,
        engine=_engine(args.config, args.model),
    )
    uvicorn.run(
        application,
        host=args.host,
        port=args.port,
        workers=1,
        proxy_headers=True,
        log_level=args.log_level,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hyperion", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="create a deterministic evidence bundle")
    audit.add_argument("input", type=Path)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--config", type=Path)
    audit.add_argument("--model", type=Path)
    audit.set_defaults(handler=_audit)

    verify = subparsers.add_parser("verify", help="replay and verify an evidence bundle")
    verify.add_argument("bundle", type=Path)
    verify.add_argument("--config", type=Path)
    verify.add_argument("--model", type=Path)
    verify.set_defaults(handler=_verify)

    serve = subparsers.add_parser("serve", help="run the Hyperion HTTP service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--data-dir", type=Path, default=Path("state/hyperion"))
    serve.add_argument("--max-observations", type=int, default=50_000)
    serve.add_argument("--api-key")
    serve.add_argument("--require-api-key", action="store_true")
    serve.add_argument("--config", type=Path)
    serve.add_argument("--model", type=Path)
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(handler=_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"hyperion: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
