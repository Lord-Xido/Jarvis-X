"""CLI launcher for the NEXUS-3D same-origin workspace."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvisx-nexus3d",
        description="Run the bounded NEXUS-3D multimodal workspace.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--static-dir", type=Path)
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default="info",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be within [1, 65535]")
    if args.static_dir is not None:
        os.environ["JARVISX_NEXUS3D_STATIC_DIR"] = str(args.static_dir.resolve())
    uvicorn.run(
        "jarvisx.nexus3d_api:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
