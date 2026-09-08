from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .neural_feedback_runtime import (
    JsonWebFeed,
    NeuralFeedbackRuntime,
    RuntimeConfig,
    SyntheticWebFeed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvisx-neural-feedback",
        description="Proof-gated live-web/simulation autoencoding runtime",
    )
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--feed-url", default=None, help="HTTP(S) JSON feed containing web observations")
    parser.add_argument(
        "--allow-domain",
        action="append",
        default=[],
        help="Optional domain allowlist entry for --feed-url; may be repeated",
    )
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--verification-threshold", type=float, default=0.58)
    parser.add_argument("--checkpoint", default="jarvisx-neural-feedback.npz")
    parser.add_argument("--resume", default=None, help="Load an existing runtime checkpoint")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", action="store_true", help="Emit one JSON report per cycle")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cycles < 1:
        raise SystemExit("--cycles must be >= 1")

    if args.resume:
        runtime = NeuralFeedbackRuntime.load_checkpoint(args.resume)
    else:
        runtime = NeuralFeedbackRuntime(
            RuntimeConfig(
                input_dim=args.input_dim,
                latent_dim=args.latent_dim,
                verification_threshold=args.verification_threshold,
                seed=args.seed,
            )
        )

    synthetic = SyntheticWebFeed(seed=args.seed)
    web = JsonWebFeed(
        max_bytes=runtime.config.max_feed_bytes,
        timeout_seconds=runtime.config.feed_timeout_seconds,
        allowed_domains=args.allow_domain,
    )

    for _ in range(args.cycles):
        observations = web.fetch(args.feed_url) if args.feed_url else synthetic.observations(runtime.cycle + 1)
        report = runtime.process(observations)
        if args.json:
            print(json.dumps(asdict(report), sort_keys=True))
        else:
            baseline = "-" if report.baseline_loss is None else f"{report.baseline_loss:.6f}"
            candidate = "-" if report.candidate_loss is None else f"{report.candidate_loss:.6f}"
            print(
                f"cycle={report.cycle} obs={report.observations} accepted={report.accepted_claims}/"
                f"{report.claims} replay={report.replay_size} verify={report.mean_verification:.3f} "
                f"loss={baseline}->{candidate} promoted={report.promoted} "
                f"generation={report.production_generation}"
            )

    path = runtime.save_checkpoint(args.checkpoint)
    if not args.json:
        print(f"checkpoint={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
