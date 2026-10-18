"""Dr Moagi 3D geometric virtual-address codec (bounded reference).

Maps each coordinate in [-1, 1] to a symbolic axis whose magnitude is
exp(A*abs(x))-1, A = log((10**24)**(10**24)). No huge integer power or
virtual cell is ever allocated. A signed *triple-log chart* is serialized.

The included GridTokenCodec is a deterministic *lexical* 3D codebook, not a
trained semantic model or a neural network. A compatible learned codec may be
injected. A pooled sentence point is not invertible: exact text reconstruction
requires the ordered virtual token cloud and the same codec vocabulary.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

SEPTILLION = 10**24
EXPONENT10_PER_AXIS = 24 * SEPTILLION
LOG_VIRTUAL_AXIS = float(SEPTILLION) * (24.0 * math.log(10.0))
TRIPLE_LOG_LIMIT = math.log1p(math.log1p(LOG_VIRTUAL_AXIS))
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]|\s+", re.UNICODE)


def _point(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError("expected exactly three coordinates")
    point = tuple(float(value) for value in values)
    if not all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in point):
        raise ValueError("3D chart coordinates must be finite and within [-1, 1]")
    return point  # type: ignore[return-value]


def axis_to_virtual(value: float) -> dict[str, float | int]:
    """Compact, injective float64-level representation of a virtual axis."""
    value = _point((value, 0.0, 0.0))[0]
    sign = (value > 0) - (value < 0)
    chart = math.log1p(math.log1p(LOG_VIRTUAL_AXIS * abs(value)))
    return {"sign": sign, "triple_log": chart}


def axis_from_virtual(data: dict[str, Any]) -> float:
    if not isinstance(data, dict):
        raise ValueError("virtual axis must be an object")
    sign = data.get("sign")
    chart = data.get("triple_log")
    if type(sign) is not int or sign not in (-1, 0, 1):
        raise ValueError("virtual axis sign must be -1, 0 or 1")
    if not isinstance(chart, (float, int)) or not math.isfinite(float(chart)):
        raise ValueError("virtual axis triple_log must be finite")
    chart = float(chart)
    if not 0.0 <= chart <= TRIPLE_LOG_LIMIT + 1e-12:
        raise ValueError("virtual axis outside declared extent")
    if sign == 0:
        if chart != 0.0:
            raise ValueError("zero sign requires zero logarithmic magnitude")
        return 0.0
    if chart == 0.0:
        raise ValueError("nonzero sign requires positive magnitude")
    magnitude = math.expm1(math.expm1(chart)) / LOG_VIRTUAL_AXIS
    if not 0.0 < magnitude <= 1.0 + 1e-12:
        raise ValueError("virtual axis decoded outside chart")
    return sign * min(magnitude, 1.0)


def format_virtual_coord(point: Sequence[float]) -> dict[str, Any]:
    p = _point(point)
    axes = [axis_to_virtual(v) for v in p]
    return {
        "axes": axes,
        "coordinate_notation": [
            "0" if v == 0.0 else f"{'+' if v > 0 else '-'}(10^({EXPONENT10_PER_AXIS:.4e}*{abs(v):.17g})-1)"
            for v in p
        ],
        "encoding": "signed-triple-log-v1",
    }


def restore_virtual_coord(value: dict[str, Any]) -> tuple[float, float, float]:
    if not isinstance(value, dict) or value.get("encoding") != "signed-triple-log-v1":
        raise ValueError("unknown or missing virtual coordinate encoding")
    axes = value.get("axes")
    if not isinstance(axes, list) or len(axes) != 3:
        raise ValueError("exactly three virtual axes required")
    return _point([axis_from_virtual(axis) for axis in axes])


@dataclass(frozen=True)
class EncodedText:
    tokens: list[str]
    points: list[tuple[float, float, float]]
    sentence_point: tuple[float, float, float]


@dataclass(frozen=True)
class DecodedText:
    tokens: list[str]
    text: str


class GridTokenCodec:
    """Reversible, finite 3D *lexical* codebook; no learned semantic claims."""

    def __init__(self, max_tokens: int = 4096, vocabulary: Sequence[str] = ()) -> None:
        if type(max_tokens) is not int or not 1 <= max_tokens <= 4096:
            raise ValueError("max_tokens must be in [1, 4096]")
        self.max_tokens = max_tokens
        self.side = math.ceil(max_tokens ** (1.0 / 3.0))
        while self.side**3 < max_tokens:
            self.side += 1
        self.vocab: list[str] = []
        self.ids: dict[str, int] = {}
        for token in vocabulary:
            self._register(token)

    def _register(self, token: str) -> int:
        if token not in self.ids:
            if len(self.vocab) == self.max_tokens:
                raise ValueError("lexical codebook capacity exceeded")
            self.ids[token] = len(self.vocab)
            self.vocab.append(token)
        return self.ids[token]

    def _id_to_point(self, index: int) -> tuple[float, float, float]:
        side = self.side
        x, remainder = divmod(index, side * side)
        y, z = divmod(remainder, side)
        if side == 1:
            return (0.0, 0.0, 0.0)
        return (2.0 * x / (side - 1) - 1.0, 2.0 * y / (side - 1) - 1.0,
                2.0 * z / (side - 1) - 1.0)

    def encode(self, text: str) -> EncodedText:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        tokens = TOKEN_PATTERN.findall(text)
        points = [self._id_to_point(self._register(token)) for token in tokens]
        center = _point([sum(p[i] for p in points) / len(points) for i in range(3)]) if points else (0.0, 0.0, 0.0)
        return EncodedText(tokens=tokens, points=points, sentence_point=center)

    def decode(self, points: Sequence[Sequence[float]], method: str = "cosine") -> DecodedText:
        if method not in ("cosine", "nearest"):
            raise ValueError("unsupported decoding method")
        tokens = []
        for point in points:
            p = _point(point)
            # Cartesian Euclidean nearest neighbour on a fixed, uniform grid.
            if self.side == 1:
                index = 0
            else:
                coords = [min(self.side - 1, max(0, round((v + 1) * (self.side - 1) / 2))) for v in p]
                index = coords[0] * self.side**2 + coords[1] * self.side + coords[2]
            if index >= len(self.vocab):
                raise ValueError("coordinate references an unknown token; restore matching vocabulary")
            tokens.append(self.vocab[index])
        return DecodedText(tokens=tokens, text="".join(tokens))


class FractalOctreeNode:
    """Sparse point-index octree; occupied children only, redistribute on split."""

    def __init__(self, center: Sequence[float] = (0, 0, 0), half_size: float = 1.0,
                 depth: int = 0, max_depth: int = 10, capacity: int = 8) -> None:
        self.center = _point(center)
        if not math.isfinite(half_size) or half_size <= 0:
            raise ValueError("half_size must be positive and finite")
        if not 0 <= depth <= max_depth <= 32 or capacity < 1:
            raise ValueError("invalid octree depth or capacity")
        self.half_size = half_size
        self.depth = depth
        self.max_depth = max_depth
        self.capacity = capacity
        self.children: dict[int, FractalOctreeNode] = {}
        self.samples: list[tuple[float, float, float]] = []
        self.token_count = 0

    def contains(self, point: Sequence[float]) -> bool:
        return all(abs(p - c) <= self.half_size for p, c in zip(point, self.center))

    def _route(self, p: tuple[float, float, float]) -> None:
        key = sum((1 << axis) if p[axis] >= self.center[axis] else 0 for axis in range(3))
        child = self.children.get(key)
        if child is None:
            half = self.half_size / 2.0
            center = [self.center[i] + (half if key & (1 << i) else -half) for i in range(3)]
            child = FractalOctreeNode(center, half, self.depth + 1, self.max_depth, self.capacity)
            self.children[key] = child
        child.insert(p)

    def insert(self, point: Sequence[float]) -> bool:
        p = _point(point)
        if not self.contains(p):
            return False
        self.token_count += 1
        if self.children:
            self._route(p)
        elif self.depth == self.max_depth or len(self.samples) < self.capacity:
            self.samples.append(p)
        else:
            old = self.samples
            self.samples = []
            for prior in old:
                self._route(prior)
            self._route(p)
        return True

    def count_nodes(self) -> int:
        return 1 + sum(node.count_nodes() for node in self.children.values())

    def count_leaves(self) -> int:
        return sum(node.count_leaves() for node in self.children.values()) if self.children else 1

    def validate_counts(self) -> bool:
        if not self.children:
            return self.token_count == len(self.samples)
        return not self.samples and self.token_count == sum(c.token_count for c in self.children.values()) and all(c.validate_counts() for c in self.children.values())


class MoagiGeometricIntelligence:
    """Bounded reference: local 3D codec + virtual chart + indexed sparse points."""

    def __init__(self, base: Any | None = None, *, decoder_gain: float = 1.0,
                 max_depth: int = 10) -> None:
        if not math.isfinite(decoder_gain) or not 0.0 < decoder_gain <= 1.0:
            raise ValueError("decoder_gain must be in (0, 1]")
        self.base = base if base is not None else GridTokenCodec()
        self.decoder_gain = decoder_gain
        self.octree = FractalOctreeNode(max_depth=max_depth)
        self.evolution_log: list[dict[str, float | int | bool]] = []

    def encode_to_virtual(self, text: str) -> dict[str, Any]:
        encoding = self.base.encode(text)
        sentence = _point(encoding.sentence_point)
        points = [_point(point) for point in encoding.points]
        for p in points:
            self.octree.insert(p)
        return {
            "format": "moagi-virtual-token-cloud-v1",
            "virtual_sentence": format_virtual_coord(sentence),
            "virtual_tokens": [format_virtual_coord(p) for p in points],
            "token_count": len(points),
            "octree_nodes": self.octree.count_nodes(),
            "virtual_extent": "each axis up to 10^(24*10^24)-1; full cube symbolic only",
        }

    def _decode_points(self, payload: dict[str, Any], gain: float) -> list[tuple[float, float, float]]:
        if not isinstance(payload, dict) or payload.get("format") != "moagi-virtual-token-cloud-v1":
            raise ValueError("full token-cloud payload with version is required")
        tokens = payload.get("virtual_tokens")
        if not isinstance(tokens, list) or len(tokens) != payload.get("token_count"):
            raise ValueError("missing or inconsistent ordered virtual token cloud")
        return [_point([gain * p for p in restore_virtual_coord(v)]) for v in tokens]

    def decode_from_virtual(self, payload: dict[str, Any]) -> dict[str, Any]:
        points = self._decode_points(payload, self.decoder_gain)
        if isinstance(self.base, GridTokenCodec):
            result = self.base.decode(points, method="nearest")
        else:
            # Learned codecs may expect an ndarray and cosine decoding. NumPy
            # is only imported when such a codec is injected by the caller.
            import numpy as np

            result = self.base.decode(np.asarray(points, dtype=float), method="cosine")
        return {"decoded_text": result.text, "tokens": list(result.tokens),
                "real_points_used": [list(p) for p in points]}

    def auto_evolve(self, texts: Sequence[str], iterations: int = 5,
                    learning_rate: float = 0.5) -> list[dict[str, float | int | bool]]:
        """Measured CTR-gated scalar geometry calibration; not neural training.

        Optimizes gain on actual ordered token-point reconstruction loss, accepting
        only steps that strictly improve or match the measured objective.
        """
        if type(iterations) is not int or not 0 <= iterations <= 1000:
            raise ValueError("iterations must be in [0, 1000]")
        if not math.isfinite(learning_rate) or not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        pairs: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
        for text in texts:
            e = self.base.encode(text)
            true_points = [_point(p) for p in e.points]
            virtual = [format_virtual_coord(p) for p in true_points]
            pairs.extend((p, restore_virtual_coord(v)) for p, v in zip(true_points, virtual))
            for p in true_points:
                self.octree.insert(p)
        if not pairs:
            raise ValueError("a nonempty token training set is required")

        def loss(gain: float) -> float:
            return sum(sum((gain * q[i] - p[i]) ** 2 for i in range(3)) for p, q in pairs) / (3 * len(pairs))

        self.evolution_log = []
        for step in range(iterations):
            before = loss(self.decoder_gain)
            gradient = 2.0 * sum(sum((self.decoder_gain * q[i] - p[i]) * q[i] for i in range(3)) for p, q in pairs) / (3 * len(pairs))
            proposal = min(1.0, max(1e-9, self.decoder_gain - learning_rate * gradient))
            after = loss(proposal)
            accepted = after <= before + 1e-24
            if accepted:
                self.decoder_gain = proposal
            self.evolution_log.append({"iteration": step, "mse_before": before,
                                       "mse_after": loss(self.decoder_gain),
                                       "decoder_gain": self.decoder_gain,
                                       "accepted": accepted,
                                       "octree_nodes": self.octree.count_nodes()})
        return list(self.evolution_log)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", action="append", help="input text; repeat for multiple sentences")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--decoder-gain", type=float, default=1.0)
    parser.add_argument("--output", type=Path, help="optional JSON manifest path")
    args = parser.parse_args(argv)
    texts = args.text or ["the king sees the crown", "the knight holds the sword", "Dr Moagi maps virtual space"]
    engine = MoagiGeometricIntelligence(decoder_gain=args.decoder_gain)
    encoded = [engine.encode_to_virtual(text) for text in texts]
    log = engine.auto_evolve(texts, iterations=args.iterations)
    results = [engine.decode_from_virtual(payload) for payload in encoded]
    round_trip = [a == b["decoded_text"] for a, b in zip(texts, results)]
    octree_valid = engine.octree.validate_counts()
    report = {"engine": "Dr Moagi 3D geometric reference", "scope": "virtual coordinates + lexical token codec",
              "physical_allocation": "only observed token points and sparse octree nodes",
              "virtual_axis": "10^(24*10^24)-1", "virtual_cube_cells": "symbolic, not allocated",
              "token_round_trip": round_trip,
              "octree_nodes": engine.octree.count_nodes(), "octree_counts_valid": octree_valid,
              "evolution": log, "vocabulary": engine.base.vocab, "texts": texts,
              "decoded": [r["decoded_text"] for r in results]}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if all(round_trip) and octree_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
