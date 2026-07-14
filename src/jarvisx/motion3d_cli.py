"""Command-line interface for the Dr Moagi 3D motion engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

from .motion3d import (
    DrMoagiMotionEngine,
    MotionConstraints,
    MotionObservation,
    MotionState,
)


def _load_json(value: str):
    candidate = Path(value)
    raw = candidate.read_text(encoding="utf-8") if candidate.exists() else value
    return json.loads(raw)


def _tuple_field(payload: Dict[str, object], name: str, fallback):
    value = payload.get(name, fallback)
    if not isinstance(value, (list, tuple)):
        raise ValueError("%s must be a JSON array" % name)
    return tuple(float(item) for item in value)


def _state(payload: Dict[str, object]) -> MotionState:
    defaults = MotionState()
    return MotionState(
        position=_tuple_field(payload, "position", defaults.position),
        orientation=_tuple_field(payload, "orientation", defaults.orientation),
        velocity=_tuple_field(payload, "velocity", defaults.velocity),
        angular_velocity=_tuple_field(
            payload, "angular_velocity", defaults.angular_velocity
        ),
        acceleration=_tuple_field(payload, "acceleration", defaults.acceleration),
        angular_acceleration=_tuple_field(
            payload, "angular_acceleration", defaults.angular_acceleration
        ),
        force=_tuple_field(payload, "force", defaults.force),
        torque=_tuple_field(payload, "torque", defaults.torque),
        mass=float(payload.get("mass", defaults.mass)),
        inertia=_tuple_field(payload, "inertia", defaults.inertia),
        memory_position=_tuple_field(
            payload, "memory_position", defaults.memory_position
        ),
        memory_velocity=_tuple_field(
            payload, "memory_velocity", defaults.memory_velocity
        ),
        time_seconds=float(payload.get("time_seconds", defaults.time_seconds)),
        step_index=int(payload.get("step_index", defaults.step_index)),
    )


def _constraints(payload: Optional[Dict[str, object]]) -> MotionConstraints:
    defaults = MotionConstraints()
    if payload is None:
        return defaults
    floor_value = payload.get("floor_z", defaults.floor_z)
    return MotionConstraints(
        world_min=_tuple_field(payload, "world_min", defaults.world_min),
        world_max=_tuple_field(payload, "world_max", defaults.world_max),
        max_speed=float(payload.get("max_speed", defaults.max_speed)),
        max_acceleration=float(
            payload.get("max_acceleration", defaults.max_acceleration)
        ),
        max_angular_speed=float(
            payload.get("max_angular_speed", defaults.max_angular_speed)
        ),
        max_angular_acceleration=float(
            payload.get(
                "max_angular_acceleration", defaults.max_angular_acceleration
            )
        ),
        floor_z=None if floor_value is None else float(floor_value),
        restitution=float(payload.get("restitution", defaults.restitution)),
        max_dt=float(payload.get("max_dt", defaults.max_dt)),
    )


def _observations(value: Optional[str], steps: int):
    if value is None:
        return None
    payload = _load_json(value)
    if not isinstance(payload, list) or len(payload) != steps:
        raise ValueError("observations must be a JSON array matching --steps")
    result = []
    for item in payload:
        if item is None:
            result.append(None)
            continue
        if not isinstance(item, dict):
            raise ValueError("each observation must be an object or null")
        result.append(
            MotionObservation(
                position=(
                    _tuple_field(item, "position", (0.0, 0.0, 0.0))
                    if "position" in item
                    else None
                ),
                orientation=(
                    _tuple_field(item, "orientation", (1.0, 0.0, 0.0, 0.0))
                    if "orientation" in item
                    else None
                ),
                velocity=(
                    _tuple_field(item, "velocity", (0.0, 0.0, 0.0))
                    if "velocity" in item
                    else None
                ),
                angular_velocity=(
                    _tuple_field(item, "angular_velocity", (0.0, 0.0, 0.0))
                    if "angular_velocity" in item
                    else None
                ),
                confidence=float(item.get("confidence", 1.0)),
            )
        )
    return tuple(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="drmoagi-motion")
    parser.add_argument("state", help="JSON object or path to an initial state")
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--force", default="[0,0,0]")
    parser.add_argument("--torque", default="[0,0,0]")
    parser.add_argument("--constraints")
    parser.add_argument("--observations")
    parser.add_argument("--summary-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    state_payload = _load_json(args.state)
    if not isinstance(state_payload, dict):
        raise ValueError("state must be a JSON object")
    constraint_payload = _load_json(args.constraints) if args.constraints else None
    if constraint_payload is not None and not isinstance(constraint_payload, dict):
        raise ValueError("constraints must be a JSON object")
    force = _load_json(args.force)
    torque = _load_json(args.torque)
    if not isinstance(force, list) or not isinstance(torque, list):
        raise ValueError("force and torque must be JSON arrays")
    engine = DrMoagiMotionEngine(_constraints(constraint_payload))
    results = engine.run(
        _state(state_payload),
        args.dt,
        args.steps,
        external_force=tuple(float(value) for value in force),
        external_torque=tuple(float(value) for value in torque),
        observations=_observations(args.observations, args.steps),
    )
    output = {
        "equation": (
            "M[t+dt] = Pi_Lambda(K_dt(M[t]) + D_dt(F,tau) "
            "+ K*E_motion + DeltaOmega)"
        ),
        "steps": len(results),
        "final": results[-1].snapshot(),
    }
    if not args.summary_only:
        output["trajectory"] = [result.snapshot() for result in results]
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
