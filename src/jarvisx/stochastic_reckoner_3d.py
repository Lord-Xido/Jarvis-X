"""Bounded 3D Bayesian belief and risk-constrained predictive planning.

Pure numerical reference; no actuator or hardware access.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Sequence

Point = tuple[int, int, int]
Belief = tuple[float, ...]
ACTIONS: tuple[Point, ...] = (
    (0, 0, 0), (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
)


@dataclass(frozen=True)
class Plan:
    actions: tuple[Point, ...]
    cost: float
    risk_upper: float


@dataclass(frozen=True)
class SearchResult:
    plan: Plan | None
    evaluations: int
    exhausted: bool


class BeliefGrid3D:
    """Finite bounded grid, with a move-or-stay stochastic transition."""

    def __init__(self, side: int, success: float = 0.8) -> None:
        if type(side) is not int or not 2 <= side <= 8:
            raise ValueError("side must be an integer in [2, 8]")
        if not math.isfinite(success) or not 0 <= success <= 1:
            raise ValueError("success probability outside [0, 1]")
        self.side = side
        self.success = success
        self.points: tuple[Point, ...] = tuple(product(range(side), repeat=3))
        self.index = {p: i for i, p in enumerate(self.points)}
        self.destinations = {
            a: tuple(
                self.index[(
                    min(side - 1, max(0, p[0] + a[0])),
                    min(side - 1, max(0, p[1] + a[1])),
                    min(side - 1, max(0, p[2] + a[2])),
                )]
                for p in self.points
            )
            for a in ACTIONS
        }

    def validate(self, belief: Sequence[float]) -> Belief:
        b = tuple(float(x) for x in belief)
        if len(b) != len(self.points) or any(
            not math.isfinite(x) or x < 0 for x in b
        ) or not math.isclose(math.fsum(b), 1.0, rel_tol=0, abs_tol=1e-10):
            raise ValueError("invalid probability field")
        return b

    def point_mass(self, p: Point) -> Belief:
        if p not in self.index:
            raise ValueError("point outside grid")
        return tuple(float(q == p) for q in self.points)

    def predict(self, belief: Sequence[float], action: Point) -> Belief:
        b = self.validate(belief)
        if action not in self.destinations:
            raise ValueError("invalid action")
        out = [0.0] * len(b)
        for i, j in enumerate(self.destinations[action]):
            out[i] += (1 - self.success) * b[i]
            out[j] += self.success * b[i]
        return self.validate(out)

    def observe(
        self, prior: Sequence[float], observation: Sequence[float], sigma: float
    ) -> Belief:
        """Gaussian likelihood update with stable log normalization."""
        b = self.validate(prior)
        o = tuple(float(x) for x in observation)
        if len(o) != 3 or any(not math.isfinite(x) for x in o):
            raise ValueError("invalid observation")
        if not math.isfinite(sigma) or not 1e-6 <= sigma <= 1e6:
            raise ValueError("invalid observation noise scale")
        logp = [
            math.log(m) - 0.5 * sum(((x - y) / sigma) ** 2 for x, y in zip(p, o))
            if m > 0 else -math.inf
            for p, m in zip(self.points, b)
        ]
        peak = max(logp)
        if not math.isfinite(peak):
            raise ValueError("unrepresentable observation likelihood")
        w = [math.exp(x - peak) for x in logp]
        total = math.fsum(w)
        return self.validate([x / total for x in w])

    def plan(
        self, belief: Sequence[float], goal: Point, forbidden: set[Point],
        *, horizon: int = 3, budget: int = 400, risk_limit: float = 0.05,
        discount: float = 0.95, action_cost: float = 0.01,
    ) -> SearchResult:
        """Search open-loop sequences; execute nothing.

        Risk is a conservative union bound on visiting any forbidden voxel,
        including the initial belief. Search budget counts transition calls.
        A truncated search never implies global optimality.
        """
        b = self.validate(belief)
        if goal not in self.index or not forbidden <= self.index.keys():
            raise ValueError("goal or forbidden point outside grid")
        if type(horizon) is not int or not 1 <= horizon <= 4:
            raise ValueError("invalid horizon")
        if type(budget) is not int or not 1 <= budget <= 3000:
            raise ValueError("invalid budget")
        if not math.isfinite(risk_limit) or not 0 <= risk_limit <= 1:
            raise ValueError("invalid risk limit")
        if not math.isfinite(discount) or not 0 < discount <= 1:
            raise ValueError("invalid discount")
        if not math.isfinite(action_cost) or not 0 <= action_cost <= 1e6:
            raise ValueError("invalid action cost")
        indices = tuple(sorted(self.index[p] for p in forbidden))
        distance = tuple(sum((x - y) ** 2 for x, y in zip(p, goal)) for p in self.points)

        def mass(field: Belief) -> float:
            return math.fsum(field[i] for i in indices)

        best: Plan | None = None
        evaluations = 0
        exhausted = False

        def visit(field: Belief, path: tuple[Point, ...], cost: float, risk: float) -> None:
            nonlocal best, evaluations, exhausted
            if risk > risk_limit:
                return
            if len(path) == horizon:
                if best is None or cost < best.cost:
                    best = Plan(path, cost, risk)
                return
            for action in ACTIONS:
                if evaluations >= budget:
                    exhausted = True
                    return
                nxt = self.predict(field, action)
                evaluations += 1
                step_cost = math.fsum(p * d for p, d in zip(nxt, distance))
                step_cost += action_cost * float(action != ACTIONS[0])
                visit(nxt, path + (action,), cost + discount ** len(path) * step_cost,
                      min(1.0, risk + mass(nxt)))

        visit(b, (), 0.0, mass(b))
        return SearchResult(best, evaluations, exhausted)
