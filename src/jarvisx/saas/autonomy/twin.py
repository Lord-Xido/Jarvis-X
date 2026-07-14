"""Deterministic enterprise digital twin and downside-aware scenario engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class EnterpriseState:
    cash_minor: int
    monthly_revenue_minor: int
    monthly_cost_minor: int
    receivables_minor: int
    pipeline_minor: int
    delivery_health: float
    finance_health: float
    governance_health: float
    churn_rate: float
    collection_rate: float

    def validate(self) -> None:
        for name in (
            "delivery_health",
            "finance_health",
            "governance_health",
            "churn_rate",
            "collection_rate",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError("%s must be inside [0, 1]" % name)


@dataclass(frozen=True)
class Scenario:
    name: str
    revenue_growth: float = 0.0
    cost_growth: float = 0.0
    churn_delta: float = 0.0
    collection_delta: float = 0.0
    pipeline_conversion: float = 0.0
    delivery_delta: float = 0.0
    governance_delta: float = 0.0
    one_off_cost_minor: int = 0


@dataclass(frozen=True)
class SimulationResult:
    scenario: str
    horizon_months: int
    terminal_cash_minor: int
    terminal_revenue_minor: int
    terminal_cost_minor: int
    cumulative_profit_minor: int
    survival_probability: float
    expected_shortfall_minor: int
    health: float
    trajectory: Tuple[Dict[str, float], ...]


def _halton(index: int, base: int) -> float:
    result = 0.0
    fraction = 1.0
    while index > 0:
        fraction /= base
        result += fraction * (index % base)
        index //= base
    return result


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


class DigitalTwin:
    """Runs deterministic quasi-Monte-Carlo simulations with no random global state."""

    def simulate(
        self,
        state: EnterpriseState,
        scenario: Scenario,
        *,
        horizon_months: int = 12,
        paths: int = 128,
    ) -> SimulationResult:
        state.validate()
        if horizon_months <= 0 or paths <= 0:
            raise ValueError("horizon and paths must be positive")
        terminal_cash: List[int] = []
        profits: List[int] = []
        representative: List[Dict[str, float]] = []
        representative_terminal = None
        for path in range(1, paths + 1):
            cash = float(state.cash_minor - scenario.one_off_cost_minor)
            revenue = float(state.monthly_revenue_minor)
            cost = float(state.monthly_cost_minor)
            receivables = float(state.receivables_minor)
            churn = _clip(state.churn_rate + scenario.churn_delta, 0, 1)
            collection = _clip(state.collection_rate + scenario.collection_delta, 0, 1)
            delivery = _clip(state.delivery_health + scenario.delivery_delta, 0, 1)
            governance = _clip(
                state.governance_health + scenario.governance_delta, 0, 1
            )
            cumulative_profit = 0.0
            path_trace: List[Dict[str, float]] = []
            for month in range(1, horizon_months + 1):
                demand_noise = (_halton(path * (month + 1), 2) - 0.5) * 0.12
                cost_noise = (_halton(path * (month + 1), 3) - 0.5) * 0.08
                collection_noise = (_halton(path * (month + 1), 5) - 0.5) * 0.06
                growth = scenario.revenue_growth + demand_noise - churn * 0.25
                revenue = max(0.0, revenue * (1.0 + growth))
                converted = (
                    state.pipeline_minor * scenario.pipeline_conversion / horizon_months
                )
                recognized = revenue + converted
                cost = max(0.0, cost * (1.0 + scenario.cost_growth + cost_noise))
                collected = (
                    receivables
                    * _clip(collection + collection_noise, 0, 1)
                    / horizon_months
                )
                profit = recognized + collected - cost
                cash += profit
                cumulative_profit += profit
                receivables = max(0.0, receivables + recognized * 0.15 - collected)
                health = (
                    (max(delivery, 1e-9) ** 0.4)
                    * (max(state.finance_health, 1e-9) ** 0.35)
                    * (max(governance, 1e-9) ** 0.25)
                )
                if path == 1:
                    path_trace.append(
                        {
                            "month": float(month),
                            "cash_minor": cash,
                            "revenue_minor": revenue,
                            "cost_minor": cost,
                            "health": health,
                        }
                    )
            terminal_cash.append(int(round(cash)))
            profits.append(int(round(cumulative_profit)))
            if path == 1:
                representative = path_trace
                representative_terminal = (revenue, cost)
        ordered = sorted(terminal_cash)
        cutoff_count = max(1, int(math.ceil(paths * 0.05)))
        tail = ordered[:cutoff_count]
        expected_shortfall = max(0, -int(round(sum(tail) / len(tail))))
        survival = sum(value >= 0 for value in terminal_cash) / paths
        revenue_terminal, cost_terminal = representative_terminal or (
            state.monthly_revenue_minor,
            state.monthly_cost_minor,
        )
        health = (
            (max(state.delivery_health + scenario.delivery_delta, 0.0) ** 0.4)
            * (max(state.finance_health, 0.0) ** 0.35)
            * (max(state.governance_health + scenario.governance_delta, 0.0) ** 0.25)
        )
        return SimulationResult(
            scenario=scenario.name,
            horizon_months=horizon_months,
            terminal_cash_minor=int(round(sum(terminal_cash) / paths)),
            terminal_revenue_minor=int(round(revenue_terminal)),
            terminal_cost_minor=int(round(cost_terminal)),
            cumulative_profit_minor=int(round(sum(profits) / paths)),
            survival_probability=survival,
            expected_shortfall_minor=expected_shortfall,
            health=health,
            trajectory=tuple(representative),
        )

    def rank(
        self,
        state: EnterpriseState,
        scenarios: Iterable[Scenario],
        *,
        risk_aversion: float = 1.0,
        horizon_months: int = 12,
        paths: int = 128,
    ) -> Tuple[SimulationResult, ...]:
        results = [
            self.simulate(
                state,
                scenario,
                horizon_months=horizon_months,
                paths=paths,
            )
            for scenario in scenarios
        ]
        return tuple(
            sorted(
                results,
                key=lambda result: (
                    result.cumulative_profit_minor
                    - risk_aversion * result.expected_shortfall_minor,
                    result.survival_probability,
                    result.health,
                ),
                reverse=True,
            )
        )
