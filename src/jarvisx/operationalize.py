"""Repository-wide operational preflight, smoke verification, and service launcher."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Sequence


@dataclass(frozen=True)
class OperationalCheck:
    """One bounded operational verification result."""

    name: str
    ok: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass(frozen=True)
class OperationalReport:
    """Aggregate operational verification receipt."""

    mode: str
    checks: tuple[OperationalCheck, ...]

    @property
    def healthy(self) -> bool:
        return all(check.ok for check in self.checks)

    def as_dict(self) -> dict[str, object]:
        return {
            "system": "Jarvis-X",
            "mode": self.mode,
            "healthy": self.healthy,
            "checks": [check.as_dict() for check in self.checks],
        }


def _capture(name: str, operation: Callable[[], str]) -> OperationalCheck:
    try:
        detail = operation()
    except Exception as exc:
        return OperationalCheck(name=name, ok=False, detail=f"{type(exc).__name__}: {exc}")
    return OperationalCheck(name=name, ok=True, detail=detail)


def run_preflight() -> OperationalReport:
    """Verify that the canonical Python operational surfaces are importable."""

    modules = (
        "jarvisx.core",
        "jarvisx.system_runtime",
        "jarvisx.dr_moagi_os",
        "jarvisx.dr_moagi_meta_optimizer",
        "jarvisx.dr_moagi_system_evolution",
        "jarvisx.dr_moagi_os_api",
        "jarvisx.nexus3d_api",
    )

    checks = tuple(
        _capture(
            f"import:{module_name}",
            lambda module_name=module_name: (
                import_module(module_name).__name__ + " importable"
            ),
        )
        for module_name in modules
    )
    return OperationalReport(mode="doctor", checks=checks)


def _transaction_smoke() -> str:
    from .assembler import Assembler
    from .parser import Parser
    from .system_runtime import CAP_VM_EXECUTE, ExecutionRequest, ResourceBudget, SystemRuntime

    program = tuple(
        Assembler().assemble(
            Parser().parse(
                "SET A 19\n"
                "SET B 23\n"
                "ADD Ψ A B\n"
                "HALT"
            )
        )
    )
    runtime = SystemRuntime()
    receipt = runtime.execute(
        ExecutionRequest(
            request_id="jarvisx-operational-smoke-v1",
            program=program,
            granted_capabilities=frozenset({CAP_VM_EXECUTE}),
            budget=ResourceBudget(max_cycles=128, max_program_words=64, max_candidates=4),
        )
    )
    if not receipt.committed:
        raise RuntimeError(f"transaction was not committed: {receipt.error_message}")
    if receipt.state_dict().get("Ψ") != 42:
        raise RuntimeError("transactional VM returned an unexpected state")
    if not runtime.verify():
        raise RuntimeError("transactional runtime verification failed")
    return f"commit={receipt.state_hash} cycles={receipt.cycles}"


def _os_smoke() -> str:
    from .dr_moagi_os import DrMoagiOSConfig, DrMoagiOSKernel, demo_field

    kernel = DrMoagiOSKernel(
        DrMoagiOSConfig(
            side=8,
            max_active_cells=512,
            deep_distiller_max_latent_cells=256,
            fixed_point_passes=1,
            state_dir=None,
        )
    )
    kernel.boot(restore=False)
    kernel.load(demo_field(8))
    report = kernel.step()
    if not report.committed:
        raise RuntimeError(report.rejection_reason or "Dr Moagi OS cycle rejected")
    status = kernel.status()
    if status.get("journal_valid") is not True:
        raise RuntimeError("Dr Moagi OS journal verification failed")
    return (
        f"cycle={status.get('cycle')} active={status.get('active_cells')} "
        f"state={status.get('state_hash')}"
    )


def _api_smoke() -> str:
    from .dr_moagi_os_api import app

    required = {
        "/healthz",
        "/v1/os/capabilities",
        "/v1/os/boot",
        "/v1/os/load",
        "/v1/os/step",
        "/v1/os/run",
        "/v1/os/status",
        "/metrics",
    }
    available = {getattr(route, "path", "") for route in app.routes}
    missing = sorted(required - available)
    if missing:
        raise RuntimeError("missing control-plane routes: " + ", ".join(missing))
    return f"{len(required)} required control-plane routes present"


def run_smoke() -> OperationalReport:
    """Execute bounded end-to-end checks without external network dependencies."""

    preflight = run_preflight()
    checks = list(preflight.checks)
    checks.extend(
        (
            _capture("transactional-runtime", _transaction_smoke),
            _capture("dr-moagi-os-cycle", _os_smoke),
            _capture("control-plane-api", _api_smoke),
        )
    )
    return OperationalReport(mode="smoke", checks=tuple(checks))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvisx-operationalize",
        description="Prove and launch the bounded Jarvis-X operational control plane.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="verify canonical runtime imports")
    doctor.add_argument("--json", action="store_true", dest="as_json")

    smoke = sub.add_parser("smoke", help="run deterministic end-to-end smoke verification")
    smoke.add_argument("--json", action="store_true", dest="as_json")

    serve = sub.add_parser("serve", help="serve the authoritative Dr Moagi OS control plane")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=10000)
    return parser


def _print_report(report: OperationalReport, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report.as_dict(), sort_keys=True))
        return

    verdict = "HEALTHY" if report.healthy else "FAILED"
    print(f"Jarvis-X {report.mode}: {verdict}")
    for check in report.checks:
        marker = "PASS" if check.ok else "FAIL"
        print(f"[{marker}] {check.name}: {check.detail}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "serve":
        if args.port <= 0 or args.port > 65535:
            raise SystemExit("--port must be in [1, 65535]")
        import uvicorn

        uvicorn.run(
            "jarvisx.dr_moagi_os_api:app",
            host=str(args.host),
            port=int(args.port),
            log_level="info",
        )
        return 0

    report = run_preflight() if args.command == "doctor" else run_smoke()
    _print_report(report, as_json=bool(args.as_json))
    return 0 if report.healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
