from __future__ import annotations

from jarvisx.operationalize import main, run_preflight, run_smoke


def test_operational_preflight_is_healthy() -> None:
    report = run_preflight()

    assert report.healthy
    assert report.mode == "doctor"
    assert all(check.ok for check in report.checks)


def test_operational_smoke_executes_transaction_os_and_api() -> None:
    report = run_smoke()
    by_name = {check.name: check for check in report.checks}

    assert report.healthy
    assert by_name["transactional-runtime"].ok
    assert by_name["dr-moagi-os-cycle"].ok
    assert by_name["control-plane-api"].ok


def test_operational_cli_returns_zero_for_healthy_smoke(capsys) -> None:
    exit_code = main(["smoke", "--json"])
    payload = capsys.readouterr().out

    assert exit_code == 0
    assert '"healthy": true' in payload
