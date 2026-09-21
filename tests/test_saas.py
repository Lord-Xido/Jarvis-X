from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from sqlalchemy import select

from jarvisx.saas.billing import BillingLine, calculate_totals, subscription_lines
from jarvisx.saas.db import Database, Invoice, InvoiceLine, UsageEvent
from jarvisx.saas.geometry import engagement_point, enterprise_centroid
from jarvisx.saas.security import Principal, TokenService
from jarvisx.saas.service import ConsultancyService


def make_service(tmp_path):
    return ConsultancyService(Database("sqlite:///%s" % (tmp_path / "saas.db")))


def bootstrap(service):
    ids = service.bootstrap(
        "Dr Moagi Software Consultancy",
        "dr-moagi",
        "Dr Moagi Software Consultancy (Pty) Ltd",
        "Platform Admin",
        "admin@example.com",
        "correct-horse-battery-staple",
        tax_rate_bps=1500,
    )
    return Principal(ids["user_id"], ids["tenant_id"], "platform_admin")


def test_subscription_formula_and_tax():
    lines = subscription_lines(
        10000,
        5,
        2,
        1000,
        {"api_calls": Decimal("250")},
        {"api_calls": Decimal("100")},
        {"api_calls": 10},
    )
    totals = calculate_totals(lines, 1500)
    assert totals.subtotal_minor == 14500
    assert totals.tax_minor == 2175
    assert totals.total_minor == 16675


def test_consultancy_invoice_and_payment(tmp_path):
    service = make_service(tmp_path)
    principal = bootstrap(service)
    client = service.create_client(principal, "Acme", "accounts@acme.test")
    engagement = service.create_engagement(
        principal, client.id, "Cloud Transformation", 500000, 120000
    )
    service.record_time(principal, engagement.id, 90, "Architecture", approved=True)
    service.record_expense(principal, engagement.id, 10000, "Travel", approved=True)
    start = datetime.now(timezone.utc) - timedelta(days=1)
    end = datetime.now(timezone.utc) + timedelta(days=1)
    invoice = service.generate_consultancy_invoice(principal, client.id, start, end)
    assert invoice.subtotal_minor == 190000
    assert invoice.tax_minor == 28500
    assert invoice.total_minor == 218500
    payment = service.record_payment(principal, invoice.id, invoice.total_minor)
    assert payment.amount_minor == 218500
    with service.db.session() as session:
        reloaded = session.get(Invoice, invoice.id)
        assert reloaded.status == "paid"
        assert reloaded.balance_minor == 0
        assert (
            len(
                session.scalars(
                    select(InvoiceLine).where(InvoiceLine.invoice_id == invoice.id)
                ).all()
            )
            == 2
        )


def test_metered_platform_invoice_is_idempotent(tmp_path):
    service = make_service(tmp_path)
    principal = bootstrap(service)
    plan = service.create_plan(
        principal,
        "enterprise",
        "Enterprise",
        50000,
        3,
        5000,
        {"abstract3d_cycles": 100},
        {"abstract3d_cycles": 20},
    )
    subscription = service.subscribe(principal, plan.id, seats=5)
    service.record_usage(principal, "abstract3d_cycles", 130, "evt-1")
    with pytest.raises(ValueError):
        service.record_usage(principal, "abstract3d_cycles", 130, "evt-1")
    invoice = service.generate_platform_invoice(principal, subscription.id)
    assert invoice.subtotal_minor == 60600
    with service.db.session() as session:
        event = session.scalar(
            select(UsageEvent).where(UsageEvent.idempotency_key == "evt-1")
        )
        assert event.invoiced_invoice_id == invoice.id


def test_tenant_isolation(tmp_path):
    service = make_service(tmp_path)
    principal = bootstrap(service)
    client = service.create_client(principal, "Tenant One")
    outsider = Principal("other-user", "other-tenant", "tenant_owner")
    with pytest.raises(LookupError):
        service.create_engagement(outsider, client.id, "Illegal access", 1, 1)


def test_geometry_penalizes_risk_and_dispersion():
    healthy = engagement_point(
        {
            "progress": 0.9,
            "quality": 0.95,
            "schedule": 0.9,
            "budget_utilization": 0.4,
            "collection_ratio": 1.0,
            "margin_ratio": 0.7,
            "risk": 0.05,
            "governance": 0.95,
        }
    )
    weak = engagement_point(
        {
            "progress": 0.2,
            "quality": 0.5,
            "budget_utilization": 1.0,
            "collection_ratio": 0.2,
            "margin_ratio": 0.1,
            "risk": 0.8,
            "governance": 0.3,
        }
    )
    portfolio = enterprise_centroid([(healthy, 100), (weak, 100)])
    assert healthy.health > weak.health
    assert healthy.risk < weak.risk
    assert portfolio["dispersion"] > 0


def test_signed_token_round_trip(monkeypatch):
    monkeypatch.setenv("DM_TOKEN_SECRET", "a" * 48)
    service = TokenService(ttl_seconds=60)
    principal = Principal("u1", "t1", "finance_admin")
    assert service.verify(service.issue(principal)) == principal
