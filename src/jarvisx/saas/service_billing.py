"""Subscription, metering, invoicing, and payment operations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from .billing import BillingLine, calculate_totals, subscription_lines
from .db import (
    Client,
    Engagement,
    Expense,
    Invoice,
    InvoiceLine,
    Payment,
    Plan,
    Subscription,
    Tenant,
    TimeEntry,
    UsageEvent,
    utcnow,
)
from .security import Principal


class BillingMixin:
    db = None
    _audit = None

    def create_plan(
        self,
        principal: Principal,
        code: str,
        name: str,
        monthly_fee_minor: int,
        included_seats: int,
        extra_seat_minor: int,
        included_usage: Optional[dict] = None,
        usage_prices: Optional[dict] = None,
        currency: str = "ZAR",
    ) -> Plan:
        with self.db.session() as session:
            plan = Plan(
                code=code,
                name=name,
                monthly_fee_minor=monthly_fee_minor,
                included_seats=included_seats,
                extra_seat_minor=extra_seat_minor,
                included_usage_json=included_usage or {},
                usage_prices_json=usage_prices or {},
                currency=currency,
            )
            session.add(plan)
            session.flush()
            self._audit(session, principal, "plan.create", "plan", plan.id)
            return plan

    def subscribe(
        self, principal: Principal, plan_id: str, seats: int = 1
    ) -> Subscription:
        with self.db.session() as session:
            plan = session.get(Plan, plan_id)
            if plan is None or not plan.active:
                raise LookupError("plan not found")
            subscription = Subscription(
                tenant_id=principal.tenant_id,
                plan_id=plan.id,
                status="active",
                seats=max(1, seats),
                period_start=utcnow(),
                period_end=utcnow() + timedelta(days=30),
            )
            session.add(subscription)
            session.flush()
            self._audit(
                session,
                principal,
                "subscription.create",
                "subscription",
                subscription.id,
            )
            return subscription

    def record_usage(
        self,
        principal: Principal,
        metric: str,
        quantity: float,
        idempotency_key: str,
        metadata: Optional[dict] = None,
    ) -> UsageEvent:
        if quantity <= 0:
            raise ValueError("usage quantity must be positive")
        with self.db.session() as session:
            event = UsageEvent(
                tenant_id=principal.tenant_id,
                metric=metric,
                quantity=Decimal(str(quantity)),
                idempotency_key=idempotency_key,
                metadata_json=metadata or {},
            )
            session.add(event)
            try:
                session.flush()
            except IntegrityError as exc:
                raise ValueError("duplicate usage idempotency key") from exc
            self._audit(
                session,
                principal,
                "usage.record",
                "usage_event",
                event.id,
                {"metric": metric},
            )
            return event

    @staticmethod
    def _next_invoice_number(session, tenant_id: str) -> str:
        count = (
            session.scalar(
                select(func.count())
                .select_from(Invoice)
                .where(Invoice.tenant_id == tenant_id)
            )
            or 0
        )
        return "INV-%s-%06d" % (datetime.now(timezone.utc).strftime("%Y"), count + 1)

    def generate_consultancy_invoice(
        self,
        principal: Principal,
        client_id: str,
        period_start: datetime,
        period_end: datetime,
        due_days: int = 14,
    ) -> Invoice:
        with self.db.session() as session:
            tenant = session.get(Tenant, principal.tenant_id)
            client = session.scalar(
                select(Client).where(
                    Client.id == client_id, Client.tenant_id == principal.tenant_id
                )
            )
            if tenant is None or client is None:
                raise LookupError("tenant or client not found")
            engagements = session.scalars(
                select(Engagement).where(
                    Engagement.tenant_id == principal.tenant_id,
                    Engagement.client_id == client_id,
                )
            ).all()
            engagement_ids = [item.id for item in engagements]
            lines = []
            entries = []
            expenses = []
            if engagement_ids:
                entries = session.scalars(
                    select(TimeEntry).where(
                        TimeEntry.tenant_id == principal.tenant_id,
                        TimeEntry.engagement_id.in_(engagement_ids),
                        TimeEntry.billable.is_(True),
                        TimeEntry.approved.is_(True),
                        TimeEntry.invoiced_invoice_id.is_(None),
                        TimeEntry.occurred_at >= period_start,
                        TimeEntry.occurred_at < period_end,
                    )
                ).all()
                expenses = session.scalars(
                    select(Expense).where(
                        Expense.tenant_id == principal.tenant_id,
                        Expense.engagement_id.in_(engagement_ids),
                        Expense.reimbursable.is_(True),
                        Expense.approved.is_(True),
                        Expense.invoiced_invoice_id.is_(None),
                        Expense.incurred_at >= period_start,
                        Expense.incurred_at < period_end,
                    )
                ).all()
            for entry in entries:
                quantity = Decimal(entry.minutes) / Decimal(60)
                amount = int(
                    (quantity * Decimal(entry.rate_minor)).quantize(Decimal("1"))
                )
                lines.append(
                    BillingLine(
                        entry.description or "Consulting time",
                        quantity,
                        entry.rate_minor,
                        amount,
                        "time_entry",
                        entry.id,
                    )
                )
            for expense in expenses:
                lines.append(
                    BillingLine(
                        "Reimbursable expense: %s" % expense.category,
                        Decimal(1),
                        expense.amount_minor,
                        expense.amount_minor,
                        "expense",
                        expense.id,
                    )
                )
            for engagement in engagements:
                if (
                    engagement.billing_model == "retainer"
                    and engagement.retainer_minor > 0
                ):
                    lines.append(
                        BillingLine(
                            "Retainer: %s" % engagement.name,
                            Decimal(1),
                            engagement.retainer_minor,
                            engagement.retainer_minor,
                            "retainer",
                            engagement.id,
                        )
                    )
            totals = calculate_totals(lines, tenant.tax_rate_bps)
            invoice = Invoice(
                tenant_id=tenant.id,
                client_id=client.id,
                number=self._next_invoice_number(session, tenant.id),
                status="open",
                currency=tenant.currency,
                subtotal_minor=totals.subtotal_minor,
                tax_minor=totals.tax_minor,
                total_minor=totals.total_minor,
                balance_minor=totals.total_minor,
                issued_at=utcnow(),
                due_at=utcnow() + timedelta(days=due_days),
            )
            session.add(invoice)
            session.flush()
            for line in totals.lines:
                session.add(
                    InvoiceLine(
                        tenant_id=tenant.id,
                        invoice_id=invoice.id,
                        description=line.description,
                        quantity=line.quantity,
                        unit_amount_minor=line.unit_amount_minor,
                        amount_minor=line.amount_minor,
                        source_type=line.source_type,
                        source_id=line.source_id,
                    )
                )
                if line.source_type == "time_entry":
                    session.get(TimeEntry, line.source_id).invoiced_invoice_id = (
                        invoice.id
                    )
                elif line.source_type == "expense":
                    session.get(Expense, line.source_id).invoiced_invoice_id = (
                        invoice.id
                    )
            self._audit(session, principal, "invoice.generate", "invoice", invoice.id)
            return invoice

    def generate_platform_invoice(
        self, principal: Principal, subscription_id: str
    ) -> Invoice:
        with self.db.session() as session:
            tenant = session.get(Tenant, principal.tenant_id)
            subscription = session.scalar(
                select(Subscription).where(
                    Subscription.id == subscription_id,
                    Subscription.tenant_id == principal.tenant_id,
                )
            )
            if tenant is None or subscription is None:
                raise LookupError("subscription not found")
            plan = session.get(Plan, subscription.plan_id)
            if plan is None:
                raise LookupError("plan not found")
            period_end = subscription.period_end or utcnow()
            usage_rows = session.execute(
                select(UsageEvent.metric, func.sum(UsageEvent.quantity))
                .where(
                    UsageEvent.tenant_id == principal.tenant_id,
                    UsageEvent.invoiced_invoice_id.is_(None),
                    UsageEvent.occurred_at >= subscription.period_start,
                    UsageEvent.occurred_at < period_end,
                )
                .group_by(UsageEvent.metric)
            ).all()
            usage = {metric: Decimal(str(quantity)) for metric, quantity in usage_rows}
            lines = subscription_lines(
                plan.monthly_fee_minor,
                subscription.seats,
                plan.included_seats,
                plan.extra_seat_minor,
                usage,
                {k: Decimal(str(v)) for k, v in plan.included_usage_json.items()},
                {k: int(v) for k, v in plan.usage_prices_json.items()},
            )
            totals = calculate_totals(lines, tenant.tax_rate_bps)
            invoice = Invoice(
                tenant_id=tenant.id,
                subscription_id=subscription.id,
                number=self._next_invoice_number(session, tenant.id),
                status="open",
                currency=plan.currency,
                subtotal_minor=totals.subtotal_minor,
                tax_minor=totals.tax_minor,
                total_minor=totals.total_minor,
                balance_minor=totals.total_minor,
                issued_at=utcnow(),
                due_at=utcnow() + timedelta(days=7),
            )
            session.add(invoice)
            session.flush()
            for line in totals.lines:
                session.add(
                    InvoiceLine(
                        tenant_id=tenant.id,
                        invoice_id=invoice.id,
                        description=line.description,
                        quantity=line.quantity,
                        unit_amount_minor=line.unit_amount_minor,
                        amount_minor=line.amount_minor,
                        source_type=line.source_type,
                    )
                )
            events = session.scalars(
                select(UsageEvent).where(
                    UsageEvent.tenant_id == principal.tenant_id,
                    UsageEvent.invoiced_invoice_id.is_(None),
                    UsageEvent.occurred_at >= subscription.period_start,
                    UsageEvent.occurred_at < period_end,
                )
            ).all()
            for event in events:
                event.invoiced_invoice_id = invoice.id
            self._audit(
                session, principal, "platform_invoice.generate", "invoice", invoice.id
            )
            return invoice

    def record_payment(
        self,
        principal: Principal,
        invoice_id: str,
        amount_minor: int,
        provider: str = "manual",
        external_id: str = "",
    ) -> Payment:
        if amount_minor <= 0:
            raise ValueError("payment must be positive")
        with self.db.session() as session:
            invoice = session.scalar(
                select(Invoice).where(
                    Invoice.id == invoice_id,
                    Invoice.tenant_id == principal.tenant_id,
                )
            )
            if invoice is None:
                raise LookupError("invoice not found")
            applied = min(amount_minor, invoice.balance_minor)
            payment = Payment(
                tenant_id=principal.tenant_id,
                invoice_id=invoice.id,
                provider=provider,
                external_id=external_id or None,
                amount_minor=applied,
                currency=invoice.currency,
            )
            invoice.balance_minor -= applied
            invoice.status = "paid" if invoice.balance_minor == 0 else "partially_paid"
            if invoice.balance_minor == 0:
                invoice.paid_at = utcnow()
            session.add(payment)
            session.flush()
            self._audit(session, principal, "payment.record", "payment", payment.id)
            return payment
