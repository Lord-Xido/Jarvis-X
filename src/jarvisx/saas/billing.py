"""Provider-neutral billing, SaaS metering, and Stripe REST integration."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, Iterable, Mapping, Optional


def money(value: Decimal) -> int:
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class BillingLine:
    description: str
    quantity: Decimal
    unit_amount_minor: int
    amount_minor: int
    source_type: str = "manual"
    source_id: Optional[str] = None


@dataclass(frozen=True)
class BillingTotals:
    lines: tuple
    subtotal_minor: int
    tax_minor: int
    total_minor: int


def calculate_totals(lines: Iterable[BillingLine], tax_rate_bps: int) -> BillingTotals:
    frozen = tuple(lines)
    subtotal = sum(line.amount_minor for line in frozen)
    tax = money(Decimal(subtotal) * Decimal(tax_rate_bps) / Decimal(10000))
    return BillingTotals(frozen, subtotal, tax, subtotal + tax)


def subscription_lines(
    monthly_fee_minor: int,
    seats: int,
    included_seats: int,
    extra_seat_minor: int,
    usage: Mapping[str, Decimal],
    included_usage: Mapping[str, Decimal],
    usage_prices: Mapping[str, int],
) -> tuple:
    lines = [
        BillingLine(
            "Platform subscription",
            Decimal(1),
            monthly_fee_minor,
            monthly_fee_minor,
            "plan",
        )
    ]
    extra_seats = max(0, seats - included_seats)
    if extra_seats:
        amount = extra_seats * extra_seat_minor
        lines.append(
            BillingLine(
                "Additional seats",
                Decimal(extra_seats),
                extra_seat_minor,
                amount,
                "seat",
            )
        )
    for metric, quantity in sorted(usage.items()):
        included = Decimal(str(included_usage.get(metric, 0)))
        billable = max(Decimal(0), Decimal(quantity) - included)
        unit_price = int(usage_prices.get(metric, 0))
        if billable > 0 and unit_price > 0:
            lines.append(
                BillingLine(
                    "Usage: %s" % metric,
                    billable,
                    unit_price,
                    money(billable * Decimal(unit_price)),
                    "usage",
                )
            )
    return tuple(lines)


class StripeRESTGateway:
    """Minimal Stripe adapter without imposing an SDK dependency."""

    api_base = "https://api.stripe.com/v1"

    def __init__(self, api_key: str = "", webhook_secret: str = "") -> None:
        self.api_key = api_key or os.getenv("STRIPE_SECRET_KEY", "")
        self.webhook_secret = webhook_secret or os.getenv("STRIPE_WEBHOOK_SECRET", "")

    def _post(self, path: str, values: Dict[str, str]) -> dict:
        if not self.api_key:
            raise RuntimeError("STRIPE_SECRET_KEY is not configured")
        request = urllib.request.Request(
            self.api_base + path,
            data=urllib.parse.urlencode(values).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": "Bearer %s" % self.api_key,
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "dr-moagi-consultancy-saas/1.0",
            },
        )
        with urllib.request.urlopen(request, timeout=20) as response:  # nosec B310
            return json.loads(response.read().decode("utf-8"))

    def create_checkout(
        self, price_id: str, success_url: str, cancel_url: str, tenant_id: str
    ) -> dict:
        return self._post(
            "/checkout/sessions",
            {
                "mode": "subscription",
                "line_items[0][price]": price_id,
                "line_items[0][quantity]": "1",
                "success_url": success_url,
                "cancel_url": cancel_url,
                "client_reference_id": tenant_id,
                "metadata[tenant_id]": tenant_id,
            },
        )

    def verify_webhook(
        self, payload: bytes, signature_header: str, tolerance: int = 300
    ) -> dict:
        if not self.webhook_secret:
            raise RuntimeError("STRIPE_WEBHOOK_SECRET is not configured")
        fields = {}
        for item in signature_header.split(","):
            key, _, value = item.partition("=")
            fields.setdefault(key, []).append(value)
        timestamp = int(fields.get("t", ["0"])[0])
        if abs(int(time.time()) - timestamp) > tolerance:
            raise ValueError("webhook timestamp outside tolerance")
        signed = (str(timestamp) + ".").encode("utf-8") + payload
        expected = hmac.new(
            self.webhook_secret.encode(), signed, hashlib.sha256
        ).hexdigest()
        if not any(
            hmac.compare_digest(expected, candidate)
            for candidate in fields.get("v1", [])
        ):
            raise ValueError("invalid webhook signature")
        return json.loads(payload.decode("utf-8"))
