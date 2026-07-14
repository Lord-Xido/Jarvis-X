"""Unified consultancy application service."""

from functools import wraps
from typing import Optional

from .db import Database
from .rls import tenant_scope
from .service_base import FoundationMixin
from .service_billing import BillingMixin
from .service_corporate import CorporateMixin


class ConsultancyService(FoundationMixin, BillingMixin, CorporateMixin):
    def __init__(self, database: Optional[Database] = None) -> None:
        self.db = database or Database()
        self.db.create_schema()


def _tenant_method(method):
    @wraps(method)
    def wrapped(self, principal, *args, **kwargs):
        tenant_id = getattr(principal, "tenant_id", None)
        if not tenant_id:
            raise PermissionError("tenant principal is required")
        with tenant_scope(tenant_id):
            return method(self, principal, *args, **kwargs)

    return wrapped


def _authentication_method(method):
    @wraps(method)
    def wrapped(self, email, password, tenant_id=None):
        with tenant_scope(tenant_id):
            return method(self, email, password, tenant_id)

    return wrapped


for _method_name in (
    "create_user",
    "create_client",
    "create_lead",
    "create_engagement",
    "record_time",
    "record_expense",
    "create_employee_shell",
    "create_plan",
    "subscribe",
    "record_usage",
    "generate_consultancy_invoice",
    "generate_platform_invoice",
    "record_payment",
    "create_employee",
    "request_leave",
    "create_vendor",
    "create_purchase_order",
    "dashboard",
):
    setattr(
        ConsultancyService,
        _method_name,
        _tenant_method(getattr(ConsultancyService, _method_name)),
    )

ConsultancyService.authenticate = _authentication_method(ConsultancyService.authenticate)
