"""FastAPI surface for the Dr Moagi Consultancy Cloud Network."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from .billing import StripeRESTGateway
from .security import Principal, TokenService, require_role
from .service import ConsultancyService

app = FastAPI(
    title="Dr Moagi Software Consultancy Cloud Network",
    version="1.0.0",
    description="Multi-tenant consultancy operations, billing, and corporate administration SaaS.",
)
service = ConsultancyService()


def _token_service() -> TokenService:
    return TokenService()


def _serialize(row):
    return {column.name: getattr(row, column.name) for column in row.__table__.columns}


def current_principal(authorization: Optional[str] = Header(default=None)) -> Principal:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="bearer token required")
    try:
        return _token_service().verify(authorization[7:])
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def authorize(principal: Principal, roles) -> None:
    try:
        require_role(principal, roles)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


class BootstrapRequest(BaseModel):
    company_name: str = Field(min_length=2, max_length=200)
    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,98}[a-z0-9]$")
    legal_name: str = Field(min_length=2, max_length=250)
    admin_name: str = Field(min_length=2, max_length=200)
    admin_email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=10, max_length=256)
    currency: str = Field(default="ZAR", min_length=3, max_length=3)
    tax_rate_bps: int = Field(default=0, ge=0, le=10000)


class LoginRequest(BaseModel):
    email: str
    password: str
    tenant_id: Optional[str] = None


class UserRequest(BaseModel):
    email: str
    full_name: str
    password: str = Field(min_length=10)
    role: str


class ClientRequest(BaseModel):
    name: str
    billing_email: str = ""
    tax_number: str = ""


class LeadRequest(BaseModel):
    company: str
    contact_name: str
    email: str = ""
    estimated_value_minor: int = Field(default=0, ge=0)


class EngagementRequest(BaseModel):
    client_id: str
    name: str
    budget_minor: int = Field(default=0, ge=0)
    hourly_rate_minor: int = Field(default=0, ge=0)
    billing_model: str = "time_and_materials"
    retainer_minor: int = Field(default=0, ge=0)


class TimeRequest(BaseModel):
    engagement_id: str
    minutes: int = Field(gt=0, le=1440)
    description: str = ""
    billable: bool = True
    approved: bool = False
    rate_minor: int = Field(default=0, ge=0)


class ExpenseRequest(BaseModel):
    engagement_id: str
    amount_minor: int = Field(gt=0)
    category: str
    description: str = ""
    reimbursable: bool = True
    approved: bool = False


class PlanRequest(BaseModel):
    code: str
    name: str
    monthly_fee_minor: int = Field(ge=0)
    included_seats: int = Field(default=1, ge=0)
    extra_seat_minor: int = Field(default=0, ge=0)
    included_usage: dict = Field(default_factory=dict)
    usage_prices: dict = Field(default_factory=dict)
    currency: str = "ZAR"


class SubscriptionRequest(BaseModel):
    plan_id: str
    seats: int = Field(default=1, ge=1)


class UsageRequest(BaseModel):
    metric: str
    quantity: float = Field(gt=0)
    idempotency_key: str
    metadata: dict = Field(default_factory=dict)


class ConsultancyInvoiceRequest(BaseModel):
    client_id: str
    period_start: datetime
    period_end: datetime
    due_days: int = Field(default=14, ge=0, le=365)


class PlatformInvoiceRequest(BaseModel):
    subscription_id: str


class PaymentRequest(BaseModel):
    invoice_id: str
    amount_minor: int = Field(gt=0)
    provider: str = "manual"
    external_id: str = ""


class EmployeeRequest(BaseModel):
    employee_number: str
    title: str
    department: str = "Consulting"
    salary_minor: int = Field(default=0, ge=0)
    user_id: Optional[str] = None


class LeaveRequestModel(BaseModel):
    employee_id: str
    start_at: datetime
    end_at: datetime
    leave_type: str = "annual"
    reason: str = ""


class VendorRequest(BaseModel):
    name: str
    email: str = ""


class PurchaseOrderRequest(BaseModel):
    vendor_id: str
    number: str
    total_minor: int = Field(ge=0)
    description: str = ""


class StripeCheckoutRequest(BaseModel):
    price_id: str
    success_url: str
    cancel_url: str


@app.get("/health")
def health():
    return {"status": "ok", "service": "dr-moagi-consultancy-saas", "version": "1.0.0"}


@app.post("/v1/bootstrap", status_code=status.HTTP_201_CREATED)
def bootstrap(
    request: BootstrapRequest, x_bootstrap_token: Optional[str] = Header(default=None)
):
    expected = os.getenv("DM_BOOTSTRAP_TOKEN", "")
    if not expected or x_bootstrap_token != expected:
        raise HTTPException(status_code=401, detail="invalid bootstrap token")
    try:
        return service.bootstrap(
            request.company_name,
            request.slug,
            request.legal_name,
            request.admin_name,
            request.admin_email,
            request.password,
            request.currency.upper(),
            request.tax_rate_bps,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/auth/token")
def login(request: LoginRequest):
    try:
        principal = service.authenticate(
            request.email, request.password, request.tenant_id
        )
        return {
            "access_token": _token_service().issue(principal),
            "token_type": "bearer",
        }
    except (PermissionError, ValueError) as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@app.post("/v1/users", status_code=201)
def create_user(
    request: UserRequest, principal: Principal = Depends(current_principal)
):
    authorize(principal, {"tenant_owner", "platform_admin"})
    try:
        return _serialize(
            service.create_user(
                principal,
                request.email,
                request.full_name,
                request.password,
                request.role,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/clients", status_code=201)
def create_client(
    request: ClientRequest, principal: Principal = Depends(current_principal)
):
    authorize(
        principal, {"tenant_owner", "operations_manager", "finance_admin", "consultant"}
    )
    return _serialize(
        service.create_client(
            principal, request.name, request.billing_email, request.tax_number
        )
    )


@app.post("/v1/leads", status_code=201)
def create_lead(
    request: LeadRequest, principal: Principal = Depends(current_principal)
):
    authorize(principal, {"tenant_owner", "operations_manager", "consultant"})
    return _serialize(
        service.create_lead(
            principal,
            request.company,
            request.contact_name,
            request.email,
            request.estimated_value_minor,
        )
    )


@app.post("/v1/engagements", status_code=201)
def create_engagement(
    request: EngagementRequest, principal: Principal = Depends(current_principal)
):
    authorize(principal, {"tenant_owner", "operations_manager"})
    try:
        return _serialize(
            service.create_engagement(
                principal,
                request.client_id,
                request.name,
                request.budget_minor,
                request.hourly_rate_minor,
                request.billing_model,
                request.retainer_minor,
            )
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/time-entries", status_code=201)
def record_time(
    request: TimeRequest, principal: Principal = Depends(current_principal)
):
    authorize(principal, {"tenant_owner", "operations_manager", "consultant"})
    try:
        return _serialize(
            service.record_time(
                principal,
                request.engagement_id,
                request.minutes,
                request.description,
                request.billable,
                request.approved,
                request.rate_minor,
            )
        )
    except (ValueError, LookupError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/expenses", status_code=201)
def record_expense(
    request: ExpenseRequest, principal: Principal = Depends(current_principal)
):
    authorize(
        principal, {"tenant_owner", "operations_manager", "finance_admin", "consultant"}
    )
    try:
        return _serialize(
            service.record_expense(
                principal,
                request.engagement_id,
                request.amount_minor,
                request.category,
                request.description,
                request.reimbursable,
                request.approved,
            )
        )
    except (ValueError, LookupError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/billing/plans", status_code=201)
def create_plan(
    request: PlanRequest, principal: Principal = Depends(current_principal)
):
    authorize(principal, {"platform_admin"})
    return _serialize(
        service.create_plan(
            principal,
            request.code,
            request.name,
            request.monthly_fee_minor,
            request.included_seats,
            request.extra_seat_minor,
            request.included_usage,
            request.usage_prices,
            request.currency,
        )
    )


@app.post("/v1/billing/subscriptions", status_code=201)
def subscribe(
    request: SubscriptionRequest, principal: Principal = Depends(current_principal)
):
    authorize(principal, {"tenant_owner", "platform_admin"})
    try:
        return _serialize(service.subscribe(principal, request.plan_id, request.seats))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/billing/usage", status_code=202)
def record_usage(
    request: UsageRequest, principal: Principal = Depends(current_principal)
):
    authorize(principal, {"tenant_owner", "operations_manager", "platform_admin"})
    try:
        return _serialize(
            service.record_usage(
                principal,
                request.metric,
                request.quantity,
                request.idempotency_key,
                request.metadata,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/v1/invoices/consultancy", status_code=201)
def generate_consultancy_invoice(
    request: ConsultancyInvoiceRequest,
    principal: Principal = Depends(current_principal),
):
    authorize(principal, {"tenant_owner", "finance_admin"})
    try:
        return _serialize(
            service.generate_consultancy_invoice(
                principal,
                request.client_id,
                request.period_start,
                request.period_end,
                request.due_days,
            )
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/invoices/platform", status_code=201)
def generate_platform_invoice(
    request: PlatformInvoiceRequest, principal: Principal = Depends(current_principal)
):
    authorize(principal, {"tenant_owner", "finance_admin", "platform_admin"})
    try:
        return _serialize(
            service.generate_platform_invoice(principal, request.subscription_id)
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/payments", status_code=201)
def record_payment(
    request: PaymentRequest, principal: Principal = Depends(current_principal)
):
    authorize(principal, {"tenant_owner", "finance_admin", "platform_admin"})
    try:
        return _serialize(
            service.record_payment(
                principal,
                request.invoice_id,
                request.amount_minor,
                request.provider,
                request.external_id,
            )
        )
    except (ValueError, LookupError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/corporate/employees", status_code=201)
def create_employee(
    request: EmployeeRequest, principal: Principal = Depends(current_principal)
):
    authorize(principal, {"tenant_owner", "operations_manager"})
    return _serialize(
        service.create_employee(
            principal,
            request.employee_number,
            request.title,
            request.department,
            request.salary_minor,
            request.user_id,
        )
    )


@app.post("/v1/corporate/leave", status_code=201)
def request_leave(
    request: LeaveRequestModel, principal: Principal = Depends(current_principal)
):
    authorize(principal, {"tenant_owner", "operations_manager", "consultant"})
    try:
        return _serialize(
            service.request_leave(
                principal,
                request.employee_id,
                request.start_at,
                request.end_at,
                request.leave_type,
                request.reason,
            )
        )
    except (ValueError, LookupError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/vendors", status_code=201)
def create_vendor(
    request: VendorRequest, principal: Principal = Depends(current_principal)
):
    authorize(principal, {"tenant_owner", "finance_admin", "operations_manager"})
    return _serialize(service.create_vendor(principal, request.name, request.email))


@app.post("/v1/purchase-orders", status_code=201)
def create_purchase_order(
    request: PurchaseOrderRequest, principal: Principal = Depends(current_principal)
):
    authorize(principal, {"tenant_owner", "finance_admin", "operations_manager"})
    try:
        return _serialize(
            service.create_purchase_order(
                principal,
                request.vendor_id,
                request.number,
                request.total_minor,
                request.description,
            )
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/dashboard")
def dashboard(principal: Principal = Depends(current_principal)):
    return service.dashboard(principal)


@app.post("/v1/billing/stripe/checkout")
def stripe_checkout(
    request: StripeCheckoutRequest, principal: Principal = Depends(current_principal)
):
    authorize(principal, {"tenant_owner", "platform_admin"})
    try:
        return StripeRESTGateway().create_checkout(
            request.price_id,
            request.success_url,
            request.cancel_url,
            principal.tenant_id,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/v1/billing/stripe/webhook")
async def stripe_webhook(
    request: Request, stripe_signature: str = Header(alias="Stripe-Signature")
):
    payload = await request.body()
    try:
        event = StripeRESTGateway().verify_webhook(payload, stripe_signature)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "received": True,
        "event_id": event.get("id"),
        "event_type": event.get("type"),
    }
