"""Tenant, identity, CRM, engagement, time, and expense operations."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .db import (
    AuditEvent,
    Client,
    Employee,
    Engagement,
    Expense,
    Lead,
    Membership,
    Tenant,
    TimeEntry,
    User,
    set_tenant_context,
    utcnow,
)
from .security import Principal, hash_password, verify_password


class FoundationMixin:
    db = None

    @staticmethod
    def _audit(
        session: Session,
        principal: Optional[Principal],
        action: str,
        entity_type: str,
        entity_id: Optional[str],
        payload: Optional[dict] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        session.add(
            AuditEvent(
                tenant_id=tenant_id or (principal.tenant_id if principal else None),
                actor_user_id=principal.user_id if principal else None,
                action=action,
                entity_type=entity_type,
                entity_id=entity_id,
                payload_json=payload or {},
            )
        )

    def bootstrap(
        self,
        company_name: str,
        slug: str,
        legal_name: str,
        admin_name: str,
        admin_email: str,
        password: str,
        currency: str = "ZAR",
        tax_rate_bps: int = 0,
    ) -> dict:
        with self.db.session() as session:
            if session.scalar(select(func.count()).select_from(Tenant)):
                raise RuntimeError("platform is already bootstrapped")
            tenant = Tenant(
                name=company_name,
                slug=slug,
                legal_name=legal_name,
                currency=currency,
                tax_rate_bps=tax_rate_bps,
            )
            user = User(
                email=admin_email.lower(),
                full_name=admin_name,
                password_hash=hash_password(password),
            )
            session.add_all([tenant, user])
            session.flush()
            session.add(
                Membership(tenant_id=tenant.id, user_id=user.id, role="platform_admin")
            )
            self._audit(
                session,
                None,
                "platform.bootstrap",
                "tenant",
                tenant.id,
                {"slug": slug},
                tenant.id,
            )
            return {"tenant_id": tenant.id, "user_id": user.id}

    def authenticate(
        self, email: str, password: str, tenant_id: Optional[str] = None
    ) -> Principal:
        with self.db.session() as session:
            user = session.scalar(
                select(User).where(User.email == email.lower(), User.active.is_(True))
            )
            if user is None or not verify_password(password, user.password_hash):
                raise PermissionError("invalid credentials")
            query = select(Membership).where(Membership.user_id == user.id)
            if tenant_id:
                query = query.where(Membership.tenant_id == tenant_id)
            membership = session.scalar(query.order_by(Membership.created_at.asc()))
            if membership is None:
                raise PermissionError("user has no tenant membership")
            return Principal(user.id, membership.tenant_id, membership.role)

    def create_user(
        self, principal: Principal, email: str, full_name: str, password: str, role: str
    ) -> User:
        with self.db.session() as session:
            set_tenant_context(session, principal.tenant_id)
            user = User(
                email=email.lower(),
                full_name=full_name,
                password_hash=hash_password(password),
            )
            session.add(user)
            session.flush()
            session.add(
                Membership(tenant_id=principal.tenant_id, user_id=user.id, role=role)
            )
            self._audit(
                session, principal, "user.create", "user", user.id, {"role": role}
            )
            return user

    def create_client(
        self,
        principal: Principal,
        name: str,
        billing_email: str = "",
        tax_number: str = "",
    ) -> Client:
        with self.db.session() as session:
            set_tenant_context(session, principal.tenant_id)
            client = Client(
                tenant_id=principal.tenant_id,
                name=name,
                billing_email=billing_email or None,
                tax_number=tax_number or None,
            )
            session.add(client)
            session.flush()
            self._audit(session, principal, "client.create", "client", client.id)
            return client

    def create_lead(
        self,
        principal: Principal,
        company: str,
        contact_name: str,
        email: str = "",
        value_minor: int = 0,
    ) -> Lead:
        with self.db.session() as session:
            lead = Lead(
                tenant_id=principal.tenant_id,
                company=company,
                contact_name=contact_name,
                email=email or None,
                estimated_value_minor=value_minor,
            )
            session.add(lead)
            session.flush()
            self._audit(session, principal, "lead.create", "lead", lead.id)
            return lead

    def create_engagement(
        self,
        principal: Principal,
        client_id: str,
        name: str,
        budget_minor: int,
        hourly_rate_minor: int,
        billing_model: str = "time_and_materials",
        retainer_minor: int = 0,
    ) -> Engagement:
        with self.db.session() as session:
            set_tenant_context(session, principal.tenant_id)
            client = session.scalar(
                select(Client).where(
                    Client.id == client_id, Client.tenant_id == principal.tenant_id
                )
            )
            if client is None:
                raise LookupError("client not found")
            engagement = Engagement(
                tenant_id=principal.tenant_id,
                client_id=client_id,
                name=name,
                budget_minor=budget_minor,
                hourly_rate_minor=hourly_rate_minor,
                billing_model=billing_model,
                retainer_minor=retainer_minor,
                status="active",
                start_at=utcnow(),
            )
            session.add(engagement)
            session.flush()
            self._audit(
                session, principal, "engagement.create", "engagement", engagement.id
            )
            return engagement

    def record_time(
        self,
        principal: Principal,
        engagement_id: str,
        minutes: int,
        description: str,
        billable: bool = True,
        approved: bool = False,
        rate_minor: int = 0,
    ) -> TimeEntry:
        if minutes <= 0 or minutes > 24 * 60:
            raise ValueError("minutes must be inside (0, 1440]")
        with self.db.session() as session:
            engagement = session.scalar(
                select(Engagement).where(
                    Engagement.id == engagement_id,
                    Engagement.tenant_id == principal.tenant_id,
                )
            )
            if engagement is None:
                raise LookupError("engagement not found")
            entry = TimeEntry(
                tenant_id=principal.tenant_id,
                engagement_id=engagement_id,
                user_id=principal.user_id,
                minutes=minutes,
                rate_minor=rate_minor or engagement.hourly_rate_minor,
                description=description,
                billable=billable,
                approved=approved,
            )
            session.add(entry)
            session.flush()
            self._audit(
                session,
                principal,
                "time.record",
                "time_entry",
                entry.id,
                {"minutes": minutes},
            )
            return entry

    def record_expense(
        self,
        principal: Principal,
        engagement_id: str,
        amount_minor: int,
        category: str,
        description: str = "",
        reimbursable: bool = True,
        approved: bool = False,
    ) -> Expense:
        if amount_minor <= 0:
            raise ValueError("expense must be positive")
        with self.db.session() as session:
            engagement = session.scalar(
                select(Engagement).where(
                    Engagement.id == engagement_id,
                    Engagement.tenant_id == principal.tenant_id,
                )
            )
            if engagement is None:
                raise LookupError("engagement not found")
            expense = Expense(
                tenant_id=principal.tenant_id,
                engagement_id=engagement_id,
                amount_minor=amount_minor,
                category=category,
                description=description,
                reimbursable=reimbursable,
                approved=approved,
            )
            session.add(expense)
            session.flush()
            self._audit(session, principal, "expense.record", "expense", expense.id)
            return expense

    def create_employee_shell(
        self,
        principal: Principal,
        employee_number: str,
        title: str,
        department: str = "Consulting",
        salary_minor: int = 0,
        user_id: Optional[str] = None,
    ) -> Employee:
        """Compatibility helper used by corporate administration workflows."""

        with self.db.session() as session:
            employee = Employee(
                tenant_id=principal.tenant_id,
                user_id=user_id,
                employee_number=employee_number,
                title=title,
                department=department,
                salary_minor=salary_minor,
            )
            session.add(employee)
            session.flush()
            self._audit(session, principal, "employee.create", "employee", employee.id)
            return employee
