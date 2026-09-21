"""HR, procurement, and enterprise dashboard operations."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import func, select

from .db import (
    Employee,
    Engagement,
    Invoice,
    Lead,
    LeaveRequest,
    PurchaseOrder,
    TimeEntry,
    Vendor,
)
from .geometry import engagement_point, enterprise_centroid
from .security import Principal


class CorporateMixin:
    db = None
    _audit = None

    def create_employee(
        self,
        principal: Principal,
        employee_number: str,
        title: str,
        department: str = "Consulting",
        salary_minor: int = 0,
        user_id: Optional[str] = None,
    ) -> Employee:
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

    def request_leave(
        self,
        principal: Principal,
        employee_id: str,
        start_at: datetime,
        end_at: datetime,
        leave_type: str = "annual",
        reason: str = "",
    ) -> LeaveRequest:
        if end_at <= start_at:
            raise ValueError("leave end must be after start")
        with self.db.session() as session:
            employee = session.scalar(
                select(Employee).where(
                    Employee.id == employee_id,
                    Employee.tenant_id == principal.tenant_id,
                )
            )
            if employee is None:
                raise LookupError("employee not found")
            request = LeaveRequest(
                tenant_id=principal.tenant_id,
                employee_id=employee.id,
                leave_type=leave_type,
                start_at=start_at,
                end_at=end_at,
                reason=reason,
            )
            session.add(request)
            session.flush()
            self._audit(
                session, principal, "leave.request", "leave_request", request.id
            )
            return request

    def create_vendor(self, principal: Principal, name: str, email: str = "") -> Vendor:
        with self.db.session() as session:
            vendor = Vendor(
                tenant_id=principal.tenant_id, name=name, email=email or None
            )
            session.add(vendor)
            session.flush()
            self._audit(session, principal, "vendor.create", "vendor", vendor.id)
            return vendor

    def create_purchase_order(
        self,
        principal: Principal,
        vendor_id: str,
        number: str,
        total_minor: int,
        description: str = "",
    ) -> PurchaseOrder:
        with self.db.session() as session:
            vendor = session.scalar(
                select(Vendor).where(
                    Vendor.id == vendor_id, Vendor.tenant_id == principal.tenant_id
                )
            )
            if vendor is None:
                raise LookupError("vendor not found")
            order = PurchaseOrder(
                tenant_id=principal.tenant_id,
                vendor_id=vendor.id,
                number=number,
                total_minor=total_minor,
                description=description,
            )
            session.add(order)
            session.flush()
            self._audit(
                session, principal, "purchase_order.create", "purchase_order", order.id
            )
            return order

    def dashboard(self, principal: Principal) -> dict:
        with self.db.session() as session:
            engagements = session.scalars(
                select(Engagement).where(Engagement.tenant_id == principal.tenant_id)
            ).all()
            points = []
            rows = []
            for engagement in engagements:
                approved_value = (
                    session.scalar(
                        select(
                            func.coalesce(
                                func.sum(TimeEntry.minutes * TimeEntry.rate_minor / 60),
                                0,
                            )
                        ).where(
                            TimeEntry.engagement_id == engagement.id,
                            TimeEntry.approved.is_(True),
                        )
                    )
                    or 0
                )
                budget_util = (
                    float(approved_value) / engagement.budget_minor
                    if engagement.budget_minor
                    else 0.0
                )
                point = engagement_point(
                    {
                        "progress": engagement.progress_bps / 10000.0,
                        "quality": engagement.quality_bps / 10000.0,
                        "budget_utilization": budget_util,
                        "risk": engagement.risk_bps / 10000.0,
                        "governance": engagement.governance_bps / 10000.0,
                    }
                )
                points.append((point, max(1.0, float(engagement.budget_minor))))
                rows.append(
                    {
                        "id": engagement.id,
                        "name": engagement.name,
                        "point": point.as_tuple(),
                        "health": point.health,
                        "risk": point.risk,
                    }
                )
            receivables = (
                session.scalar(
                    select(func.coalesce(func.sum(Invoice.balance_minor), 0)).where(
                        Invoice.tenant_id == principal.tenant_id,
                        Invoice.status.in_(["open", "partially_paid", "past_due"]),
                    )
                )
                or 0
            )
            pipeline = (
                session.scalar(
                    select(
                        func.coalesce(
                            func.sum(
                                Lead.estimated_value_minor
                                * Lead.probability_bps
                                / 10000
                            ),
                            0,
                        )
                    ).where(
                        Lead.tenant_id == principal.tenant_id,
                        Lead.stage.notin_(["won", "lost"]),
                    )
                )
                or 0
            )
            return {
                "tenant_id": principal.tenant_id,
                "geometry": enterprise_centroid(points),
                "engagements": rows,
                "receivables_minor": int(receivables),
                "weighted_pipeline_minor": int(pipeline),
                "active_engagements": sum(
                    item.status == "active" for item in engagements
                ),
            }
