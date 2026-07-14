"""Request-scoped PostgreSQL tenant context for row-level security."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

from sqlalchemy import event, text
from sqlalchemy.orm import Session

_tenant_id: ContextVar[Optional[str]] = ContextVar("dm_tenant_id", default=None)


@contextmanager
def tenant_scope(tenant_id: Optional[str]) -> Iterator[None]:
    token = _tenant_id.set(tenant_id)
    try:
        yield
    finally:
        _tenant_id.reset(token)


@event.listens_for(Session, "after_begin")
def apply_postgres_tenant_context(session, transaction, connection) -> None:
    del session, transaction
    tenant_id = _tenant_id.get()
    if tenant_id and connection.dialect.name == "postgresql":
        connection.execute(
            text("SELECT set_config('app.tenant_id', :tenant_id, true)"),
            {"tenant_id": tenant_id},
        )
