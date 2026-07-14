"""Unified consultancy application service."""

from typing import Optional

from .db import Database
from .service_base import FoundationMixin
from .service_billing import BillingMixin
from .service_corporate import CorporateMixin


class ConsultancyService(FoundationMixin, BillingMixin, CorporateMixin):
    def __init__(self, database: Optional[Database] = None) -> None:
        self.db = database or Database()
        self.db.create_schema()
