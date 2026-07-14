"""OpenTelemetry-compatible zero-dependency telemetry envelope."""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterator, List, Optional


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    started_at_ns: int
    ended_at_ns: Optional[int] = None
    status: str = "UNSET"
    attributes: Dict[str, object] = field(default_factory=dict)
    events: List[Dict[str, object]] = field(default_factory=list)

    def add_event(self, name: str, **attributes: object) -> None:
        self.events.append(
            {"name": name, "time_unix_nano": time.time_ns(), "attributes": attributes}
        )

    def otlp_dict(self) -> Dict[str, object]:
        return asdict(self)


class Telemetry:
    def __init__(self) -> None:
        self.spans: List[Span] = []

    @contextmanager
    def span(
        self,
        name: str,
        *,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        **attributes: object,
    ) -> Iterator[Span]:
        span = Span(
            name=name,
            trace_id=trace_id or uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=parent_span_id,
            started_at_ns=time.time_ns(),
            attributes=dict(attributes),
        )
        try:
            yield span
            span.status = "OK"
        except Exception as exc:
            span.status = "ERROR"
            span.add_event("exception", type=type(exc).__name__, message=str(exc))
            raise
        finally:
            span.ended_at_ns = time.time_ns()
            self.spans.append(span)
