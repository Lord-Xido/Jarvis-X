from __future__ import annotations

import uvicorn


def start_web() -> None:
    # The FastAPI application serves both the operational dashboard at `/`
    # and the JSON runtime endpoints. Keep a separate port for the legacy
    # `jarvisx web` command while sharing one implementation.
    uvicorn.run("jarvisx.api:app", host="0.0.0.0", port=5000)
