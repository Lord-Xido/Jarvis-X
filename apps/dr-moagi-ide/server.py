"""Container/process entry point for Dr Moagi ANN IDE."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "jarvisx.dr_moagi_ide_api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        proxy_headers=True,
    )


if __name__ == "__main__":
    main()
