"""Model inference adapters for HyperCloud workers.

The local backend keeps the stack runnable without external credentials. The
OpenAI-compatible adapter can target a self-hosted or managed chat endpoint by
configuration, allowing the worker/data-plane boundary to remain provider
neutral.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ChatBackend(Protocol):
    name: str

    def generate(self, *, prompt: str, system: str | None = None) -> dict[str, object]: ...


@dataclass
class LocalReferenceBackend:
    """Deterministic offline backend used for smoke tests and local operation.

    This backend is deliberately labelled reference/non-LLM. It proves the job,
    worker and response path without making a neural-model capability claim.
    """

    name: str = "local-reference-non-llm"

    def generate(self, *, prompt: str, system: str | None = None) -> dict[str, object]:
        normalized = " ".join(prompt.strip().split())
        if not normalized:
            raise ValueError("prompt must not be empty")
        prefix = "Jarvis-X operational reference backend"
        if system:
            prefix += f" [{system.strip()[:160]}]"
        text = f"{prefix}: {normalized}"
        return {
            "backend": self.name,
            "model": "deterministic-reference",
            "text": text,
            "input_characters": len(prompt),
            "output_characters": len(text),
            "neural_model": False,
        }


@dataclass
class OpenAICompatibleBackend:
    """Minimal dependency-free client for OpenAI-compatible chat servers."""

    base_url: str
    model: str
    api_key: str | None = None
    timeout_seconds: float = 120.0
    name: str = "openai-compatible"

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("model base URL must start with http:// or https://")
        if not self.model:
            raise ValueError("model name must not be empty")

    def generate(self, *, prompt: str, system: str | None = None) -> dict[str, object]:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0.0,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{self.base_url}/v1/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise RuntimeError(f"model backend HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"model backend unavailable: {exc.reason}") from exc

        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("model backend returned an invalid chat-completions payload") from exc
        usage = payload.get("usage") if isinstance(payload, dict) else None
        return {
            "backend": self.name,
            "model": self.model,
            "text": text,
            "usage": usage,
            "neural_model": True,
        }


def backend_from_environment() -> ChatBackend:
    base_url = os.getenv("JARVISX_MODEL_BASE_URL", "").strip()
    model = os.getenv("JARVISX_MODEL_NAME", "").strip()
    if not base_url:
        return LocalReferenceBackend()
    if not model:
        raise RuntimeError("JARVISX_MODEL_NAME is required when JARVISX_MODEL_BASE_URL is set")
    timeout = float(os.getenv("JARVISX_MODEL_TIMEOUT_SECONDS", "120"))
    return OpenAICompatibleBackend(
        base_url=base_url,
        model=model,
        api_key=os.getenv("JARVISX_MODEL_API_KEY") or None,
        timeout_seconds=timeout,
    )
