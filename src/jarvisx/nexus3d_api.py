"""Same-origin backend for the NEXUS-3D multimodal workspace.

Provider credentials remain server-side. The browser issues bounded requests to
this FastAPI service, which forwards only the supported Gemini generateContent
operations. The service is a GUI/media adapter, not an authoritative Jarvis-X
state transition path.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
from pathlib import Path
import time
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[2]
REPO_STATIC_DIR = ROOT / "apps/nexus-3d"
PACKAGE_STATIC_DIR = Path(__file__).resolve().parent / "_static/nexus3d"
DEFAULT_STATIC_DIR = REPO_STATIC_DIR if REPO_STATIC_DIR.is_dir() else PACKAGE_STATIC_DIR
STATIC_DIR = Path(
    os.getenv("JARVISX_NEXUS3D_STATIC_DIR", str(DEFAULT_STATIC_DIR))
)

PROVIDER_BASE_URL = os.getenv(
    "JARVISX_NEXUS3D_PROVIDER_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta",
).rstrip("/")
TEXT_MODEL = os.getenv("JARVISX_NEXUS3D_TEXT_MODEL", "gemini-3-flash-preview")
IMAGE_MODEL = os.getenv("JARVISX_NEXUS3D_IMAGE_MODEL", "gemini-3.1-flash-image")
TTS_MODEL = os.getenv("JARVISX_NEXUS3D_TTS_MODEL", "gemini-2.5-flash-preview-tts")
PROVIDER_TIMEOUT_SECONDS = float(
    os.getenv("JARVISX_NEXUS3D_PROVIDER_TIMEOUT_SECONDS", "45")
)
MAX_PROVIDER_RESPONSE_BYTES = int(
    os.getenv("JARVISX_NEXUS3D_MAX_PROVIDER_RESPONSE_BYTES", str(24 * 1024 * 1024))
)
MAX_IMAGE_BYTES = int(
    os.getenv("JARVISX_NEXUS3D_MAX_IMAGE_BYTES", str(8 * 1024 * 1024))
)

app = FastAPI(
    title="NEXUS-3D Multimodal Workspace",
    version="1.0.0",
    description="Bounded same-origin multimedia GUI adapter for Jarvis-X.",
)

if STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="nexus3d-assets")


class ProviderError(RuntimeError):
    """A bounded upstream provider call failed."""


class InlineImage(BaseModel):
    mime_type: str = Field(pattern=r"^image/[a-zA-Z0-9.+-]+$", max_length=80)
    data: str = Field(min_length=1, max_length=16 * 1024 * 1024)

    @field_validator("data")
    @classmethod
    def validate_base64_image(cls, value: str) -> str:
        try:
            decoded = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("image data must be valid base64") from exc
        if len(decoded) > MAX_IMAGE_BYTES:
            raise ValueError("image exceeds configured upload limit")
        return value


class ChatRequest(BaseModel):
    prompt: str = Field("", max_length=32_768)
    image: InlineImage | None = None
    grounding: bool = False

    @field_validator("prompt")
    @classmethod
    def require_prompt_or_image(cls, value: str) -> str:
        return value.strip()


class ImageRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8_192)


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8_192)
    voice: str = Field("Zephyr", min_length=1, max_length=64)


class GeminiGateway:
    """Small synchronous gateway with explicit request and response bounds."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = PROVIDER_BASE_URL,
        timeout_seconds: float = PROVIDER_TIMEOUT_SECONDS,
        max_response_bytes: int = MAX_PROVIDER_RESPONSE_BYTES,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv(
            "GOOGLE_GEMINI_API_KEY"
        )
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max_response_bytes

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def generate(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        api_key = self.api_key
        if not api_key:
            raise ProviderError(
                "Gemini provider is not configured; set GEMINI_API_KEY "
                "or GOOGLE_GEMINI_API_KEY on the server"
            )
        url = f"{self.base_url}/models/{model}:generateContent"
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
                "User-Agent": "Jarvis-X-NEXUS3D/1.0",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as upstream:
                raw = upstream.read(self.max_response_bytes + 1)
                if len(raw) > self.max_response_bytes:
                    raise ProviderError("provider response exceeded configured limit")
        except HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            raise ProviderError(
                f"provider HTTP {exc.code}: {detail[:1000]}"
            ) from exc
        except URLError as exc:
            raise ProviderError(f"provider connection failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise ProviderError("provider request timed out") from exc

        try:
            payload_out = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError("provider returned invalid JSON") from exc
        if not isinstance(payload_out, dict):
            raise ProviderError("provider returned an unexpected JSON shape")
        return cast(dict[str, Any], payload_out)


gateway = GeminiGateway()


def _candidate(result: dict[str, Any]) -> dict[str, Any]:
    candidates = result.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        feedback = result.get("promptFeedback")
        raise ProviderError(f"provider returned no candidate: {feedback!r}")
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise ProviderError("provider candidate has invalid shape")
    return cast(dict[str, Any], candidate)


def _parts(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    content = candidate.get("content")
    if not isinstance(content, dict):
        return []
    raw_parts = content.get("parts")
    if not isinstance(raw_parts, list):
        return []
    return [cast(dict[str, Any], part) for part in raw_parts if isinstance(part, dict)]


def _text(candidate: dict[str, Any]) -> str:
    texts = [part.get("text") for part in _parts(candidate)]
    return "\n".join(str(item) for item in texts if isinstance(item, str)).strip()


def _inline_data(candidate: dict[str, Any]) -> dict[str, str] | None:
    for part in _parts(candidate):
        inline = part.get("inlineData")
        if not isinstance(inline, dict):
            continue
        data = inline.get("data")
        mime = inline.get("mimeType")
        if isinstance(data, str) and isinstance(mime, str):
            return {"data": data, "mime_type": mime}
    return None


def _grounding_sources(candidate: dict[str, Any]) -> list[dict[str, str]]:
    metadata = candidate.get("groundingMetadata")
    if not isinstance(metadata, dict):
        return []
    sources: list[dict[str, str]] = []
    seen: set[str] = set()

    chunks = metadata.get("groundingChunks")
    if isinstance(chunks, list):
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            web = chunk.get("web")
            if not isinstance(web, dict):
                continue
            uri = web.get("uri")
            title = web.get("title")
            if isinstance(uri, str) and uri.startswith(("https://", "http://")):
                if uri not in seen:
                    seen.add(uri)
                    sources.append(
                        {"uri": uri, "title": str(title or "Source")[:240]}
                    )

    attributions = metadata.get("groundingAttributions")
    if isinstance(attributions, list):
        for attribution in attributions:
            if not isinstance(attribution, dict):
                continue
            web = attribution.get("web")
            if not isinstance(web, dict):
                continue
            uri = web.get("uri")
            title = web.get("title")
            if isinstance(uri, str) and uri.startswith(("https://", "http://")):
                if uri not in seen:
                    seen.add(uri)
                    sources.append(
                        {"uri": uri, "title": str(title or "Source")[:240]}
                    )
    return sources[:12]


def _provider_http_error(exc: ProviderError) -> HTTPException:
    message = str(exc)
    status = 503 if "not configured" in message else 502
    return HTTPException(status_code=status, detail=message)


@app.middleware("http")
async def security_headers(request: Any, call_next: Any) -> Response:
    response = cast(Response, await call_next(request))
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), geolocation=(), microphone=(self)"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com "
        "https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com "
        "https://fonts.googleapis.com; "
        "font-src 'self' https://cdnjs.cloudflare.com https://fonts.gstatic.com data:; "
        "img-src 'self' data: blob:; "
        "media-src 'self' blob:; "
        "connect-src 'self'; "
        "object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
    )
    return response


@app.get("/", response_class=HTMLResponse)
def index() -> Response:
    page = STATIC_DIR / "index.html"
    if page.is_file():
        return FileResponse(page)
    return HTMLResponse(
        "<h1>NEXUS-3D</h1><p>Static bundle not found; API is operational.</p>"
    )


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "nexus-3d",
        "version": app.version,
        "time": time.time(),
        "provider_configured": gateway.configured,
        "static_bundle": STATIC_DIR.is_dir(),
        "authority": "gui_media_adapter_only",
    }


@app.get("/api/nexus3d/capabilities")
def capabilities() -> dict[str, Any]:
    return {
        "provider": "gemini_generate_content",
        "configured": gateway.configured,
        "models": {
            "text": TEXT_MODEL,
            "image": IMAGE_MODEL,
            "tts": TTS_MODEL,
        },
        "features": {
            "chat": True,
            "vision": True,
            "search_grounding": True,
            "image_generation": True,
            "tts": True,
            "browser_speech_input": True,
        },
        "limits": {
            "image_bytes": MAX_IMAGE_BYTES,
            "provider_response_bytes": MAX_PROVIDER_RESPONSE_BYTES,
            "provider_timeout_seconds": PROVIDER_TIMEOUT_SECONDS,
        },
        "gui_contract": {
            "authoritative_state_mutation": False,
            "credential_exposure_to_browser": False,
            "provider_calls_same_origin": True,
        },
    }


@app.post("/api/nexus3d/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    if not req.prompt and req.image is None:
        raise HTTPException(422, "prompt or image is required")

    parts: list[dict[str, Any]] = []
    if req.prompt:
        parts.append({"text": req.prompt})
    else:
        parts.append({"text": "Analyze the attached image."})
    if req.image is not None:
        parts.append(
            {
                "inlineData": {
                    "mimeType": req.image.mime_type,
                    "data": req.image.data,
                }
            }
        )

    payload: dict[str, Any] = {
        "contents": [{"parts": parts}],
        "systemInstruction": {
            "parts": [
                {
                    "text": (
                        "You are NEXUS-3D, a Jarvis-X multimodal workspace "
                        "assistant. Be clear, precise, and distinguish measured "
                        "facts from simulation or visualization."
                    )
                }
            ]
        },
    }
    if req.grounding:
        payload["tools"] = [{"google_search": {}}]

    try:
        candidate = _candidate(gateway.generate(TEXT_MODEL, payload))
        text = _text(candidate)
        if not text:
            raise ProviderError("provider returned an empty text response")
        return {"text": text, "sources": _grounding_sources(candidate)}
    except ProviderError as exc:
        raise _provider_http_error(exc) from exc


@app.post("/api/nexus3d/image")
def image(req: ImageRequest) -> dict[str, Any]:
    payload = {
        "contents": [{"parts": [{"text": req.prompt}]}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
    }
    try:
        candidate = _candidate(gateway.generate(IMAGE_MODEL, payload))
        image_data = _inline_data(candidate)
        if image_data is None or not image_data["mime_type"].startswith("image/"):
            raise ProviderError("provider returned no image")
        return {
            "image": image_data,
            "text": _text(candidate),
        }
    except ProviderError as exc:
        raise _provider_http_error(exc) from exc


@app.post("/api/nexus3d/tts")
def tts(req: TTSRequest) -> dict[str, Any]:
    payload = {
        "contents": [{"parts": [{"text": f"Say clearly: {req.text}"}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": req.voice}
                }
            },
        },
    }
    try:
        candidate = _candidate(gateway.generate(TTS_MODEL, payload))
        audio = _inline_data(candidate)
        if audio is None or not audio["mime_type"].startswith("audio/"):
            raise ProviderError("provider returned no audio")
        return {"audio": audio}
    except ProviderError as exc:
        raise _provider_http_error(exc) from exc
