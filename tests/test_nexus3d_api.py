from __future__ import annotations

from pathlib import Path

import pytest

import jarvisx.nexus3d_api as nexus
from jarvisx.nexus3d_api import (
    ChatRequest,
    ImageRequest,
    InlineImage,
    ProviderError,
    TTSRequest,
)


def test_capabilities_never_expose_provider_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nexus.gateway, "api_key", "super-secret-test-key")

    payload = nexus.capabilities()

    assert payload["configured"] is True
    assert "super-secret-test-key" not in repr(payload)
    assert payload["gui_contract"]["credential_exposure_to_browser"] is False
    assert payload["gui_contract"]["authoritative_state_mutation"] is False


def test_chat_shapes_grounded_vision_request_and_filters_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_generate(model: str, payload: dict[str, object]) -> dict[str, object]:
        captured["model"] = model
        captured["payload"] = payload
        return {
            "candidates": [
                {
                    "content": {"parts": [{"text": "grounded response"}]},
                    "groundingMetadata": {
                        "groundingChunks": [
                            {"web": {"title": "Good", "uri": "https://example.com/a"}},
                            {"web": {"title": "Bad", "uri": "javascript:alert(1)"}},
                        ]
                    },
                }
            ]
        }

    monkeypatch.setattr(nexus.gateway, "generate", fake_generate)

    response = nexus.chat(
        ChatRequest(
            prompt="inspect",
            image=InlineImage(mime_type="image/png", data="aGVsbG8="),
            grounding=True,
        )
    )

    assert response["text"] == "grounded response"
    assert response["sources"] == [
        {"title": "Good", "uri": "https://example.com/a"}
    ]
    assert captured["model"] == nexus.TEXT_MODEL
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["tools"] == [{"google_search": {}}]
    contents = payload["contents"]
    assert isinstance(contents, list)
    parts = contents[0]["parts"]
    assert parts[0] == {"text": "inspect"}
    assert parts[1]["inlineData"]["mimeType"] == "image/png"


def test_chat_requires_prompt_or_image() -> None:
    with pytest.raises(Exception):
        nexus.chat(ChatRequest(prompt="", image=None, grounding=False))


def test_image_and_tts_extract_inline_media(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_generate(model: str, payload: dict[str, object]) -> dict[str, object]:
        calls.append(model)
        if model == nexus.IMAGE_MODEL:
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "image note"},
                                {
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": "aW1hZ2U=",
                                    }
                                },
                            ]
                        }
                    }
                ]
            }
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "audio/L16;rate=24000",
                                    "data": "AAAAAA==",
                                }
                            }
                        ]
                    }
                }
            ]
        }

    monkeypatch.setattr(nexus.gateway, "generate", fake_generate)

    image = nexus.image(ImageRequest(prompt="a geometric lattice"))
    audio = nexus.tts(TTSRequest(text="hello", voice="Zephyr"))

    assert image["image"]["mime_type"] == "image/png"
    assert image["text"] == "image note"
    assert audio["audio"]["mime_type"].startswith("audio/")
    assert calls == [nexus.IMAGE_MODEL, nexus.TTS_MODEL]


def test_gateway_fails_closed_without_server_key() -> None:
    gateway = nexus.GeminiGateway(api_key=None)
    gateway.api_key = None

    with pytest.raises(ProviderError, match="not configured"):
        gateway.generate("model", {"contents": []})


def test_frontend_uses_same_origin_proxy_and_contains_no_provider_key_literal() -> None:
    root = Path(__file__).resolve().parents[1]
    page = (root / "apps/nexus-3d/index.html").read_text(encoding="utf-8")

    assert "/api/nexus3d/chat" in page
    assert "/api/nexus3d/image" in page
    assert "/api/nexus3d/tts" in page
    assert "generativelanguage.googleapis.com" not in page
    assert "const apiKey" not in page
    assert "MEASURED AUDIO SPECTRUM" in page


def test_inline_image_rejects_invalid_base64() -> None:
    with pytest.raises(ValueError):
        InlineImage(mime_type="image/png", data="not-base64%%%")


def test_repository_and_packaged_frontend_are_identical() -> None:
    root = Path(__file__).resolve().parents[1]
    repository_page = (root / "apps/nexus-3d/index.html").read_text(encoding="utf-8")
    packaged_page = (
        root / "src/jarvisx/_static/nexus3d/index.html"
    ).read_text(encoding="utf-8")

    assert packaged_page == repository_page
