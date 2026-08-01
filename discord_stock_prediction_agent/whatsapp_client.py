"""Small, transport-only client for the official WhatsApp Cloud API."""
from __future__ import annotations

import logging
from typing import Any

import requests

from .config import config


LOGGER = logging.getLogger("discord_stock_prediction_agent.whatsapp")


def _embed_text(embed: Any) -> str:
    if embed is None:
        return ""
    try:
        payload = embed.to_dict()
    except Exception:
        return str(embed or "")
    lines: list[str] = []
    if payload.get("title"):
        lines.append(str(payload["title"]))
    if payload.get("description"):
        lines.append(str(payload["description"]))
    for field in payload.get("fields") or []:
        name = str(field.get("name") or "").strip()
        value = str(field.get("value") or "").strip()
        if name and value:
            lines.append(f"{name}: {value}")
        elif value:
            lines.append(value)
    footer = (payload.get("footer") or {}).get("text")
    if footer:
        lines.append(str(footer))
    return "\n".join(lines)


def output_text(content: str = "", embed: Any = None) -> str:
    parts = [str(content or "").strip(), _embed_text(embed).strip()]
    return "\n\n".join(part for part in parts if part).strip()


def _chunks(text: str, limit: int = 3_800) -> list[str]:
    remaining = str(text or "").strip()
    chunks: list[str] = []
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        split = remaining.rfind("\n", 0, limit)
        if split < limit // 2:
            split = remaining.rfind(" ", 0, limit)
        if split < limit // 2:
            split = limit
        chunks.append(remaining[:split].strip())
        remaining = remaining[split:].strip()
    return chunks


def send_whatsapp_text(target: str, text: str, *, is_group: bool = False) -> tuple[bool, str]:
    if not config.has_whatsapp:
        return False, "WhatsApp Cloud API is not configured"
    target_id = str(target or "").strip()
    if not target_id:
        return False, "WhatsApp reply target is missing"
    endpoint = (
        "https://graph.facebook.com/"
        f"{config.whatsapp_graph_api_version}/{config.whatsapp_phone_number_id}/messages"
    )
    headers = {
        "Authorization": f"Bearer {config.whatsapp_access_token}",
        "Content-Type": "application/json",
    }
    for chunk in _chunks(text):
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "group" if is_group else "individual",
            "to": target_id,
            "type": "text",
            "text": {"preview_url": False, "body": chunk},
        }
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=max(5, config.alpaca_request_timeout_seconds),
            )
        except requests.RequestException as exc:
            return False, f"WhatsApp transport error: {type(exc).__name__}"
        if response.status_code >= 300:
            LOGGER.warning("WhatsApp send rejected with HTTP %s", response.status_code)
            return False, f"WhatsApp HTTP {response.status_code}"
    return True, ""

