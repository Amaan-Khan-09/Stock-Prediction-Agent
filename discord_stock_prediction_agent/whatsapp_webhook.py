"""Signed WhatsApp Cloud API webhook that feeds the shared durable signal queue."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .config import config
from .durable_signal_queue import enqueue_signal, queue_stats


LOGGER = logging.getLogger("discord_stock_prediction_agent.whatsapp")
_SERVER: ThreadingHTTPServer | None = None
_SERVER_LOCK = threading.Lock()


def _allowed(value: str, configured: str) -> bool:
    allowed = {item.strip() for item in str(configured or "").split(",") if item.strip()}
    return not allowed or str(value or "").strip() in allowed


def _message_text(message: dict[str, Any]) -> str:
    kind = str(message.get("type") or "").lower()
    if kind == "text":
        return str((message.get("text") or {}).get("body") or "").strip()
    if kind in {"image", "video", "document"}:
        return str((message.get(kind) or {}).get("caption") or "").strip()
    if kind == "button":
        return str((message.get("button") or {}).get("text") or "").strip()
    if kind == "interactive":
        interactive = message.get("interactive") or {}
        reply = interactive.get("button_reply") or interactive.get("list_reply") or {}
        return str(reply.get("title") or reply.get("id") or "").strip()
    return ""


def _extract_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    extracted: list[dict[str, Any]] = []
    for entry in payload.get("entry") or []:
        for change in entry.get("changes") or []:
            value = change.get("value") or {}
            for message in value.get("messages") or []:
                text = _message_text(message)
                if not text:
                    continue
                group_id = str(message.get("group_id") or value.get("group_id") or "").strip()
                sender = str(message.get("from") or "").strip()
                target = group_id or sender
                extracted.append(
                    {
                        "id": str(message.get("id") or ""),
                        "sender": sender,
                        "target": target,
                        "group_id": group_id,
                        "text": text,
                    }
                )
    return extracted


def verify_webhook_signature(body: bytes, signature: str) -> bool:
    if not config.whatsapp_app_secret or not signature:
        return False
    expected = "sha256=" + hmac.new(
        config.whatsapp_app_secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(str(signature), expected)


class WhatsAppWebhookServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256


class WhatsAppWebhookHandler(BaseHTTPRequestHandler):
    server_version = "StockPredictionWhatsApp/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("WhatsApp webhook: " + format, *args)

    def _reply(self, status: int, body: str, content_type: str = "text/plain") -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/healthz":
            stats = queue_stats()
            body = json.dumps(
                {
                    "status": "ok",
                    "transport": "whatsapp",
                    "configured": config.has_whatsapp,
                    "queue": stats,
                }
            )
            self._reply(HTTPStatus.OK, body, "application/json")
            return
        if parsed.path != "/webhook":
            self._reply(HTTPStatus.NOT_FOUND, "Not found")
            return
        query = parse_qs(parsed.query)
        mode = (query.get("hub.mode") or [""])[0]
        token = (query.get("hub.verify_token") or [""])[0]
        challenge = (query.get("hub.challenge") or [""])[0]
        if mode == "subscribe" and hmac.compare_digest(token, config.whatsapp_verify_token):
            self._reply(HTTPStatus.OK, challenge)
        else:
            self._reply(HTTPStatus.FORBIDDEN, "Verification failed")

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/webhook":
            self._reply(HTTPStatus.NOT_FOUND, "Not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._reply(HTTPStatus.BAD_REQUEST, "Invalid content length")
            return
        if length <= 0 or length > max(1_024, config.whatsapp_max_body_bytes):
            self._reply(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Invalid body size")
            return
        body = self.rfile.read(length)
        signature = self.headers.get("X-Hub-Signature-256", "")
        if not verify_webhook_signature(body, signature):
            self._reply(HTTPStatus.UNAUTHORIZED, "Invalid signature")
            return
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._reply(HTTPStatus.BAD_REQUEST, "Invalid JSON")
            return

        try:
            for item in _extract_messages(payload):
                if not _allowed(item["sender"], config.whatsapp_allowed_sender_ids):
                    continue
                if item["group_id"] and not _allowed(
                    item["group_id"], config.whatsapp_allowed_group_ids
                ):
                    continue
                enqueue_signal(
                    item["text"],
                    item["sender"],
                    item["target"],
                    item["id"],
                    config.signal_queue_limit,
                    transport="whatsapp",
                    reply_target=item["target"],
                    is_group=bool(item["group_id"]),
                )
        except Exception as exc:
            LOGGER.error("WhatsApp event could not be persisted: %s", type(exc).__name__)
            # A non-2xx response asks Meta to redeliver the same idempotent message.
            self._reply(HTTPStatus.SERVICE_UNAVAILABLE, "Queue unavailable")
            return
        # Meta expects a quick acknowledgment; processing occurs in the shared worker.
        self._reply(HTTPStatus.OK, "EVENT_RECEIVED")


def start_whatsapp_webhook_server() -> ThreadingHTTPServer:
    global _SERVER
    with _SERVER_LOCK:
        if _SERVER is not None:
            return _SERVER
        server = WhatsAppWebhookServer(
            (config.whatsapp_host, config.whatsapp_port), WhatsAppWebhookHandler
        )
        thread = threading.Thread(
            target=server.serve_forever,
            name="whatsapp-webhook",
            daemon=True,
        )
        thread.start()
        _SERVER = server
        LOGGER.info(
            "WhatsApp webhook listening on %s:%s",
            config.whatsapp_host,
            config.whatsapp_port,
        )
        return server


def main() -> None:
    if not config.has_whatsapp:
        raise SystemExit(
            "WhatsApp configuration is incomplete. Set WHATSAPP_VERIFY_TOKEN, "
            "WHATSAPP_ACCESS_TOKEN, WHATSAPP_PHONE_NUMBER_ID, and WHATSAPP_APP_SECRET."
        )
    server = WhatsAppWebhookServer(
        (config.whatsapp_host, config.whatsapp_port), WhatsAppWebhookHandler
    )
    LOGGER.info(
        "WhatsApp webhook listening on %s:%s",
        config.whatsapp_host,
        config.whatsapp_port,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
