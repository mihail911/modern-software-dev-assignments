#!/usr/bin/env python3
"""Metering gateway: the one path from a coding agent to a model.

Agents talk to this process instead of to a provider. It

  * forwards requests to Anthropic (``/anthropic/...``) or OpenAI (``/openai/...``),
    and, in practice only (``--allow-login``), to the ChatGPT backend that Codex uses
    with a ChatGPT login (``/chatgpt/...``),
  * refuses any model that is not on the rate card,
  * records the token usage of every response, priced from the rate card,
  * stops forwarding once a run has spent its budget.

Credentials: when the gateway has a provider key (``ANTHROPIC_API_KEY`` /
``OPENAI_API_KEY`` in its environment or ``--secrets`` file) it replaces
whatever the client sent, so the agent only ever holds a placeholder. Without
one it passes the client's own credentials through unchanged, which is how a
subscription login is metered during practice.

Standard library only, so the grader image and a student laptop run the same
file.
"""

import argparse
import http.client
import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import costs  # noqa: E402

PROVIDERS = {
    "anthropic": {
        "upstream_env": "ANTHROPIC_UPSTREAM",
        "upstream": "https://api.anthropic.com",
        "key_env": "ANTHROPIC_API_KEY",
        # POST to a metered path must name a rate-card model and is priced.
        # Free paths are forwarded as they are. Everything else is a 404.
        "metered": ("/v1/messages",),
        "free": ("/v1/messages/count_tokens", "/v1/models"),
    },
    "openai": {
        "upstream_env": "OPENAI_UPSTREAM",
        "upstream": "https://api.openai.com",
        "key_env": "OPENAI_API_KEY",
        "metered": ("/v1/responses", "/v1/responses/compact", "/v1/chat/completions"),
        "free": ("/v1/responses/input_tokens", "/v1/models"),
    },
    # Codex signed in with ChatGPT. Its own login is passed through and metered against
    # the OpenAI models on the rate card. Practice only: never enabled in grading.
    "chatgpt": {
        "upstream_env": "CHATGPT_UPSTREAM",
        "upstream": "https://chatgpt.com/backend-api/codex",
        "key_env": "",
        "rate_provider": "openai",
        "login_only": True,
        "metered": ("/responses", "/responses/compact"),
        "free": ("/models",),
    },
}
HOP_HEADERS = {
    "host", "content-length", "connection", "keep-alive", "accept-encoding",
    "transfer-encoding", "te", "trailer", "upgrade", "proxy-authorization", "proxy-connection",
}  # fmt: skip


def anthropic_usage(u: dict) -> dict:
    usage = costs.empty_usage()
    usage["input"] = u.get("input_tokens") or 0
    usage["output"] = u.get("output_tokens") or 0
    usage["cache_read"] = u.get("cache_read_input_tokens") or 0
    written = u.get("cache_creation_input_tokens") or 0
    one_hour = (u.get("cache_creation") or {}).get("ephemeral_1h_input_tokens") or 0
    usage["cache_write_1h"] = min(one_hour, written)
    usage["cache_write"] = written - usage["cache_write_1h"]
    return usage


def openai_usage(u: dict) -> dict:
    """Responses API (input_tokens) or Chat Completions (prompt_tokens) usage."""
    usage = costs.empty_usage()
    total_in = u.get("input_tokens", u.get("prompt_tokens")) or 0
    details = u.get("input_tokens_details") or u.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") or 0
    written = details.get("cache_write_tokens") or details.get("cache_creation_tokens") or 0
    usage["cache_read"] = cached
    usage["cache_write"] = written
    usage["input"] = max(total_in - cached - written, 0)
    usage["output"] = u.get("output_tokens", u.get("completion_tokens")) or 0
    return usage


class UsageReader:
    """Pulls the final usage out of a JSON body or a server-sent event stream."""

    def __init__(self, provider: str, streaming: bool):
        self.provider = provider
        self.streaming = streaming
        self.raw = {}
        self._buffer = b""

    def feed(self, chunk: bytes) -> None:
        self._buffer += chunk
        if not self.streaming:
            return
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            line = line.strip()
            if line.startswith(b"data:"):
                self._event(line[5:].strip())

    def finish(self) -> "dict | None":
        if not self.streaming:
            try:
                body = json.loads(self._buffer or b"{}")
            except ValueError:
                # An event stream sent without its Content-Type (the ChatGPT backend does this).
                body = {}
                for line in self._buffer.split(b"\n"):
                    line = line.strip()
                    if line.startswith(b"data:"):
                        self._event(line[5:].strip())
            if isinstance(body, dict):
                self._take(body.get("usage"))
        if not self.raw:
            return None
        return anthropic_usage(self.raw) if self.provider == "anthropic" else openai_usage(self.raw)

    def _event(self, data: bytes) -> None:
        if not data or data == b"[DONE]":
            return
        try:
            event = json.loads(data)
        except ValueError:
            return
        if not isinstance(event, dict):
            return
        # Anthropic: message_start carries the input side, message_delta the running output.
        self._take((event.get("message") or {}).get("usage"))
        # OpenAI Responses: response.completed carries the whole response.
        self._take((event.get("response") or {}).get("usage"))
        self._take(event.get("usage"))

    def _take(self, usage) -> None:
        if isinstance(usage, dict):
            self.raw.update({k: v for k, v in usage.items() if v is not None})


class Gateway:
    def __init__(self, card: dict, log_path: Path, max_cost: float, settings: dict):
        self.card = card
        self.log_path = log_path
        self.max_cost = max_cost
        self.settings = settings
        self.allow_login = False
        self.spent = 0.0
        self.lock = threading.Lock()

    def setting(self, name: str, default: str = "") -> str:
        return self.settings.get(name) or os.environ.get(name) or default

    def log(self, **entry) -> None:
        entry["ts"] = round(time.time(), 3)
        with self.lock:
            self.spent += entry.get("cost_usd", 0.0)
            with open(self.log_path, "a") as handle:
                handle.write(json.dumps(entry) + "\n")

    def over_budget(self) -> bool:
        with self.lock:
            return bool(self.max_cost) and self.spent >= self.max_cost


def make_handler(gateway: Gateway):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep stderr quiet; the usage log is the record
            pass

        def _reply(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def _refuse(self, provider: str, status: int, reason: str, message: str) -> None:
            gateway.log(provider=provider, path=self.path, rejected=reason)
            if provider == "anthropic":
                kind = "not_found_error" if status == 404 else "permission_error"
                body = {"type": "error", "error": {"type": kind, "message": message}}
            else:
                body = {"error": {"message": message, "type": "invalid_request_error", "code": reason}}
            self._reply(status, body)

        def _body(self) -> bytes:
            if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
                data = b""
                while True:
                    size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                    if size == 0:
                        while self.rfile.readline().strip():
                            pass
                        return data
                    data += self.rfile.read(size)
                    self.rfile.readline()
            return self.rfile.read(int(self.headers.get("Content-Length") or 0))

        def _handle(self) -> None:
            parts = urlsplit(self.path)
            if parts.path == "/healthz":
                return self._reply(200, {"ok": True, "spent_usd": round(gateway.spent, 6)})
            provider, _, rest = parts.path.lstrip("/").partition("/")
            rest = "/" + rest
            if provider not in PROVIDERS or (PROVIDERS[provider].get("login_only") and not gateway.allow_login):
                return self._reply(404, {"error": "use /anthropic/... or /openai/..."})
            spec = PROVIDERS[provider]
            rates_from = spec.get("rate_provider", provider)
            body = self._body()

            if provider == "anthropic" and rest == "/api/hello":
                return self._reply(200, {"ok": True})  # Claude Code's startup probe
            metered = self.command == "POST" and rest in spec["metered"]
            free = any(rest == path or rest.startswith(path + "/") for path in spec["free"])
            if not metered and not free:
                return self._refuse(
                    provider, 404, "endpoint_not_available",
                    f"{rest} is not available through the course gateway",
                )  # fmt: skip

            model, streaming = None, False
            if metered:
                if self.headers.get("Content-Encoding", "identity").lower() != "identity":
                    return self._refuse(
                        provider, 415, "compressed_request",
                        "send request bodies uncompressed through the course gateway",
                    )  # fmt: skip
                try:
                    payload = json.loads(body)
                except ValueError:
                    payload = None
                requested = payload.get("model") if isinstance(payload, dict) else None
                model = costs.resolve_model(gateway.card, requested)
                if model is None or gateway.card["models"][model]["provider"] != rates_from:
                    allowed = ", ".join(
                        name for name, r in gateway.card["models"].items() if r["provider"] == rates_from
                    )
                    return self._refuse(
                        provider, 403, f"model_not_allowed:{requested}",
                        f"model {requested!r} is not on the rate card; allowed here: {allowed}",
                    )  # fmt: skip
                if gateway.over_budget():
                    return self._refuse(
                        provider, 403, "budget_exhausted",
                        f"this run has spent its ${gateway.max_cost:.2f} budget",
                    )  # fmt: skip
                streaming = bool(payload.get("stream"))
                if streaming and rest.startswith("/v1/chat/completions"):
                    # Chat Completions only reports usage on a stream when asked to.
                    payload.setdefault("stream_options", {})["include_usage"] = True
                    body = json.dumps(payload).encode()

            self._forward(provider, spec, rest, parts.query, body, metered, model, streaming)

        def _forward(self, provider, spec, rest, query, body, metered, model, streaming) -> None:
            upstream = urlsplit(gateway.setting(spec["upstream_env"], spec["upstream"]))
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
            key = gateway.setting(spec["key_env"]) if spec["key_env"] else ""
            if key:
                for name in [h for h in headers if h.lower() in ("authorization", "x-api-key")]:
                    del headers[name]
                if provider == "anthropic":
                    headers["x-api-key"] = key
                else:
                    headers["Authorization"] = f"Bearer {key}"
            headers["Accept-Encoding"] = "identity"
            headers["Host"] = upstream.netloc
            if body or self.command in ("POST", "PUT", "PATCH"):
                headers["Content-Length"] = str(len(body))

            connect = (
                http.client.HTTPSConnection if upstream.scheme == "https" else http.client.HTTPConnection
            )
            connection = connect(upstream.netloc, timeout=900)
            target = upstream.path.rstrip("/") + rest + (f"?{query}" if query else "")
            try:
                connection.request(self.command, target, body=body or None, headers=headers)
                response = connection.getresponse()
            except OSError as exc:
                connection.close()
                gateway.log(provider=provider, path=rest, model=model, metered=metered, status=502)
                return self._reply(502, {"error": f"gateway could not reach the provider: {exc}"})

            is_stream = "text/event-stream" in response.getheader("Content-Type", "")
            reader = UsageReader(provider, is_stream)

            def record() -> None:
                if metered:
                    usage = reader.finish() or costs.empty_usage()
                    gateway.log(
                        provider=provider, path=rest, model=model, metered=True,
                        status=response.status, stream=is_stream, usage=usage,
                        cost_usd=costs.cost_usd(gateway.card, model, usage),
                    )  # fmt: skip

            # Usage is always logged before the client sees the end of the
            # response, so a run that exits right after its last call is
            # still fully counted.
            data = b"" if is_stream else response.read()
            if not is_stream:
                reader.feed(data)
                record()
            self.send_response(response.status)
            for name, value in response.getheaders():
                if name.lower() not in HOP_HEADERS | {"content-encoding", "date", "server"}:
                    self.send_header(name, value)
            try:
                if is_stream:
                    # No length is known up front, so the stream ends when we close.
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    while True:
                        chunk = response.read1(65536)
                        if not chunk:
                            break
                        reader.feed(chunk)
                        try:
                            self.wfile.write(chunk)
                            self.wfile.flush()
                        except OSError:
                            pass  # the client went away; keep reading so usage is complete
                else:
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    if self.command != "HEAD":
                        self.wfile.write(data)
            except OSError:
                self.close_connection = True
            finally:
                if is_stream:
                    record()
                connection.close()

        do_GET = do_POST = do_HEAD = do_PUT = do_DELETE = do_PATCH = _handle

    return Handler


def read_secrets(path) -> dict:
    """KEY=VALUE lines; blank lines and # comments are skipped."""
    settings = {}
    if not path:
        return settings
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, _, value = line.partition("=")
            settings[name.strip()] = value.strip().strip("'\"")
    return settings


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--rate-card", default=str(here.parent / "rate_card.json"))
    parser.add_argument("--log", default="usage.jsonl", help="usage log to append to")
    parser.add_argument("--max-cost", type=float, default=0.0, help="USD budget; 0 for none")
    parser.add_argument("--secrets", help="file of KEY=VALUE settings, read instead of the environment")
    parser.add_argument("--allow-login", action="store_true", help="practice: also forward /chatgpt/ (a Codex ChatGPT login)")
    args = parser.parse_args(argv)

    gateway = Gateway(
        costs.load_rate_card(args.rate_card), Path(args.log), args.max_cost, read_secrets(args.secrets)
    )
    gateway.allow_login = args.allow_login
    server = ThreadingHTTPServer((args.host, args.port), make_handler(gateway))
    # Not daemon threads: on shutdown, requests in flight finish and are logged.
    server.daemon_threads = False
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start())
    for name, spec in PROVIDERS.items():
        if spec.get("login_only") and not gateway.allow_login:
            continue
        mode = "course key" if spec["key_env"] and gateway.setting(spec["key_env"]) else "client credentials passed through"
        print(f"gateway: /{name} -> {gateway.setting(spec['upstream_env'], spec['upstream'])} ({mode})")
    print(f"gateway: listening on http://{args.host}:{args.port}, logging to {args.log}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
