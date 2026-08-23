from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import json
import os
import random
import sqlite3
import time
import urllib.error
import urllib.request

from .config import ExperimentConfig, Price
from .util import canonical_json, estimate_tokens, sha256_json


class BudgetExceeded(RuntimeError):
    pass


class APIError(RuntimeError):
    pass


class APIHTTPError(APIError):
    """Sanitized OpenAI HTTP error that retains actionable response metadata."""

    def __init__(self, status: int, message: str, *, parameter: str | None = None,
                 code: str | None = None, request_id: str | None = None):
        self.status = status
        self.message = message
        self.parameter = parameter
        self.code = code
        self.request_id = request_id
        details = []
        if parameter:
            details.append(f"param={parameter}")
        if code:
            details.append(f"code={code}")
        if request_id:
            details.append(f"request_id={request_id}")
        suffix = f" ({', '.join(details)})" if details else ""
        super().__init__(f"OpenAI HTTP {status}: {message}{suffix}")


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass(frozen=True)
class APIResult:
    request_hash: str
    response_id: str
    requested_model: str
    returned_model: str
    output_text: str
    usage: Usage
    cost_usd: float
    latency_ms: int
    retries: int
    raw: dict[str, Any]

    def json(self) -> Any:
        try:
            return json.loads(self.output_text)
        except json.JSONDecodeError as exc:
            raise APIError(f"response was not valid JSON: {self.output_text[:160]!r}") from exc


Transport = Callable[[str, dict[str, Any], dict[str, str]], dict[str, Any]]


def calculate_cost(usage: Usage, price: Price) -> float:
    ordinary = max(0, usage.input_tokens - usage.cached_tokens - usage.cache_write_tokens)
    cache_write_rate = price.cache_write_per_million or price.input_per_million
    return (
        ordinary * price.input_per_million
        + usage.cached_tokens * price.cached_input_per_million
        + usage.cache_write_tokens * cache_write_rate
        + usage.output_tokens * price.output_per_million
    ) / 1_000_000


class APILedger:
    """Transactional request cache and spend-cap authority."""

    def __init__(self, path: Path, cap_usd: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.cap_usd = cap_usd
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS api_calls (
              request_hash TEXT PRIMARY KEY,
              purpose TEXT NOT NULL,
              endpoint TEXT NOT NULL,
              requested_model TEXT NOT NULL,
              returned_model TEXT,
              status TEXT NOT NULL,
              estimated_cost_usd REAL NOT NULL,
              actual_cost_usd REAL,
              response_id TEXT,
              response_json TEXT,
              input_tokens INTEGER DEFAULT 0,
              cached_tokens INTEGER DEFAULT 0,
              cache_write_tokens INTEGER DEFAULT 0,
              output_tokens INTEGER DEFAULT 0,
              reasoning_tokens INTEGER DEFAULT 0,
              latency_ms INTEGER DEFAULT 0,
              retries INTEGER DEFAULT 0,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              completed_at TEXT,
              error TEXT
            );
            """
        )
        self.db.commit()

    def totals(self) -> dict[str, float]:
        row = self.db.execute(
            """SELECT
              COALESCE(SUM(CASE WHEN status='complete' THEN actual_cost_usd ELSE 0 END),0) actual,
              COALESCE(SUM(CASE WHEN status='reserved' THEN estimated_cost_usd ELSE 0 END),0) reserved
            FROM api_calls"""
        ).fetchone()
        return {"actual": float(row["actual"]), "reserved": float(row["reserved"])}

    def get_complete(self, request_hash: str) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT * FROM api_calls WHERE request_hash=? AND status='complete'",
            (request_hash,),
        ).fetchone()

    def reserve(self, request_hash: str, purpose: str, endpoint: str,
                model: str, estimated_cost: float) -> None:
        with self.db:
            existing = self.db.execute(
                "SELECT status FROM api_calls WHERE request_hash=?", (request_hash,)
            ).fetchone()
            if existing and existing["status"] == "complete":
                return
            totals = self.totals()
            old_reservation = 0.0
            if existing and existing["status"] == "reserved":
                row = self.db.execute(
                    "SELECT estimated_cost_usd FROM api_calls WHERE request_hash=?", (request_hash,)
                ).fetchone()
                old_reservation = float(row[0])
            projected = totals["actual"] + totals["reserved"] - old_reservation + estimated_cost
            if projected > self.cap_usd + 1e-12:
                raise BudgetExceeded(
                    f"request would exceed ${self.cap_usd:.2f} cap "
                    f"(${projected:.6f} projected)"
                )
            self.db.execute(
                """INSERT INTO api_calls
                   (request_hash,purpose,endpoint,requested_model,status,estimated_cost_usd)
                   VALUES (?,?,?,?, 'reserved',?)
                   ON CONFLICT(request_hash) DO UPDATE SET
                     purpose=excluded.purpose, endpoint=excluded.endpoint,
                     requested_model=excluded.requested_model, status='reserved',
                     estimated_cost_usd=excluded.estimated_cost_usd, error=NULL""",
                (request_hash, purpose, endpoint, model, estimated_cost),
            )

    def complete(self, request_hash: str, response: dict[str, Any], model: str,
                 usage: Usage, cost: float, latency_ms: int, retries: int) -> None:
        response_id = str(response.get("id", ""))
        returned_model = str(response.get("model", model))
        with self.db:
            self.db.execute(
                """UPDATE api_calls SET status='complete', actual_cost_usd=?,
                   returned_model=?, response_id=?, response_json=?, input_tokens=?,
                   cached_tokens=?, cache_write_tokens=?, output_tokens=?,
                   reasoning_tokens=?, latency_ms=?, retries=?, completed_at=CURRENT_TIMESTAMP
                   WHERE request_hash=?""",
                (cost, returned_model, response_id, canonical_json(response),
                 usage.input_tokens, usage.cached_tokens, usage.cache_write_tokens,
                 usage.output_tokens, usage.reasoning_tokens, latency_ms, retries,
                 request_hash),
            )

    def fail(self, request_hash: str, error: str, retries: int) -> None:
        with self.db:
            self.db.execute(
                "UPDATE api_calls SET status='failed', error=?, retries=? WHERE request_hash=?",
                (error[:1000], retries, request_hash),
            )


class OpenAIClient:
    base_url = "https://api.openai.com/v1"

    def __init__(self, config: ExperimentConfig, ledger: APILedger,
                 api_key: str | None = None, transport: Transport | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.config = config
        self.ledger = ledger
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.transport = transport or self._http_transport
        self.sleep = sleep
        self.response_compatibility: dict[str, set[str]] = {}

    def _http_transport(self, url: str, payload: dict[str, Any],
                        headers: dict[str, str]) -> dict[str, Any]:
        if not self.api_key:
            raise APIError("OPENAI_API_KEY is required")
        request = urllib.request.Request(
            url, data=canonical_json(payload).encode(), method="POST",
            headers={**headers, "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            request_id = exc.headers.get("x-request-id") if exc.headers else None
            try:
                body = json.loads(exc.read().decode("utf-8", errors="replace"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                body = {}
            error = body.get("error") if isinstance(body, dict) else None
            error = error if isinstance(error, dict) else {}
            message = str(error.get("message") or exc.reason or "request rejected")
            parameter = error.get("param")
            code = error.get("code")
            raise APIHTTPError(
                exc.code, message, parameter=str(parameter) if parameter else None,
                code=str(code) if code else None, request_id=request_id,
            ) from exc

    def _call(self, endpoint: str, payload: dict[str, Any], purpose: str,
              estimated_usage: Usage, max_retries: int = 5) -> APIResult:
        model = str(payload["model"])
        if model not in self.config.prices:
            raise APIError(f"no pinned price for model {model}")
        request_hash = sha256_json({"endpoint": endpoint, "payload": payload, "purpose": purpose})
        cached = self.ledger.get_complete(request_hash)
        if cached:
            raw = json.loads(cached["response_json"])
            return self._result_from_row(cached, raw)
        estimate = calculate_cost(estimated_usage, self.config.prices[model])
        self.ledger.reserve(request_hash, purpose, endpoint, model, estimate)
        started = time.monotonic()
        retries = 0
        headers = {"Content-Type": "application/json", "Idempotency-Key": request_hash}
        while True:
            try:
                response = self.transport(f"{self.base_url}{endpoint}", payload, headers)
                usage = parse_usage(response)
                cost = calculate_cost(usage, self.config.prices[model])
                latency = round((time.monotonic() - started) * 1000)
                self.ledger.complete(request_hash, response, model, usage, cost, latency, retries)
                row = self.ledger.get_complete(request_hash)
                assert row is not None
                return self._result_from_row(row, response)
            except APIHTTPError as exc:
                retryable = exc.status in (408, 409, 429) or exc.status >= 500
                if retries >= max_retries or not retryable:
                    self.ledger.fail(request_hash, str(exc), retries)
                    raise
                delay = min(30.0, (2 ** retries) + random.Random(request_hash + str(retries)).random())
                retries += 1
                self.sleep(delay)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, APIError) as exc:
                retryable = not isinstance(exc, urllib.error.HTTPError) or exc.code in (408, 409, 429) or exc.code >= 500
                if retries >= max_retries or not retryable:
                    self.ledger.fail(request_hash, repr(exc), retries)
                    raise APIError(f"OpenAI request failed after {retries} retries: {exc}") from exc
                delay = min(30.0, (2 ** retries) + random.Random(request_hash + str(retries)).random())
                retries += 1
                self.sleep(delay)
            except Exception as exc:
                self.ledger.fail(request_hash, repr(exc), retries)
                raise

    def _result_from_row(self, row: sqlite3.Row, raw: dict[str, Any]) -> APIResult:
        usage = Usage(
            int(row["input_tokens"]), int(row["cached_tokens"]),
            int(row["cache_write_tokens"]), int(row["output_tokens"]),
            int(row["reasoning_tokens"]),
        )
        return APIResult(
            row["request_hash"], row["response_id"] or "", row["requested_model"],
            row["returned_model"] or row["requested_model"], extract_output_text(raw), usage,
            float(row["actual_cost_usd"] or 0), int(row["latency_ms"]),
            int(row["retries"]), raw,
        )

    def respond_json(self, *, model: str, instructions: str, input_text: str,
                     schema_name: str, schema: dict[str, Any], purpose: str,
                     max_output_tokens: int | None = None) -> APIResult:
        max_output = max_output_tokens or self.config.max_output_tokens
        payload = {
            "model": model,
            "instructions": instructions,
            "input": input_text,
            "temperature": 0,
            "max_output_tokens": max_output,
            "store": False,
            "reasoning": {"effort": self.config.reasoning_effort, "context": "current_turn"},
            "text": {"format": {"type": "json_schema", "name": schema_name,
                     "strict": True, "schema": schema}},
        }
        estimate = Usage(input_tokens=estimate_tokens(instructions + input_text) + 100,
                         output_tokens=max_output)
        output_limit = max_output
        output_retry = 0
        known_adjustments = self.response_compatibility.setdefault(model, set())
        adjustments = [name for name in ("no-reasoning-context", "no-temperature")
                       if name in known_adjustments]
        if "no-reasoning-context" in known_adjustments:
            payload = {**payload, "reasoning": {**payload["reasoning"]}}
            payload["reasoning"].pop("context", None)
        if "no-temperature" in known_adjustments:
            payload = {key: value for key, value in payload.items() if key != "temperature"}
        while True:
            adjusted_purpose = purpose
            if adjustments:
                adjusted_purpose += ":compat:" + "+".join(adjustments)
            if output_retry:
                adjusted_purpose += f":output-retry-{output_limit}"
            try:
                result = self._call("/responses", payload, adjusted_purpose, estimate)
                incomplete_reason = (
                    (result.raw.get("incomplete_details") or {}).get("reason")
                    if result.raw.get("status") == "incomplete" else None
                )
                valid_json = True
                try:
                    json.loads(result.output_text)
                except json.JSONDecodeError:
                    valid_json = False
                if incomplete_reason == "max_output_tokens" or not valid_json:
                    if output_limit >= 4_000:
                        detail = incomplete_reason or "invalid structured JSON"
                        raise APIError(
                            f"structured response remained incomplete at {output_limit} "
                            f"output tokens: {detail}"
                        )
                    output_limit = min(4_000, max(800, output_limit * 2))
                    output_retry += 1
                    payload = {**payload, "max_output_tokens": output_limit}
                    estimate = Usage(
                        input_tokens=estimate.input_tokens, output_tokens=output_limit,
                    )
                    continue
                if incomplete_reason:
                    raise APIError(f"OpenAI response incomplete: {incomplete_reason}")
                return result
            except APIHTTPError as exc:
                rejected = f"{exc.parameter or ''} {exc.message}".casefold()
                if ("context" in payload.get("reasoning", {}) and
                        "reasoning" in rejected and "context" in rejected and
                        unsupported_parameter_message(rejected)):
                    payload = {**payload, "reasoning": {**payload["reasoning"]}}
                    payload["reasoning"].pop("context", None)
                    adjustments.append("no-reasoning-context")
                    known_adjustments.add("no-reasoning-context")
                    continue
                if ("temperature" in payload and "temperature" in rejected and
                        unsupported_parameter_message(rejected)):
                    payload = {key: value for key, value in payload.items() if key != "temperature"}
                    adjustments.append("no-temperature")
                    known_adjustments.add("no-temperature")
                    continue
                raise

    def embed(self, texts: list[str], purpose: str) -> tuple[list[list[float]], APIResult]:
        payload = {"model": self.config.embedding_model, "input": texts, "encoding_format": "float"}
        estimate = Usage(input_tokens=sum(estimate_tokens(text) for text in texts) + 20)
        result = self._call("/embeddings", payload, purpose, estimate)
        vectors = [row["embedding"] for row in sorted(result.raw.get("data", []), key=lambda r: r["index"])]
        if len(vectors) != len(texts):
            raise APIError("embedding response length mismatch")
        return vectors, result


def parse_usage(response: dict[str, Any]) -> Usage:
    raw = response.get("usage") or {}
    inp = raw.get("input_tokens_details") or {}
    out = raw.get("output_tokens_details") or {}
    return Usage(
        input_tokens=int(raw.get("input_tokens", raw.get("prompt_tokens", 0)) or 0),
        cached_tokens=int(inp.get("cached_tokens", raw.get("cached_tokens", 0)) or 0),
        cache_write_tokens=int(inp.get("cache_write_tokens", raw.get("cache_write_tokens", 0)) or 0),
        output_tokens=int(raw.get("output_tokens", raw.get("completion_tokens", 0)) or 0),
        reasoning_tokens=int(out.get("reasoning_tokens", raw.get("reasoning_tokens", 0)) or 0),
    )


def extract_output_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    pieces: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in ("output_text", "text") and isinstance(content.get("text"), str):
                pieces.append(content["text"])
    return "".join(pieces)


def unsupported_parameter_message(value: str) -> bool:
    return any(marker in value for marker in (
        "unknown parameter", "unsupported parameter", "not supported", "unrecognized",
        "extra inputs are not permitted", "invalid parameter",
    ))
