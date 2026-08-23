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
import urllib.parse
import urllib.request

from .config import ExperimentConfig, Price
from .util import canonical_json, estimate_tokens, sha256_json


class BudgetExceeded(RuntimeError):
    pass


class APIError(RuntimeError):
    pass


class APIHTTPError(APIError):
    """Sanitized provider HTTP error that retains actionable response metadata."""

    def __init__(self, status: int, message: str, *, parameter: str | None = None,
                 code: str | None = None, request_id: str | None = None,
                 provider: str = "API"):
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
        super().__init__(f"{provider} HTTP {status}: {message}{suffix}")


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
            return parse_json_text(self.output_text)
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
    ) / 1_000_000 + price.request_usd


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
              provider TEXT NOT NULL DEFAULT 'openai',
              purpose TEXT NOT NULL,
              endpoint TEXT NOT NULL,
              requested_model TEXT NOT NULL,
              returned_model TEXT,
              status TEXT NOT NULL,
              input_price_per_million REAL,
              cached_input_price_per_million REAL,
              output_price_per_million REAL,
              cache_write_price_per_million REAL,
              request_price_usd REAL,
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
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(api_calls)")}
        migrations = {
            "provider": "TEXT NOT NULL DEFAULT 'openai'",
            "input_price_per_million": "REAL",
            "cached_input_price_per_million": "REAL",
            "output_price_per_million": "REAL",
            "cache_write_price_per_million": "REAL",
            "request_price_usd": "REAL",
        }
        for name, declaration in migrations.items():
            if name not in columns:
                self.db.execute(f"ALTER TABLE api_calls ADD COLUMN {name} {declaration}")
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
                model: str, estimated_cost: float, provider: str = "openai",
                price: Price | None = None) -> None:
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
                   (request_hash,provider,purpose,endpoint,requested_model,status,
                    input_price_per_million,cached_input_price_per_million,
                    output_price_per_million,cache_write_price_per_million,request_price_usd,
                    estimated_cost_usd)
                   VALUES (?,?,?,?,?, 'reserved',?,?,?,?,?,?)
                   ON CONFLICT(request_hash) DO UPDATE SET
                     provider=excluded.provider, purpose=excluded.purpose, endpoint=excluded.endpoint,
                     requested_model=excluded.requested_model, status='reserved',
                     input_price_per_million=excluded.input_price_per_million,
                     cached_input_price_per_million=excluded.cached_input_price_per_million,
                     output_price_per_million=excluded.output_price_per_million,
                     cache_write_price_per_million=excluded.cache_write_price_per_million,
                     request_price_usd=excluded.request_price_usd,
                     estimated_cost_usd=excluded.estimated_cost_usd, error=NULL""",
                (
                    request_hash, provider, purpose, endpoint, model,
                    price.input_per_million if price else None,
                    price.cached_input_per_million if price else None,
                    price.output_per_million if price else None,
                    price.cache_write_per_million if price else None,
                    price.request_usd if price else None,
                    estimated_cost,
                ),
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


class LLMClient:
    """Provider-neutral client for direct OpenAI and OpenRouter model calls."""

    def __init__(self, config: ExperimentConfig, ledger: APILedger,
                 api_key: str | None = None, transport: Transport | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 catalog_transport: Callable[[str, dict[str, str]], dict[str, Any]] | None = None):
        self.config = config
        self.ledger = ledger
        self.base_url = config.api_base_url.rstrip("/")
        self.api_key = (
            api_key if api_key is not None else os.environ.get(config.api_key_env, "")
        )
        self.transport = transport or self._http_transport
        self.catalog_transport = catalog_transport or self._http_catalog_transport
        self.sleep = sleep
        self.response_compatibility: dict[str, set[str]] = {}
        self.discovered_prices: dict[str, Price] = {}

    def _headers(self, request_hash: str | None = None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if request_hash:
            headers["Idempotency-Key"] = request_hash
        if self.config.provider == "openrouter":
            if self.config.app_url:
                headers["HTTP-Referer"] = self.config.app_url
            if self.config.app_title:
                headers["X-OpenRouter-Title"] = self.config.app_title
        return headers

    def _http_transport(self, url: str, payload: dict[str, Any],
                        headers: dict[str, str]) -> dict[str, Any]:
        if not self.api_key:
            raise APIError(f"{self.config.api_key_env} is required for {self.config.provider}")
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
                provider=self.config.provider,
            ) from exc

    def _http_catalog_transport(self, url: str, headers: dict[str, str]) -> dict[str, Any]:
        if not self.api_key:
            raise APIError(f"{self.config.api_key_env} is required for {self.config.provider}")
        request = urllib.request.Request(
            url, method="GET", headers={**headers, "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            try:
                body = json.loads(exc.read().decode("utf-8", errors="replace"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                body = {}
            error = body.get("error") if isinstance(body, dict) else None
            error = error if isinstance(error, dict) else {}
            raise APIHTTPError(
                exc.code, str(error.get("message") or exc.reason or "catalog request rejected"),
                code=str(error.get("code")) if error.get("code") else None,
                provider=self.config.provider,
            ) from exc

    def _price_for_model(self, model: str) -> Price:
        if model in self.config.prices:
            return self.config.prices[model]
        if model in self.discovered_prices:
            return self.discovered_prices[model]
        if self.config.provider != "openrouter":
            raise APIError(
                f"no pinned price for model {model}; configure RELIABMEM_INPUT_PRICE_PER_MILLION "
                "and RELIABMEM_OUTPUT_PRICE_PER_MILLION"
            )
        encoded_model = urllib.parse.quote(model, safe="/:~")
        response = self.catalog_transport(
            f"{self.base_url}/model/{encoded_model}", self._headers(),
        )
        data = response.get("data") if isinstance(response, dict) else None
        pricing = data.get("pricing") if isinstance(data, dict) else None
        if not isinstance(pricing, dict):
            raise APIError(f"OpenRouter catalog returned no pricing for model {model}")
        try:
            prompt = float(pricing.get("prompt", 0)) * 1_000_000
            completion = float(pricing.get("completion", 0)) * 1_000_000
            reasoning = float(pricing.get("internal_reasoning", 0)) * 1_000_000
            completion = max(completion, reasoning)
            cached = float(pricing.get("input_cache_read", pricing.get("prompt", 0))) * 1_000_000
            cache_write = float(
                pricing.get("input_cache_write", pricing.get("prompt", 0))
            ) * 1_000_000
            request_price = float(pricing.get("request", 0))
        except (TypeError, ValueError) as exc:
            raise APIError(f"OpenRouter catalog returned invalid pricing for model {model}") from exc
        price = Price(prompt, cached, completion, cache_write, request_price)
        self.discovered_prices[model] = price
        return price

    def _call(self, endpoint: str, payload: dict[str, Any], purpose: str,
              estimated_usage: Usage, max_retries: int = 5) -> APIResult:
        model = str(payload["model"])
        request_hash = sha256_json({
            "provider": self.config.provider,
            "base_url": self.base_url,
            "endpoint": endpoint,
            "payload": payload,
            "purpose": purpose,
        })
        cached = self.ledger.get_complete(request_hash)
        if cached:
            raw = json.loads(cached["response_json"])
            return self._result_from_row(cached, raw)
        price = self._price_for_model(model)
        estimate = calculate_cost(estimated_usage, price)
        self.ledger.reserve(
            request_hash, purpose, endpoint, model, estimate, self.config.provider, price,
        )
        started = time.monotonic()
        retries = 0
        headers = self._headers(request_hash)
        while True:
            try:
                response = self.transport(f"{self.base_url}{endpoint}", payload, headers)
                usage = parse_usage(response)
                cost = reported_cost(response)
                if cost is None:
                    cost = calculate_cost(usage, price)
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
                    raise APIError(
                        f"{self.config.provider} request failed after {retries} retries: {exc}"
                    ) from exc
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
        if self.config.api_protocol == "responses":
            endpoint = "/responses"
            output_limit_key = "max_output_tokens"
            payload = {
                "model": model,
                "instructions": instructions,
                "input": input_text,
                "temperature": 0,
                output_limit_key: max_output,
                "store": False,
                "reasoning": {
                    "effort": self.config.reasoning_effort, "context": "current_turn",
                },
                "text": {"format": {"type": "json_schema", "name": schema_name,
                         "strict": True, "schema": schema}},
            }
        elif self.config.api_protocol == "chat_completions":
            endpoint = "/chat/completions"
            output_limit_key = "max_tokens"
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": input_text},
                ],
                "temperature": 0,
                output_limit_key: max_output,
                "reasoning": {"effort": self.config.reasoning_effort},
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name, "strict": True, "schema": schema,
                    },
                },
            }
        else:
            raise APIError(f"unsupported API protocol: {self.config.api_protocol}")
        estimate = Usage(input_tokens=estimate_tokens(instructions + input_text) + 100,
                         output_tokens=max_output)
        output_limit = max_output
        output_retry = 0
        known_adjustments = self.response_compatibility.setdefault(model, set())
        adjustments = [name for name in (
            "no-reasoning-context", "no-reasoning", "no-temperature", "no-json-schema",
        )
                       if name in known_adjustments]
        if "no-reasoning-context" in known_adjustments:
            payload = {**payload, "reasoning": {**payload["reasoning"]}}
            payload["reasoning"].pop("context", None)
        if "no-reasoning" in known_adjustments:
            payload = {key: value for key, value in payload.items() if key != "reasoning"}
        if "no-temperature" in known_adjustments:
            payload = {key: value for key, value in payload.items() if key != "temperature"}
        if "no-json-schema" in known_adjustments and "response_format" in payload:
            payload = with_prompted_json_schema(payload, schema)
        while True:
            adjusted_purpose = purpose
            if adjustments:
                adjusted_purpose += ":compat:" + "+".join(adjustments)
            if output_retry:
                adjusted_purpose += f":output-retry-{output_limit}"
            try:
                result = self._call(endpoint, payload, adjusted_purpose, estimate)
                if self.config.api_protocol == "responses":
                    incomplete_reason = (
                        (result.raw.get("incomplete_details") or {}).get("reason")
                        if result.raw.get("status") == "incomplete" else None
                    )
                else:
                    finish_reason = ((result.raw.get("choices") or [{}])[0]).get(
                        "finish_reason"
                    )
                    incomplete_reason = "max_output_tokens" if finish_reason == "length" else None
                valid_json = True
                try:
                    parsed_output = parse_json_text(result.output_text)
                    valid_json = conforms_to_schema(parsed_output, schema)
                except json.JSONDecodeError:
                    valid_json = False
                if incomplete_reason == "max_output_tokens" or not valid_json:
                    if output_limit >= 4_000:
                        detail = incomplete_reason or "JSON did not conform to the requested schema"
                        raise APIError(
                            f"structured response remained incomplete at {output_limit} "
                            f"output tokens: {detail}"
                        )
                    output_limit = min(4_000, max(800, output_limit * 2))
                    output_retry += 1
                    payload = {**payload, output_limit_key: output_limit}
                    estimate = Usage(
                        input_tokens=estimate.input_tokens, output_tokens=output_limit,
                    )
                    continue
                if incomplete_reason:
                    raise APIError(f"provider response incomplete: {incomplete_reason}")
                return result
            except APIHTTPError as exc:
                rejected = f"{exc.parameter or ''} {exc.message}".casefold()
                if (isinstance(payload.get("reasoning"), dict) and
                        "context" in payload.get("reasoning", {}) and
                        "reasoning" in rejected and "context" in rejected and
                        unsupported_parameter_message(rejected)):
                    payload = {**payload, "reasoning": {**payload["reasoning"]}}
                    payload["reasoning"].pop("context", None)
                    adjustments.append("no-reasoning-context")
                    known_adjustments.add("no-reasoning-context")
                    continue
                if ("reasoning" in payload and "reasoning" in rejected and
                        unsupported_parameter_message(rejected)):
                    payload = {key: value for key, value in payload.items() if key != "reasoning"}
                    adjustments.append("no-reasoning")
                    known_adjustments.add("no-reasoning")
                    continue
                if ("temperature" in payload and "temperature" in rejected and
                        unsupported_parameter_message(rejected)):
                    payload = {key: value for key, value in payload.items() if key != "temperature"}
                    adjustments.append("no-temperature")
                    known_adjustments.add("no-temperature")
                    continue
                if ("response_format" in payload and
                        any(name in rejected for name in ("response_format", "json_schema", "structured")) and
                        unsupported_parameter_message(rejected)):
                    payload = with_prompted_json_schema(payload, schema)
                    adjustments.append("no-json-schema")
                    known_adjustments.add("no-json-schema")
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


# Backward-compatible import for downstream users of the original test bed.
OpenAIClient = LLMClient


def parse_usage(response: dict[str, Any]) -> Usage:
    raw = response.get("usage") or {}
    inp = raw.get("input_tokens_details") or raw.get("prompt_tokens_details") or {}
    out = raw.get("output_tokens_details") or raw.get("completion_tokens_details") or {}
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
    choices = response.get("choices") or []
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(part.get("text", "")) for part in content
                if isinstance(part, dict) and part.get("type") in ("text", "output_text")
            )
    pieces: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in ("output_text", "text") and isinstance(content.get("text"), str):
                pieces.append(content["text"])
    return "".join(pieces)


def parse_json_text(value: str) -> Any:
    text_value = value.strip()
    if text_value.startswith("```") and text_value.endswith("```"):
        lines = text_value.splitlines()
        if len(lines) >= 3:
            text_value = "\n".join(lines[1:-1]).strip()
            if text_value.casefold().startswith("json\n"):
                text_value = text_value[5:].lstrip()
    return json.loads(text_value)


def with_prompted_json_schema(payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    fallback = {key: value for key, value in payload.items() if key != "response_format"}
    messages = [dict(message) for message in fallback.get("messages", [])]
    schema_prompt = (
        "\nReturn only one JSON object matching this JSON Schema exactly. Do not use Markdown "
        f"fences or add commentary. Schema: {canonical_json(schema)}"
    )
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = str(messages[0].get("content", "")) + schema_prompt
    else:
        messages.insert(0, {"role": "system", "content": schema_prompt.lstrip()})
    fallback["messages"] = messages
    return fallback


def reported_cost(response: dict[str, Any]) -> float | None:
    usage = response.get("usage")
    if not isinstance(usage, dict) or usage.get("cost") is None:
        return None
    try:
        cost = float(usage["cost"])
    except (TypeError, ValueError):
        return None
    return cost if cost >= 0 else None


def conforms_to_schema(value: Any, schema: dict[str, Any]) -> bool:
    """Validate the JSON-Schema subset used by this benchmark without extra dependencies."""
    expected = schema.get("type")
    expected_types = expected if isinstance(expected, list) else [expected] if expected else []
    if expected_types and not any(_matches_json_type(value, name) for name in expected_types):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        if any(name not in value for name in schema.get("required", [])):
            return False
        if schema.get("additionalProperties") is False and any(
                name not in properties for name in value):
            return False
        return all(
            name not in value or conforms_to_schema(value[name], child)
            for name, child in properties.items()
        )
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        return all(conforms_to_schema(item, schema["items"]) for item in value)
    return True


def _matches_json_type(value: Any, expected: Any) -> bool:
    return {
        "null": value is None,
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
    }.get(str(expected), True)


def unsupported_parameter_message(value: str) -> bool:
    return any(marker in value for marker in (
        "unknown parameter", "unsupported parameter", "not supported", "unrecognized",
        "no endpoints found that support", "extra inputs are not permitted", "invalid parameter",
    ))
