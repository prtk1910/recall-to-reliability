from __future__ import annotations

import io
import tempfile
import unittest
import urllib.error
from dataclasses import replace
from email.message import Message
from pathlib import Path
from unittest import mock

from reliabmem.api import (
    APIHTTPError, APILedger, BudgetExceeded, LLMClient, OpenAIClient, Usage,
    calculate_cost, conforms_to_schema, parse_usage, reported_cost,
)
from reliabmem.config import ExperimentConfig, Price


def response() -> dict:
    return {
        "id": "resp_fake", "model": "gpt-5.6-luna", "output_text": '{"ok":true}',
        "usage": {"input_tokens": 100, "output_tokens": 20,
                  "input_tokens_details": {"cached_tokens": 30, "cache_write_tokens": 10},
                  "output_tokens_details": {"reasoning_tokens": 7}},
    }


class APITests(unittest.TestCase):
    def test_cost_formula_and_usage_details(self) -> None:
        usage = Usage(100, 30, 10, 20, 7)
        price = Price(1.0, 0.1, 2.0, 1.5)
        expected = (60 * 1 + 30 * .1 + 10 * 1.5 + 20 * 2) / 1_000_000
        self.assertAlmostEqual(calculate_cost(usage, price), expected)
        self.assertEqual(parse_usage(response()), usage)

    def test_retry_cache_and_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = APILedger(Path(directory) / "ledger.sqlite", 50)
            self.addCleanup(ledger.db.close)
            calls = []
            def transport(url, payload, headers):
                calls.append((url, payload, headers))
                if len(calls) == 1:
                    raise urllib.error.URLError("rate limited")
                return response()
            client = OpenAIClient(ExperimentConfig(), ledger, api_key="not-logged",
                                  transport=transport, sleep=lambda _: None)
            schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
                      "required": ["ok"], "additionalProperties": False}
            first = client.respond_json(model="gpt-5.6-luna", instructions="x", input_text="y",
                                        schema_name="x", schema=schema, purpose="unit")
            second = client.respond_json(model="gpt-5.6-luna", instructions="x", input_text="y",
                                         schema_name="x", schema=schema, purpose="unit")
            self.assertEqual(len(calls), 2)
            self.assertEqual(first.request_hash, second.request_hash)
            self.assertEqual(first.retries, 1)
            db_text = Path(directory, "ledger.sqlite").read_bytes()
            self.assertNotIn(b"not-logged", db_text)

    def test_reservation_enforces_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = APILedger(Path(directory) / "ledger.sqlite", 0.000001)
            self.addCleanup(ledger.db.close)
            with self.assertRaises(BudgetExceeded):
                ledger.reserve("x", "test", "/responses", "gpt-5.6-luna", 0.01)

    def test_http_error_preserves_sanitized_openai_details(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = APILedger(Path(directory) / "ledger.sqlite", 50)
            self.addCleanup(ledger.db.close)
            client = OpenAIClient(ExperimentConfig(), ledger, api_key="secret-not-in-error")
            headers = Message()
            headers["x-request-id"] = "req_test_123"
            body = io.BytesIO(
                b'{"error":{"message":"Unknown parameter: reasoning.context",'
                b'"param":"reasoning.context","code":"unknown_parameter"}}'
            )
            error = urllib.error.HTTPError(
                "https://api.openai.com/v1/responses", 400, "Bad Request", headers, body,
            )
            with mock.patch("urllib.request.urlopen", side_effect=error):
                with self.assertRaises(APIHTTPError) as caught:
                    client._http_transport(
                        "https://api.openai.com/v1/responses", {"model": "x"}, {},
                    )
            rendered = str(caught.exception)
            self.assertIn("reasoning.context", rendered)
            self.assertIn("req_test_123", rendered)
            self.assertNotIn("secret-not-in-error", rendered)

    def test_targeted_response_parameter_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = APILedger(Path(directory) / "ledger.sqlite", 50)
            self.addCleanup(ledger.db.close)
            payloads = []
            def transport(url, payload, headers):
                payloads.append(payload)
                if "context" in payload.get("reasoning", {}):
                    raise APIHTTPError(
                        400, "Unknown parameter: reasoning.context",
                        parameter="reasoning.context", code="unknown_parameter",
                    )
                if "temperature" in payload:
                    raise APIHTTPError(
                        400, "Unsupported parameter: temperature is not supported with this model",
                        parameter="temperature", code="unsupported_parameter",
                    )
                return response()
            client = OpenAIClient(
                ExperimentConfig(), ledger, api_key="fake", transport=transport,
                sleep=lambda _: None,
            )
            schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
                      "required": ["ok"], "additionalProperties": False}
            first = client.respond_json(
                model="gpt-5.6-luna", instructions="x", input_text="y",
                schema_name="x", schema=schema, purpose="compat",
            )
            second = client.respond_json(
                model="gpt-5.6-luna", instructions="x", input_text="y",
                schema_name="x", schema=schema, purpose="compat",
            )
            self.assertEqual(first.request_hash, second.request_hash)
            self.assertEqual(len(payloads), 3)
            self.assertIn("context", payloads[0]["reasoning"])
            self.assertNotIn("context", payloads[1]["reasoning"])
            self.assertNotIn("temperature", payloads[2])
            purposes = [row[0] for row in ledger.db.execute(
                "SELECT purpose FROM api_calls WHERE status='complete'"
            )]
            self.assertEqual(
                purposes, ["compat:compat:no-reasoning-context+no-temperature"],
            )

    def test_incomplete_structured_response_retries_with_larger_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = APILedger(Path(directory) / "ledger.sqlite", 50)
            self.addCleanup(ledger.db.close)
            limits = []
            def transport(url, payload, headers):
                limits.append(payload["max_output_tokens"])
                if payload["max_output_tokens"] < 800:
                    return {
                        "id": "resp_truncated", "model": payload["model"],
                        "status": "incomplete",
                        "incomplete_details": {"reason": "max_output_tokens"},
                        "output_text": '{"ok":',
                        "usage": {"input_tokens": 10, "output_tokens": payload["max_output_tokens"]},
                    }
                return response()
            client = OpenAIClient(
                ExperimentConfig(), ledger, api_key="fake", transport=transport,
                sleep=lambda _: None,
            )
            schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
                      "required": ["ok"], "additionalProperties": False}
            result = client.respond_json(
                model="gpt-5.6-luna", instructions="x", input_text="y",
                schema_name="x", schema=schema, purpose="truncation",
                max_output_tokens=300,
            )
            self.assertEqual(result.json(), {"ok": True})
            self.assertEqual(limits, [300, 800])
            purposes = [row[0] for row in ledger.db.execute(
                "SELECT purpose FROM api_calls WHERE status='complete' ORDER BY created_at, purpose"
            )]
            self.assertEqual(
                purposes, ["truncation", "truncation:output-retry-800"],
            )

    def test_openrouter_chat_catalog_cost_and_schema_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = APILedger(Path(directory) / "ledger.sqlite", 50)
            self.addCleanup(ledger.db.close)
            config = replace(
                ExperimentConfig(), provider="openrouter",
                api_base_url="https://openrouter.ai/api/v1",
                api_protocol="chat_completions", api_key_env="OPENROUTER_API_KEY",
                primary_model="example/model", embedding_model="openai/text-embedding-3-small",
                app_url="https://example.test/research", app_title="Reliability Test",
                prices={},
            )
            catalog_calls = []
            payloads = []

            def catalog(url, headers):
                catalog_calls.append((url, headers))
                return {"data": {"id": "example/model", "pricing": {
                    "prompt": "0.000001", "completion": "0.000004",
                    "input_cache_read": "0.0000005", "request": "0",
                }}}

            def transport(url, payload, headers):
                payloads.append((url, payload, headers))
                if "response_format" in payload:
                    raise APIHTTPError(
                        400, "No endpoints found that support response_format",
                        provider="openrouter",
                    )
                return {
                    "id": "gen_test", "model": "example/model:provider",
                    "choices": [{"message": {"content": "```json\n{\"ok\":true}\n```"},
                                 "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 120, "completion_tokens": 8, "cost": 0.0123,
                        "prompt_tokens_details": {
                            "cached_tokens": 20, "cache_write_tokens": 5,
                        },
                        "completion_tokens_details": {"reasoning_tokens": 3},
                    },
                }

            client = LLMClient(
                config, ledger, api_key="openrouter-secret", transport=transport,
                catalog_transport=catalog, sleep=lambda _: None,
            )
            schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
                      "required": ["ok"], "additionalProperties": False}
            first = client.respond_json(
                model=config.primary_model, instructions="Return the result.", input_text="Test",
                schema_name="result", schema=schema, purpose="openrouter-unit",
            )
            second = client.respond_json(
                model=config.primary_model, instructions="Return the result.", input_text="Test",
                schema_name="result", schema=schema, purpose="openrouter-unit",
            )

            self.assertEqual(first.json(), {"ok": True})
            self.assertEqual(first.cost_usd, 0.0123)
            self.assertEqual(first.usage, Usage(120, 20, 5, 8, 3))
            self.assertEqual(first.request_hash, second.request_hash)
            self.assertEqual(len(catalog_calls), 1)
            self.assertEqual(len(payloads), 2)
            self.assertTrue(payloads[0][0].endswith("/chat/completions"))
            self.assertEqual(payloads[0][2]["HTTP-Referer"], "https://example.test/research")
            self.assertEqual(payloads[0][2]["X-OpenRouter-Title"], "Reliability Test")
            self.assertNotIn("response_format", payloads[1][1])
            self.assertIn("JSON Schema", payloads[1][1]["messages"][0]["content"])
            row = ledger.db.execute(
                "SELECT provider,actual_cost_usd,input_price_per_million,"
                "output_price_per_million FROM api_calls WHERE status='complete'"
            ).fetchone()
            self.assertEqual(row["provider"], "openrouter")
            self.assertAlmostEqual(row["actual_cost_usd"], 0.0123)
            self.assertEqual(row["input_price_per_million"], 1.0)
            self.assertEqual(row["output_price_per_million"], 4.0)
            self.assertNotIn(b"openrouter-secret", Path(directory, "ledger.sqlite").read_bytes())

    def test_openrouter_environment_configuration(self) -> None:
        environment = {
            "RELIABMEM_PROVIDER": "openrouter",
            "RELIABMEM_MODEL": "google/gemini-test",
            "RELIABMEM_EMBEDDING_MODEL": "openai/text-embedding-3-small",
            "RELIABMEM_INPUT_PRICE_PER_MILLION": "2.5",
            "RELIABMEM_OUTPUT_PRICE_PER_MILLION": "7.5",
            "OPENROUTER_SITE_URL": "https://example.test",
            "OPENROUTER_APP_TITLE": "Test-Bed",
        }
        with mock.patch.dict("os.environ", environment, clear=True):
            config = ExperimentConfig.from_env()
        self.assertEqual(config.provider, "openrouter")
        self.assertEqual(config.api_protocol, "chat_completions")
        self.assertEqual(config.api_key_env, "OPENROUTER_API_KEY")
        self.assertEqual(config.primary_model, "google/gemini-test")
        self.assertEqual(config.prices[config.primary_model], Price(2.5, 2.5, 7.5, 2.5, 0))
        self.assertEqual(config.app_url, "https://example.test")
        self.assertEqual(config.app_title, "Test-Bed")

    def test_openrouter_embedding_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = APILedger(Path(directory) / "ledger.sqlite", 50)
            self.addCleanup(ledger.db.close)
            embedding_model = "openai/text-embedding-3-small"
            config = replace(
                ExperimentConfig(), provider="openrouter",
                api_base_url="https://openrouter.ai/api/v1",
                api_protocol="chat_completions", api_key_env="OPENROUTER_API_KEY",
                embedding_model=embedding_model,
                prices={embedding_model: Price(0.02, 0.02, 0)},
            )
            calls = []

            def transport(url, payload, headers):
                calls.append((url, payload))
                return {
                    "id": "emb_test", "model": embedding_model,
                    "data": [{"index": 0, "embedding": [0.1, 0.2]}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 0, "cost": 0.0001},
                }

            client = LLMClient(config, ledger, api_key="fake", transport=transport)
            vectors, result = client.embed(["hello"], "embedding-unit")
            self.assertEqual(vectors, [[0.1, 0.2]])
            self.assertTrue(calls[0][0].endswith("/embeddings"))
            self.assertEqual(calls[0][1]["model"], embedding_model)
            self.assertEqual(result.cost_usd, 0.0001)

    def test_openrouter_usage_cost_helpers(self) -> None:
        shaped = {"usage": {
            "prompt_tokens": 9, "completion_tokens": 4, "cost": "0.0012",
            "prompt_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 1},
            "completion_tokens_details": {"reasoning_tokens": 3},
        }}
        self.assertEqual(parse_usage(shaped), Usage(9, 2, 1, 4, 3))
        self.assertEqual(reported_cost(shaped), 0.0012)

    def test_local_schema_validation_for_prompt_fallback(self) -> None:
        schema = {
            "type": "object", "additionalProperties": False,
            "properties": {
                "name": {"type": "string"},
                "values": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["name", "values"],
        }
        self.assertTrue(conforms_to_schema({"name": "x", "values": [1, 2]}, schema))
        self.assertFalse(conforms_to_schema({"name": "x", "values": [True]}, schema))
        self.assertFalse(conforms_to_schema({"name": "x", "values": [], "extra": 1}, schema))


if __name__ == "__main__":
    unittest.main()
