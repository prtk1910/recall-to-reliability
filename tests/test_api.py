from __future__ import annotations

import io
import tempfile
import unittest
import urllib.error
from email.message import Message
from pathlib import Path
from unittest import mock

from reliabmem.api import (
    APIHTTPError, APILedger, BudgetExceeded, OpenAIClient, Usage,
    calculate_cost, parse_usage,
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


if __name__ == "__main__":
    unittest.main()
