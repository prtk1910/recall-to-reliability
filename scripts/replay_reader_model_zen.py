from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
from dataclasses import replace
from pathlib import Path

from reliabmem.api import (
    APIError,
    APIHTTPError,
    APILedger,
    LLMClient,
)
from reliabmem.benchmark import Task
from reliabmem.config import ExperimentConfig, Price, TASK_TYPES
from reliabmem.harness import (
    ANSWER_INSTRUCTIONS,
    ANSWER_SCHEMA,
    evaluate,
)


# Used only to select the 12-row pilot.
ARCHITECTURES = (
    "query_only",
    "full_context",
    "recency",
    "vector_rag",
    "hierarchical_summary",
    "structured_temporal",
)


# -------------------------------------------------------------------
# OpenCode Zen configuration
# -------------------------------------------------------------------

def make_zen_config() -> ExperimentConfig:
    """
    Build a config for OpenCode Zen Ox Alpha Free.

    We construct ExperimentConfig directly because the repository's
    ExperimentConfig.from_env() currently only accepts:
      - openai
      - openrouter

    This lets the experimental ledger correctly record the provider
    as "opencode_zen" rather than pretending this is OpenRouter.
    """
    base = ExperimentConfig()

    prices = dict(base.prices)

    # Ox Alpha Free promotional Zen route.
    prices["x-preview-f-free"] = Price(
        input_per_million=0.0,
        cached_input_per_million=0.0,
        output_per_million=0.0,
        cache_write_per_million=0.0,
        request_usd=0.0,
    )

    return replace(
        base,
        provider="opencode_zen",
        api_base_url="https://opencode.ai/zen/v1",
        api_protocol="chat_completions",
        api_key_env="OPENCODE_ZEN_API_KEY",
        primary_model="x-preview-f-free",
        confirmation_model="",
        reasoning_effort="low",
        pricing_snapshot_date="2026-08-23-opencode-zen-free",
        pricing_sources=(
            "https://opencode.ai/docs/zen/",
        ),
        prices=prices,
    )


# -------------------------------------------------------------------
# CURL transport
# -------------------------------------------------------------------

def make_zen_curl_transport(api_key: str):
    """
    Use curl instead of urllib for Zen requests.

    Direct curl calls to the Zen endpoint have already been verified
    to work with this account.

    The repository normally supplies an Idempotency-Key header.
    We intentionally do not forward that header here.

    LLMClient still handles:
      - request hashing
      - retries/backoff
      - API ledger
      - token accounting
      - cost accounting
      - structured-output validation
    """

    def transport(
        url: str,
        payload: dict,
        headers: dict[str, str],
    ) -> dict:
        del headers  # intentionally do not forward repo-generated headers

        process = subprocess.run(
            [
                "curl",
                "-sS",
                "--connect-timeout",
                "30",
                "--max-time",
                "240",
                "-X",
                "POST",
                url,
                "-H",
                f"Authorization: Bearer {api_key}",
                "-H",
                "Content-Type: application/json",
                "-H",
                "Accept: application/json",
                "--data-binary",
                "@-",
                "-w",
                "\n%{http_code}",
            ],
            input=json.dumps(
                payload,
                separators=(",", ":"),
            ),
            text=True,
            capture_output=True,
        )

        # curl-level error: timeout, DNS failure, connection reset, etc.
        if process.returncode != 0:
            message = process.stderr.strip()

            raise APIError(
                f"Zen curl transport failed "
                f"(exit={process.returncode}): {message}"
            )

        body, separator, status_text = process.stdout.rpartition("\n")

        if not separator:
            raise APIError(
                "Zen curl transport did not return an HTTP status code"
            )

        try:
            status = int(status_text.strip())
        except ValueError as exc:
            raise APIError(
                "Could not parse Zen HTTP status. "
                f"Response tail: {process.stdout[-500:]}"
            ) from exc

        # Parse response body.
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise APIError(
                f"Zen returned non-JSON data "
                f"(HTTP {status}): {body[:1000]}"
            ) from exc

        # Convert HTTP errors into the same exception type expected
        # by the repository's retry/compatibility logic.
        if status < 200 or status >= 300:
            error = data.get("error")

            if isinstance(error, dict):
                message = (
                    error.get("message")
                    or error.get("type")
                    or str(error)
                )

                parameter = (
                    error.get("param")
                    or error.get("parameter")
                )

                code = error.get("code")

                request_id = (
                    error.get("request_id")
                    or data.get("request_id")
                )

            else:
                message = (
                    str(error)
                    if error is not None
                    else str(data)
                )

                parameter = None
                code = None
                request_id = data.get("request_id")

            raise APIHTTPError(
                status,
                str(message),
                parameter=(
                    str(parameter)
                    if parameter is not None
                    else None
                ),
                code=(
                    str(code)
                    if code is not None
                    else None
                ),
                request_id=(
                    str(request_id)
                    if request_id is not None
                    else None
                ),
                provider="opencode_zen",
            )

        return data

    return transport


# -------------------------------------------------------------------
# Read original experiment rows
# -------------------------------------------------------------------

def load_rows(
    db: sqlite3.Connection,
    mode: str,
) -> list[sqlite3.Row]:
    rows = db.execute(
        """
        SELECT
            r.result_id AS original_result_id,
            r.world_id,
            r.task_id,
            r.architecture,
            r.success AS original_success,
            r.assembled_prompt,
            r.prompt_hash,

            t.task_type,
            t.query,
            t.answer,
            t.evidence_turn_ids,
            t.hop_count,
            t.unanswerable,
            t.forgotten,
            t.valid_tool_call

        FROM results r

        JOIN tasks t
          ON t.task_id = r.task_id

        WHERE r.stage = 'screening'

        ORDER BY
            r.world_id,
            r.task_id,
            r.architecture
        """
    ).fetchall()

    if mode == "full":
        return rows

    # Pilot:
    # choose exactly one case from each of the 12 task types,
    # rotating across the six memory architectures.
    selected = []

    for index, task_type in enumerate(TASK_TYPES):
        architecture = ARCHITECTURES[
            index % len(ARCHITECTURES)
        ]

        match = next(
            row
            for row in rows
            if row["task_type"] == task_type
            and row["architecture"] == architecture
        )

        selected.append(match)

    return selected


# -------------------------------------------------------------------
# Reconstruct Task object for existing benchmark scorer
# -------------------------------------------------------------------

def make_task(row: sqlite3.Row) -> Task:
    return Task(
        task_id=row["task_id"],
        task_type=row["task_type"],
        query=row["query"],
        answer=row["answer"],
        evidence_turn_ids=tuple(
            json.loads(row["evidence_turn_ids"])
        ),
        hop_count=int(row["hop_count"]),
        unanswerable=bool(row["unanswerable"]),
        forgotten=bool(row["forgotten"]),
        valid_tool_call=(
            json.loads(row["valid_tool_call"])
            if row["valid_tool_call"]
            else None
        ),
    )


# -------------------------------------------------------------------
# Output database
# -------------------------------------------------------------------

def create_output_tables(
    output: sqlite3.Connection,
) -> None:
    output.execute(
        """
        CREATE TABLE IF NOT EXISTS replay_results (
            original_result_id TEXT PRIMARY KEY,

            world_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            task_type TEXT NOT NULL,
            architecture TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,

            original_success INTEGER NOT NULL,
            ox_success INTEGER NOT NULL,

            predicted_status TEXT NOT NULL,
            predicted_answer TEXT NOT NULL,
            predicted_tool_call TEXT,
            local_tool_result TEXT,

            provider TEXT NOT NULL,

            requested_model TEXT NOT NULL,
            returned_model TEXT NOT NULL,

            response_id TEXT NOT NULL,
            request_hash TEXT NOT NULL,

            input_tokens INTEGER NOT NULL,
            cached_tokens INTEGER NOT NULL,
            cache_write_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            reasoning_tokens INTEGER NOT NULL,

            cost_usd REAL NOT NULL,
            latency_ms INTEGER NOT NULL,
            retries INTEGER NOT NULL,

            output_text TEXT NOT NULL,

            created_at TEXT NOT NULL
                DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # API failures / invalid structured outputs are kept separate.
    # They must NOT automatically be counted as incorrect answers.
    output.execute(
        """
        CREATE TABLE IF NOT EXISTS replay_errors (
            original_result_id TEXT PRIMARY KEY,

            world_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            task_type TEXT NOT NULL,
            architecture TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,

            provider TEXT NOT NULL,
            model TEXT NOT NULL,

            error_type TEXT NOT NULL,
            error_message TEXT NOT NULL,

            attempt_count INTEGER NOT NULL
                DEFAULT 1,

            created_at TEXT NOT NULL
                DEFAULT CURRENT_TIMESTAMP,

            updated_at TEXT NOT NULL
                DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    output.commit()


def record_error(
    output: sqlite3.Connection,
    row: sqlite3.Row,
    provider: str,
    model: str,
    exc: Exception,
) -> None:
    output.execute(
        """
        INSERT INTO replay_errors (
            original_result_id,

            world_id,
            task_id,
            task_type,
            architecture,
            prompt_hash,

            provider,
            model,

            error_type,
            error_message,

            attempt_count
        )

        VALUES (
            ?, ?, ?, ?, ?, ?,
            ?, ?,
            ?, ?,
            1
        )

        ON CONFLICT(original_result_id)
        DO UPDATE SET
            error_type = excluded.error_type,
            error_message = excluded.error_message,
            attempt_count =
                replay_errors.attempt_count + 1,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            row["original_result_id"],

            row["world_id"],
            row["task_id"],
            row["task_type"],
            row["architecture"],
            row["prompt_hash"],

            provider,
            model,

            type(exc).__name__,
            str(exc),
        ),
    )

    output.commit()


# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------

def print_summary(
    output: sqlite3.Connection,
) -> None:
    successful = output.execute(
        """
        SELECT COUNT(*)
        FROM replay_results
        """
    ).fetchone()[0]

    unresolved_errors = output.execute(
        """
        SELECT COUNT(*)

        FROM replay_errors e

        WHERE NOT EXISTS (
            SELECT 1

            FROM replay_results r

            WHERE r.original_result_id =
                  e.original_result_id
        )
        """
    ).fetchone()[0]

    print()
    print("Replay pass complete.")
    print(f"Successful replay rows: {successful}")
    print(
        f"Rows still requiring retry: "
        f"{unresolved_errors}"
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay stored GPT-5.6 Luna screening prompts "
            "through OpenCode Zen Ox Alpha Free."
        )
    )

    parser.add_argument(
        "--source-db",
        required=True,
        help=(
            "Path to original GPT-5.6 Luna "
            "results.sqlite."
        ),
    )

    parser.add_argument(
        "--output-db",
        default=(
            "artifacts/ox_zen_replay/"
            "results.sqlite"
        ),
        help=(
            "SQLite database used for "
            "OpenCode Zen replay results."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=("pilot", "full"),
        default="pilot",
        help=(
            "'pilot' selects 12 cases; "
            "'full' selects all 3240 screening rows."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional maximum number of selected rows."
        ),
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------

    config = make_zen_config()

    zen_key = os.environ.get(
        "OPENCODE_ZEN_API_KEY",
        "",
    ).strip()

    if not zen_key:
        raise RuntimeError(
            "OPENCODE_ZEN_API_KEY is not loaded.\n"
            "Run:\n"
            "  set -a\n"
            "  source .env.zen\n"
            "  set +a"
        )

    # ---------------------------------------------------------------
    # Original Luna database — read only
    # ---------------------------------------------------------------

    source_path = Path(
        args.source_db
    ).resolve()

    source = sqlite3.connect(
        f"file:{source_path}?mode=ro",
        uri=True,
    )

    source.row_factory = sqlite3.Row

    # ---------------------------------------------------------------
    # New Zen output database
    # ---------------------------------------------------------------

    output_path = Path(
        args.output_db
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = sqlite3.connect(
        output_path
    )

    output.row_factory = sqlite3.Row

    create_output_tables(
        output
    )

    # ---------------------------------------------------------------
    # Zen API ledger
    # ---------------------------------------------------------------

    ledger_path = Path(
        "artifacts/"
        "ox_zen_replay/"
        "api_calls.sqlite"
    )

    ledger = APILedger(
        ledger_path,
        config.spend_cap_usd,
    )

    # ---------------------------------------------------------------
    # LLM client using CURL transport
    # ---------------------------------------------------------------

    client = LLMClient(
        config,
        ledger,
        api_key=zen_key,
        transport=make_zen_curl_transport(
            zen_key
        ),
    )

    # Ox Alpha does not reliably obey strict JSON Schema output
    # enforcement. The repository already provides a fallback that
    # embeds the required JSON schema in the system prompt and then
    # validates the returned JSON locally.
    client.response_compatibility.setdefault(
        config.primary_model,
        set(),
    ).add(
        "no-json-schema"
    )

    # ---------------------------------------------------------------
    # Select replay rows
    # ---------------------------------------------------------------

    rows = load_rows(
        source,
        args.mode,
    )

    if args.limit is not None:
        rows = rows[
            : args.limit
        ]

    print(
        f"Provider: {config.provider}",
        flush=True,
    )

    print(
        f"Model: {config.primary_model}",
        flush=True,
    )

    print(
        f"Endpoint: {config.api_base_url}",
        flush=True,
    )

    print(
        f"Mode: {args.mode}",
        flush=True,
    )

    print(
        f"Rows selected: {len(rows)}",
        flush=True,
    )

    print(
        "",
        flush=True,
    )

    # ---------------------------------------------------------------
    # Replay
    # ---------------------------------------------------------------

    for index, row in enumerate(
        rows,
        start=1,
    ):
        # Never call a row again after a successful replay.
        existing = output.execute(
            """
            SELECT 1

            FROM replay_results

            WHERE original_result_id = ?
            """,
            (
                row[
                    "original_result_id"
                ],
            ),
        ).fetchone()

        if existing:
            print(
                f"[{index}/{len(rows)}] "
                "already complete",
                flush=True,
            )

            continue

        task = make_task(
            row
        )

        print(
            f"[{index}/{len(rows)}] "
            f"{row['architecture']} / "
            f"{row['task_type']}",
            flush=True,
        )

        try:
            response = client.respond_json(
                model=config.primary_model,

                instructions=ANSWER_INSTRUCTIONS,

                # IMPORTANT:
                # exact prompt stored from the original
                # GPT-5.6 Luna experiment.
                input_text=row[
                    "assembled_prompt"
                ],

                schema_name=(
                    "memory_answer"
                ),

                schema=ANSWER_SCHEMA,

                purpose=(
                    "fixed-context-zen-replay:"
                    f"{row['original_result_id']}"
                ),
            )

            # Use the benchmark's ORIGINAL scorer.
            evaluation = evaluate(
                task,
                response.json(),
            )

        except APIError as exc:
            print(
                f"    ERROR: {exc}",
                flush=True,
            )

            print(
                "    Saved to replay_errors; "
                "continuing.",
                flush=True,
            )

            record_error(
                output,
                row,
                config.provider,
                config.primary_model,
                exc,
            )

            continue

        except Exception as exc:
            # Do not allow one unexpected case to kill
            # a multi-hour 3240-row experiment.
            print(
                "    UNEXPECTED ERROR: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

            print(
                "    Saved to replay_errors; "
                "continuing.",
                flush=True,
            )

            record_error(
                output,
                row,
                config.provider,
                config.primary_model,
                exc,
            )

            continue

        # -----------------------------------------------------------
        # Save successful replay
        # -----------------------------------------------------------

        output.execute(
            """
            INSERT INTO replay_results (
                original_result_id,

                world_id,
                task_id,
                task_type,
                architecture,
                prompt_hash,

                original_success,
                ox_success,

                predicted_status,
                predicted_answer,
                predicted_tool_call,
                local_tool_result,

                provider,

                requested_model,
                returned_model,

                response_id,
                request_hash,

                input_tokens,
                cached_tokens,
                cache_write_tokens,
                output_tokens,
                reasoning_tokens,

                cost_usd,
                latency_ms,
                retries,

                output_text
            )

            VALUES (
                ?,

                ?, ?, ?, ?, ?,

                ?, ?,

                ?, ?, ?, ?,

                ?,

                ?, ?,

                ?, ?,

                ?, ?, ?, ?, ?,

                ?, ?, ?,

                ?
            )
            """,
            (
                row[
                    "original_result_id"
                ],

                row["world_id"],
                row["task_id"],
                row["task_type"],
                row["architecture"],
                row["prompt_hash"],

                int(
                    row["original_success"]
                ),
                int(
                    evaluation.success
                ),

                evaluation.predicted_status,
                evaluation.predicted_answer,

                (
                    json.dumps(
                        evaluation.predicted_tool_call,
                        sort_keys=True,
                    )
                    if evaluation.predicted_tool_call
                    is not None
                    else None
                ),

                (
                    json.dumps(
                        evaluation.local_tool_result,
                        sort_keys=True,
                    )
                    if evaluation.local_tool_result
                    is not None
                    else None
                ),

                config.provider,

                response.requested_model,
                response.returned_model,

                response.response_id,
                response.request_hash,

                response.usage.input_tokens,
                response.usage.cached_tokens,
                response.usage.cache_write_tokens,
                response.usage.output_tokens,
                response.usage.reasoning_tokens,

                response.cost_usd,
                response.latency_ms,
                response.retries,

                response.output_text,
            ),
        )

        # If this row failed on an earlier attempt but now
        # succeeded, remove its error record.
        output.execute(
            """
            DELETE FROM replay_errors

            WHERE original_result_id = ?
            """,
            (
                row[
                    "original_result_id"
                ],
            ),
        )

        output.commit()

        print(
            "    "
            f"Luna="
            f"{bool(row['original_success'])} "
            f"Ox={evaluation.success} "
            f"| input="
            f"{response.usage.input_tokens} "
            f"cached="
            f"{response.usage.cached_tokens} "
            f"output="
            f"{response.usage.output_tokens} "
            f"reasoning="
            f"{response.usage.reasoning_tokens} "
            f"| {response.latency_ms} ms "
            f"| ${response.cost_usd:.6f}",
            flush=True,
        )

    # ---------------------------------------------------------------
    # Final pass summary
    # ---------------------------------------------------------------

    print_summary(
        output
    )

    source.close()
    output.close()


if __name__ == "__main__":
    main()
