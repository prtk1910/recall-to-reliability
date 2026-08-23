from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import re
import sqlite3
import time

from .api import APIResult, LLMClient
from .benchmark import Task, World
from .config import ExperimentConfig
from .memory import ContextTrace, MemoryBackend, create_backend
from .util import canonical_json, sha256_json


ANSWER_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["answered", "unknown", "forgotten", "tool"]},
        "answer": {"type": "string"},
        "tool_name": {"type": ["string", "null"]},
        "tool_arguments_json": {"type": ["string", "null"]},
    },
    "required": ["status", "answer", "tool_name", "tool_arguments_json"],
}


ANSWER_INSTRUCTIONS = """You answer solely from the supplied memory context.
Treat later explicit updates as superseding earlier values. Honor explicit forgetting: return status
forgotten and never reproduce the forgotten value. If the requested fact is absent, return unknown.
For a permitted local tool task, return status tool, the exact tool name, and a compact JSON object
encoded in tool_arguments_json. Do not invent arguments. Otherwise return status answered and a
minimal answer. Never call or claim to call an external service."""


@dataclass(frozen=True)
class Evaluation:
    success: bool
    predicted_status: str
    predicted_answer: str
    predicted_tool_call: dict[str, Any] | None
    local_tool_result: dict[str, Any] | None


class ResultStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY, stage TEXT NOT NULL, seed INTEGER NOT NULL,
            config_hash TEXT NOT NULL, code_revision TEXT NOT NULL,
            model TEXT NOT NULL, started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            completed_at TEXT
          );
          CREATE TABLE IF NOT EXISTS worlds (
            world_id TEXT PRIMARY KEY, seed INTEGER NOT NULL, temporal_distance INTEGER NOT NULL,
            interference INTEGER NOT NULL, contradictions INTEGER NOT NULL,
            turn_count INTEGER NOT NULL, token_estimate INTEGER NOT NULL, checksum TEXT NOT NULL
          );
          CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY, world_id TEXT NOT NULL, task_type TEXT NOT NULL,
            query TEXT NOT NULL, answer TEXT NOT NULL, evidence_turn_ids TEXT NOT NULL,
            hop_count INTEGER NOT NULL, unanswerable INTEGER NOT NULL, forgotten INTEGER NOT NULL,
            valid_tool_call TEXT
          );
          CREATE TABLE IF NOT EXISTS writes (
            run_id TEXT NOT NULL, world_id TEXT NOT NULL, architecture TEXT NOT NULL,
            turn_id INTEGER NOT NULL, operation TEXT NOT NULL, memory_ids TEXT NOT NULL,
            PRIMARY KEY(run_id,world_id,architecture,turn_id)
          );
          CREATE TABLE IF NOT EXISTS results (
            result_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, stage TEXT NOT NULL,
            world_id TEXT NOT NULL, task_id TEXT NOT NULL, architecture TEXT NOT NULL,
            repeat_index INTEGER NOT NULL DEFAULT 0, success INTEGER NOT NULL,
            predicted_status TEXT NOT NULL, predicted_answer TEXT NOT NULL,
            predicted_tool_call TEXT, local_tool_result TEXT,
            context_trace TEXT NOT NULL, assembled_prompt TEXT NOT NULL,
            prompt_hash TEXT NOT NULL, answer_request_hash TEXT NOT NULL,
            requested_model TEXT NOT NULL, returned_model TEXT NOT NULL,
            response_id TEXT NOT NULL, input_tokens INTEGER NOT NULL,
            cached_tokens INTEGER NOT NULL, cache_write_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL, reasoning_tokens INTEGER NOT NULL,
            answer_cost_usd REAL NOT NULL, latency_ms INTEGER NOT NULL,
            retries INTEGER NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(run_id,task_id,architecture,repeat_index)
          );
          CREATE TABLE IF NOT EXISTS diagnostics (
            result_id TEXT PRIMARY KEY, primary_cause TEXT NOT NULL,
            contributing_flags TEXT NOT NULL, oracle_success INTEGER NOT NULL,
            gold_only_success INTEGER NOT NULL, oracle_request_hash TEXT NOT NULL,
            gold_only_request_hash TEXT NOT NULL, evidence_selected INTEGER NOT NULL
          );
          CREATE TABLE IF NOT EXISTS manifests (
            run_id TEXT NOT NULL, artifact_path TEXT NOT NULL, sha256 TEXT NOT NULL,
            PRIMARY KEY(run_id,artifact_path)
          );
        """)
        self.db.commit()

    def start_run(self, run_id: str, stage: str, config: ExperimentConfig, revision: str,
                  model: str) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO runs(run_id,stage,seed,config_hash,code_revision,model) VALUES(?,?,?,?,?,?)",
                (run_id, stage, config.seed, config.hash(), revision, model),
            )

    def complete_run(self, run_id: str) -> None:
        with self.db:
            self.db.execute("UPDATE runs SET completed_at=CURRENT_TIMESTAMP WHERE run_id=?", (run_id,))

    def register_world(self, world: World, token_estimate: int) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO worlds VALUES(?,?,?,?,?,?,?,?)",
                (world.world_id, world.seed, world.profile.temporal_distance,
                 world.profile.interference, world.profile.contradictions, len(world.turns),
                 token_estimate, sha256_json(world.to_dict())),
            )
            for task in world.tasks:
                self.db.execute(
                    "INSERT OR REPLACE INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (task.task_id, world.world_id, task.task_type, task.query, task.answer,
                     canonical_json(task.evidence_turn_ids), task.hop_count, task.unanswerable,
                     task.forgotten, canonical_json(task.valid_tool_call) if task.valid_tool_call else None),
                )

    def has_result(self, run_id: str, task_id: str, architecture: str, repeat_index: int) -> bool:
        return self.db.execute(
            "SELECT 1 FROM results WHERE run_id=? AND task_id=? AND architecture=? AND repeat_index=?",
            (run_id, task_id, architecture, repeat_index),
        ).fetchone() is not None

    def existing_result(self, run_id: str, task_id: str, architecture: str,
                        repeat_index: int) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT * FROM results WHERE run_id=? AND task_id=? AND architecture=? AND repeat_index=?",
            (run_id, task_id, architecture, repeat_index),
        ).fetchone()

    def insert_write(self, run_id: str, world_id: str, trace: Any) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO writes VALUES(?,?,?,?,?,?)",
                (run_id, world_id, trace.architecture, trace.turn_id, trace.operation,
                 canonical_json(trace.memory_ids)),
            )

    def insert_result(self, *, result_id: str, run_id: str, stage: str, world: World,
                      task: Task, architecture: str, repeat_index: int, evaluation: Evaluation,
                      context: ContextTrace, assembled_prompt: str, response: APIResult) -> bool:
        with self.db:
            cursor = self.db.execute(
                "INSERT OR IGNORE INTO results VALUES(" + ",".join("?" for _ in range(27))
                + ",CURRENT_TIMESTAMP)",
                (result_id, run_id, stage, world.world_id, task.task_id, architecture, repeat_index,
                 evaluation.success, evaluation.predicted_status, evaluation.predicted_answer,
                 canonical_json(evaluation.predicted_tool_call) if evaluation.predicted_tool_call else None,
                 canonical_json(evaluation.local_tool_result) if evaluation.local_tool_result else None,
                 canonical_json(asdict(context)), assembled_prompt, sha256_json(assembled_prompt),
                 response.request_hash, response.requested_model, response.returned_model,
                 response.response_id, response.usage.input_tokens, response.usage.cached_tokens,
                 response.usage.cache_write_tokens, response.usage.output_tokens,
                 response.usage.reasoning_tokens, response.cost_usd, response.latency_ms,
                 response.retries),
            )
            return cursor.rowcount > 0

    def insert_diagnostic(self, result_id: str, cause: str, flags: list[str],
                          oracle: Evaluation, gold_only: Evaluation,
                          oracle_hash: str, gold_hash: str, evidence_selected: bool) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO diagnostics VALUES(?,?,?,?,?,?,?,?)",
                (result_id, cause, canonical_json(flags), oracle.success, gold_only.success,
                 oracle_hash, gold_hash, evidence_selected),
            )


class ExperimentHarness:
    def __init__(self, config: ExperimentConfig, client: LLMClient, store: ResultStore,
                 artifact_dir: Path):
        self.config = config
        self.client = client
        self.store = store
        self.artifact_dir = artifact_dir
        self.raw_path = artifact_dir / "raw" / "results.jsonl"
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)

    def run_world(self, run_id: str, stage: str, world: World, architecture: str,
                  model: str | None = None, repeat_indices: tuple[int, ...] = (0,),
                  task_ids: set[str] | None = None) -> None:
        backend = create_backend(architecture, self.config, self.client, self.artifact_dir / "state")
        backend.reset(world.world_id)
        for turn in world.turns:
            trace = backend.observe(turn)
            self.store.insert_write(run_id, world.world_id, trace)
        for task in world.tasks:
            if task_ids is not None and task.task_id not in task_ids:
                continue
            context = backend.build_context(task)
            for repeat_index in repeat_indices:
                existing = self.store.existing_result(run_id, task.task_id, architecture, repeat_index)
                if existing:
                    if (not existing["success"] and repeat_index == 0 and
                            self.store.db.execute("SELECT 1 FROM diagnostics WHERE result_id=?",
                                                  (existing["result_id"],)).fetchone() is None):
                        self._diagnose(existing["result_id"], world, task, context,
                                       model or self.config.primary_model)
                    continue
                response, evaluation, prompt = self._answer(
                    world, task, context, model or self.config.primary_model,
                    purpose=f"answer:{run_id}:{architecture}:{task.task_id}:r{repeat_index}",
                )
                result_id = sha256_json([run_id, task.task_id, architecture, repeat_index])
                inserted = self.store.insert_result(
                    result_id=result_id, run_id=run_id, stage=stage, world=world, task=task,
                    architecture=architecture, repeat_index=repeat_index, evaluation=evaluation,
                    context=context, assembled_prompt=prompt, response=response,
                )
                if inserted:
                    self._append_raw({"type": "result", "result_id": result_id, "run_id": run_id,
                                      "stage": stage, "world_id": world.world_id,
                                      "task": asdict(task), "architecture": architecture,
                                      "repeat_index": repeat_index, "evaluation": asdict(evaluation),
                                      "context_trace": asdict(context), "response": response_metadata(response)})
                if not evaluation.success and repeat_index == 0:
                    self._diagnose(result_id, world, task, context, model or self.config.primary_model)

    def _answer(self, world: World, task: Task, context: ContextTrace, model: str,
                purpose: str) -> tuple[APIResult, Evaluation, str]:
        prompt = assemble_prompt(context.context, task.query)
        response = self.client.respond_json(
            model=model, instructions=ANSWER_INSTRUCTIONS, input_text=prompt,
            schema_name="memory_answer", schema=ANSWER_SCHEMA, purpose=purpose,
        )
        evaluation = evaluate(task, response.json())
        return response, evaluation, prompt

    def _diagnose(self, result_id: str, world: World, task: Task,
                  normal: ContextTrace, model: str) -> None:
        evidence = [(turn.turn_id, turn.text) for turn in world.turns
                    if turn.turn_id in task.evidence_turn_ids]
        evidence_text = MemoryBackend.render(evidence)
        combined = evidence_text
        if normal.context:
            combined = normal.context + "\n\n[ORACLE-RETRIEVED SOURCE TURNS]\n" + evidence_text
        oracle_context = ContextTrace("oracle_retrieval", task.query, combined,
                                      task.evidence_turn_ids, (), (), 0, False)
        gold_context = ContextTrace("gold_only", task.query, evidence_text,
                                    task.evidence_turn_ids, (), (), 0, False)
        oracle_response, oracle_eval, _ = self._answer(
            world, task, oracle_context, model, f"oracle:{result_id}")
        gold_response, gold_eval, _ = self._answer(
            world, task, gold_context, model, f"gold-only:{result_id}")
        evidence_selected = set(task.evidence_turn_ids).issubset(normal.source_turn_ids)
        cause, flags = classify_failure(task, evidence_selected, oracle_eval.success, gold_eval.success,
                                        normal.architecture)
        self.store.insert_diagnostic(result_id, cause, flags, oracle_eval, gold_eval,
                                     oracle_response.request_hash, gold_response.request_hash,
                                     evidence_selected)
        self._append_raw({"type": "diagnostic", "result_id": result_id, "primary_cause": cause,
                          "contributing_flags": flags, "oracle_success": oracle_eval.success,
                          "gold_only_success": gold_eval.success, "evidence_selected": evidence_selected})

    def _append_raw(self, record: dict[str, Any]) -> None:
        with self.raw_path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(record) + "\n")
            handle.flush()


def assemble_prompt(context: str, query: str) -> str:
    return f"<memory>\n{context}\n</memory>\n<query>\n{query}\n</query>"


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def evaluate(task: Task, output: dict[str, Any]) -> Evaluation:
    status = str(output.get("status", ""))
    answer = str(output.get("answer", ""))
    tool_call = None
    tool_result = None
    if status == "tool":
        try:
            arguments = json.loads(output.get("tool_arguments_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            arguments = None
        if isinstance(arguments, dict) and isinstance(output.get("tool_name"), str):
            tool_call = {"name": output["tool_name"], "arguments": arguments}
            tool_result = execute_local_tool(tool_call)
    if task.valid_tool_call is not None:
        success = tool_call == task.valid_tool_call and tool_result is not None
    elif task.unanswerable:
        success = status == "unknown" and normalize(answer) in ("", "unknown", "not available")
    elif task.forgotten:
        success = status == "forgotten" and normalize(task.answer) == "forgotten"
    else:
        success = status == "answered" and normalize(answer) == normalize(task.answer)
    return Evaluation(success, status, answer, tool_call, tool_result)


def execute_local_tool(call: dict[str, Any]) -> dict[str, Any] | None:
    name, args = call.get("name"), call.get("arguments")
    if name == "lookup_inventory" and set(args or {}) == {"sku", "warehouse"}:
        return {"available": int(sha256_json(args)[:4], 16) % 101, "simulated": True}
    if name == "run_local_diagnostic" and set(args or {}) == {"target"}:
        return {"status": "nominal", "simulated": True}
    return None


def classify_failure(task: Task, evidence_selected: bool, oracle_success: bool,
                     gold_success: bool, architecture: str) -> tuple[str, list[str]]:
    flags: list[str] = []
    if task.unanswerable or task.forgotten:
        flags.append("policy_sensitive")
    if not evidence_selected and task.evidence_turn_ids:
        flags.append("evidence_not_selected")
    if oracle_success and not evidence_selected:
        cause = "retrieval_miss"
    elif gold_success and not oracle_success:
        cause = "retrieval_contamination"
        flags.append("distractor_interference")
    elif not gold_success:
        cause = "tool_grounding_failure" if task.valid_tool_call else (
            "abstention_or_forgetting_policy_failure" if task.unanswerable or task.forgotten
            else "reasoning_failure")
    elif architecture in ("hierarchical_summary", "structured_temporal"):
        cause = "representation_or_consolidation_loss"
    elif evidence_selected:
        cause = "context_integration_failure"
    else:
        cause = "write_or_retrieval_loss"
    return cause, flags


def response_metadata(result: APIResult) -> dict[str, Any]:
    return {
        "request_hash": result.request_hash, "response_id": result.response_id,
        "requested_model": result.requested_model, "returned_model": result.returned_model,
        "usage": asdict(result.usage), "cost_usd": result.cost_usd,
        "latency_ms": result.latency_ms, "retries": result.retries,
    }
