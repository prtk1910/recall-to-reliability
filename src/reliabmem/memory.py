from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol
import math
import re
import sqlite3

from .benchmark import Task, Turn
from .config import ExperimentConfig
from .util import estimate_tokens, pack_chronological, sha256_json


class ClientLike(Protocol):
    def respond_json(self, **kwargs: Any) -> Any: ...
    def embed(self, texts: list[str], purpose: str) -> tuple[list[list[float]], Any]: ...


@dataclass(frozen=True)
class WriteTrace:
    architecture: str
    turn_id: int
    operation: str
    memory_ids: tuple[str, ...]


@dataclass(frozen=True)
class ContextTrace:
    architecture: str
    query: str
    context: str
    source_turn_ids: tuple[int, ...]
    candidate_ids: tuple[str, ...]
    selected_ids: tuple[str, ...]
    token_count: int
    truncated: bool
    operations: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class MemorySnapshot:
    architecture: str
    world_id: str
    observed_turns: int
    memory_items: int
    token_count: int


class MemoryBackend:
    name = "base"

    def __init__(self, config: ExperimentConfig, client: ClientLike | None = None,
                 state_dir: Path | None = None):
        self.config = config
        self.client = client
        self.state_dir = state_dir
        self.world_id = ""
        self.turns: list[Turn] = []

    def reset(self, world_id: str) -> None:
        self.world_id = world_id
        self.turns = []

    def observe(self, turn: Turn) -> WriteTrace:
        self.turns.append(turn)
        return WriteTrace(self.name, turn.turn_id, "append", (f"turn:{turn.turn_id}",))

    def build_context(self, task: Task) -> ContextTrace:
        raise NotImplementedError

    def snapshot(self) -> MemorySnapshot:
        return MemorySnapshot(self.name, self.world_id, len(self.turns), len(self.turns),
                              sum(estimate_tokens(t.text) for t in self.turns))

    @staticmethod
    def render(items: list[tuple[int, str]]) -> str:
        return "\n".join(f"[turn {turn_id}] {text}" for turn_id, text in items)


class QueryOnlyMemory(MemoryBackend):
    name = "query_only"

    def build_context(self, task: Task) -> ContextTrace:
        return ContextTrace(self.name, task.query, "", (), (), (), 0, False)


class FullContextMemory(MemoryBackend):
    name = "full_context"

    def build_context(self, task: Task) -> ContextTrace:
        items = [(turn.turn_id, turn.text) for turn in self.turns]
        context = self.render(items)
        tokens = estimate_tokens(context)
        if tokens >= self.config.full_context_limit_tokens:
            raise ValueError("full context exceeds configured request ceiling")
        ids = tuple(f"turn:{turn_id}" for turn_id, _ in items)
        return ContextTrace(self.name, task.query, context, tuple(i for i, _ in items), ids, ids,
                            tokens, False)


class RecencyMemory(MemoryBackend):
    name = "recency"

    def build_context(self, task: Task) -> ContextTrace:
        all_items = [(turn.turn_id, turn.text) for turn in self.turns]
        selected: list[tuple[int, str]] = []
        for item in reversed(all_items):
            candidate = [item] + selected
            if estimate_tokens(self.render(candidate)) <= self.config.context_budget_tokens:
                selected = candidate
        if not selected and all_items:
            turn_id, value = all_items[-1]
            low, high = 0, len(value)
            while low < high:
                middle = (low + high + 1) // 2
                if estimate_tokens(self.render([(turn_id, value[-middle:])])) <= self.config.context_budget_tokens:
                    low = middle
                else:
                    high = middle - 1
            selected = [(turn_id, value[-low:])] if low else []
        context = self.render(selected)
        ids = tuple(f"turn:{turn_id}" for turn_id, _ in selected)
        return ContextTrace(self.name, task.query, context, tuple(i for i, _ in selected),
                            tuple(f"turn:{i}" for i, _ in all_items), ids,
                            estimate_tokens(context), len(selected) < len(all_items))


@dataclass
class VectorChunk:
    chunk_id: str
    source_turn_ids: tuple[int, ...]
    text: str
    vector: list[float] | None = None


class VectorRAGMemory(MemoryBackend):
    name = "vector_rag"

    def reset(self, world_id: str) -> None:
        super().reset(world_id)
        self.chunks: list[VectorChunk] = []
        self._indexed = False

    def _make_chunks(self) -> list[VectorChunk]:
        chunks: list[VectorChunk] = []
        pending: list[Turn] = []
        pending_tokens = 0
        for turn in self.turns:
            cost = estimate_tokens(turn.text)
            if pending and (len(pending) >= 6 or pending_tokens + cost > 768):
                chunks.append(self._chunk(pending))
                pending, pending_tokens = [], 0
            pending.append(turn)
            pending_tokens += cost
        if pending:
            chunks.append(self._chunk(pending))
        return chunks

    def _chunk(self, turns: list[Turn]) -> VectorChunk:
        ids = tuple(turn.turn_id for turn in turns)
        text = self.render([(turn.turn_id, turn.text) for turn in turns])
        return VectorChunk(f"chunk:{ids[0]}-{ids[-1]}", ids, text)

    def _index(self) -> None:
        if self._indexed:
            return
        if self.client is None:
            raise RuntimeError("vector RAG requires an embedding client")
        self.chunks = self._make_chunks()
        for start in range(0, len(self.chunks), 128):
            batch = self.chunks[start:start + 128]
            vectors, _ = self.client.embed([chunk.text for chunk in batch],
                purpose=f"embedding:{self.world_id}:{start}")
            for chunk, vector in zip(batch, vectors, strict=True):
                chunk.vector = vector
        self._indexed = True

    def build_context(self, task: Task) -> ContextTrace:
        self._index()
        assert self.client is not None
        vectors, _ = self.client.embed([task.query], purpose=f"query-embedding:{self.world_id}:{sha256_json(task.query)[:12]}")
        query_vector = vectors[0]
        ranked = sorted(self.chunks, key=lambda chunk: (-cosine(query_vector, chunk.vector or []),
                                                        chunk.source_turn_ids[0]))
        selected: list[VectorChunk] = []
        used = 0
        for chunk in ranked:
            cost = estimate_tokens(chunk.text)
            if used + cost <= self.config.context_budget_tokens:
                selected.append(chunk)
                used += cost
        selected.sort(key=lambda chunk: chunk.source_turn_ids[0])
        context = "\n".join(chunk.text for chunk in selected)
        source_ids = tuple(i for chunk in selected for i in chunk.source_turn_ids)
        return ContextTrace(self.name, task.query, context, source_ids,
            tuple(chunk.chunk_id for chunk in ranked), tuple(chunk.chunk_id for chunk in selected),
            estimate_tokens(context), len(selected) < len(ranked),
            ({"operation": "cosine_retrieval", "candidate_count": len(ranked),
              "selected_count": len(selected)},))

    def snapshot(self) -> MemorySnapshot:
        chunks = self.chunks if self._indexed else self._make_chunks()
        return MemorySnapshot(self.name, self.world_id, len(self.turns), len(chunks),
                              sum(estimate_tokens(c.text) for c in chunks))


def cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return -1.0
    dot = sum(a * b for a, b in zip(left, right))
    lnorm = math.sqrt(sum(a * a for a in left))
    rnorm = math.sqrt(sum(b * b for b in right))
    return dot / (lnorm * rnorm) if lnorm and rnorm else -1.0


@dataclass
class SummaryNode:
    node_id: str
    level: int
    source_turn_ids: tuple[int, ...]
    text: str


SUMMARY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "source_turn_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["summary", "source_turn_ids"],
}


class HierarchicalSummaryMemory(MemoryBackend):
    name = "hierarchical_summary"

    def reset(self, world_id: str) -> None:
        super().reset(world_id)
        self.frontier: list[SummaryNode] = []
        self._summarized = False
        self.operations: list[dict[str, Any]] = []

    def _summarize(self, nodes: list[SummaryNode] | None, turns: list[Turn] | None,
                   level: int, ordinal: int) -> SummaryNode:
        if self.client is None:
            raise RuntimeError("hierarchical summaries require a response client")
        if turns is not None:
            source_ids = tuple(t.turn_id for t in turns)
            text = self.render([(t.turn_id, t.text) for t in turns])
        else:
            assert nodes is not None
            source_ids = tuple(i for node in nodes for i in node.source_turn_ids)
            text = "\n".join(f"[{node.node_id}] {node.text}" for node in nodes)
        result = self.client.respond_json(
            model=self.config.primary_model,
            instructions=("Faithfully compress memory. Preserve entities, values, update order, "
                          "conditions, explicit forgetting, tool names/arguments, and source turn IDs."),
            input_text=text, schema_name="memory_summary", schema=SUMMARY_SCHEMA,
            purpose=f"summary:{self.world_id}:L{level}:{ordinal}", max_output_tokens=500,
        )
        parsed = result.json()
        returned_ids = tuple(int(i) for i in parsed.get("source_turn_ids", []) if int(i) in source_ids)
        node = SummaryNode(f"summary:L{level}:{ordinal}", level, returned_ids or source_ids,
                           str(parsed["summary"]))
        self.operations.append({"operation": "summarize" if level == 0 else "consolidate",
                                "node_id": node.node_id, "source_turn_ids": node.source_turn_ids,
                                "request_hash": result.request_hash})
        return node

    def _build(self) -> None:
        if self._summarized:
            return
        level = [self._summarize(None, self.turns[start:start + 10], 0, start // 10)
                 for start in range(0, len(self.turns), 10)]
        depth = 1
        while len(level) > 10:
            level = [self._summarize(level[start:start + 10], None, depth, start // 10)
                     for start in range(0, len(level), 10)]
            depth += 1
        self.frontier = level
        self._summarized = True

    def build_context(self, task: Task) -> ContextTrace:
        self._build()
        items = [(node.source_turn_ids[-1], f"[{node.node_id}] {node.text}") for node in self.frontier]
        packed = pack_chronological(items, self.config.context_budget_tokens)
        selected_ends = {turn_id for turn_id, _ in packed}
        selected = [node for node in self.frontier if node.source_turn_ids[-1] in selected_ends]
        context = "\n".join(text for _, text in packed)
        return ContextTrace(self.name, task.query, context,
            tuple(i for node in selected for i in node.source_turn_ids),
            tuple(node.node_id for node in self.frontier), tuple(node.node_id for node in selected),
            estimate_tokens(context), len(selected) < len(self.frontier), tuple(self.operations))

    def snapshot(self) -> MemorySnapshot:
        self._build()
        return MemorySnapshot(self.name, self.world_id, len(self.turns), len(self.frontier),
                              sum(estimate_tokens(node.text) for node in self.frontier))


FACT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"facts": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "entity": {"type": "string"}, "attribute": {"type": "string"},
            "value": {"type": ["string", "null"]},
            "status": {"type": "string", "enum": ["current", "historical", "forgotten"]},
            "source_turn_ids": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["entity", "attribute", "value", "status", "source_turn_ids"],
    }}},
    "required": ["facts"],
}


class StructuredTemporalMemory(MemoryBackend):
    name = "structured_temporal"

    def reset(self, world_id: str) -> None:
        super().reset(world_id)
        self._extracted = False
        self.operations: list[dict[str, Any]] = []
        if self.state_dir is None:
            self.db = sqlite3.connect(":memory:")
        else:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(self.state_dir / "structured_memory.sqlite")
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS facts (
            world_id TEXT NOT NULL, fact_id TEXT NOT NULL, entity TEXT NOT NULL,
            attribute TEXT NOT NULL, value TEXT, status TEXT NOT NULL,
            source_turn_ids TEXT NOT NULL, version INTEGER NOT NULL,
            PRIMARY KEY(world_id, fact_id)
          );
        """)
        self.db.execute("DELETE FROM facts WHERE world_id=?", (world_id,))
        self.db.commit()

    def _extract(self) -> None:
        if self._extracted:
            return
        if self.client is None:
            raise RuntimeError("structured memory requires a response client")
        for start in range(0, len(self.turns), 10):
            batch = self.turns[start:start + 10]
            result = self.client.respond_json(
                model=self.config.primary_model,
                instructions=("Extract only explicit durable facts, updates, forgetting directives, "
                              "conditions, and tool specifications. Resolve updates by marking older "
                              "facts historical and deletions forgotten. Copy source turn IDs."),
                input_text=self.render([(t.turn_id, t.text) for t in batch]),
                schema_name="temporal_facts", schema=FACT_SCHEMA,
                purpose=f"fact-extraction:{self.world_id}:{start}", max_output_tokens=700,
            )
            valid_ids = {t.turn_id for t in batch}
            for fact in result.json().get("facts", []):
                ids = tuple(int(i) for i in fact["source_turn_ids"] if int(i) in valid_ids)
                if not ids:
                    continue
                self._upsert_fact(str(fact["entity"]), str(fact["attribute"]), fact["value"],
                                  str(fact["status"]), ids)
            self.operations.append({"operation": "extract", "turn_range": [batch[0].turn_id, batch[-1].turn_id],
                                    "request_hash": result.request_hash})
        self._extracted = True

    def _upsert_fact(self, entity: str, attribute: str, value: str | None,
                     status: str, source_ids: tuple[int, ...]) -> None:
        current = self.db.execute(
            "SELECT MAX(version) FROM facts WHERE world_id=? AND lower(entity)=lower(?) AND lower(attribute)=lower(?)",
            (self.world_id, entity, attribute),
        ).fetchone()[0]
        version = int(current if current is not None else -1) + 1
        if status == "current":
            self.db.execute(
                "UPDATE facts SET status='historical' WHERE world_id=? AND lower(entity)=lower(?) "
                "AND lower(attribute)=lower(?) AND status='current'", (self.world_id, entity, attribute))
        if status == "forgotten":
            self.db.execute(
                "UPDATE facts SET status='forgotten' WHERE world_id=? AND lower(entity)=lower(?) "
                "AND lower(attribute)=lower(?)", (self.world_id, entity, attribute))
        fact_id = sha256_json([entity, attribute, value, status, source_ids])[:20]
        self.db.execute(
            "INSERT OR REPLACE INTO facts VALUES (?,?,?,?,?,?,?,?)",
            (self.world_id, fact_id, entity, attribute, value, status,
             ",".join(map(str, source_ids)), version),
        )
        self.db.commit()

    def build_context(self, task: Task) -> ContextTrace:
        self._extract()
        rows = list(self.db.execute("SELECT * FROM facts WHERE world_id=? ORDER BY version", (self.world_id,)))
        query_terms = set(re.findall(r"[a-z0-9_-]+", task.query.casefold()))
        def score(row: sqlite3.Row) -> tuple[int, int]:
            text = f"{row['entity']} {row['attribute']} {row['value'] or ''}".casefold()
            overlap = len(query_terms & set(re.findall(r"[a-z0-9_-]+", text)))
            intent = 2 if ("first" in query_terms and row["version"] == 0) else 0
            intent += 2 if ("current" in query_terms and row["status"] == "current") else 0
            intent += 2 if row["status"] == "forgotten" else 0
            return overlap + intent, int(row["version"])
        ranked = sorted(rows, key=lambda row: (-score(row)[0], score(row)[1]))
        selected: list[sqlite3.Row] = []
        used = 0
        for row in ranked:
            text = render_fact(row)
            cost = estimate_tokens(text)
            if used + cost <= self.config.context_budget_tokens:
                selected.append(row)
                used += cost
        selected.sort(key=lambda row: (row["entity"], row["attribute"], row["version"]))
        context = "\n".join(render_fact(row) for row in selected)
        source_ids = tuple(sorted({int(i) for row in selected for i in row["source_turn_ids"].split(",") if i}))
        return ContextTrace(self.name, task.query, context, source_ids,
            tuple(row["fact_id"] for row in ranked), tuple(row["fact_id"] for row in selected),
            estimate_tokens(context), len(selected) < len(ranked), tuple(self.operations))

    def snapshot(self) -> MemorySnapshot:
        self._extract()
        rows = list(self.db.execute("SELECT * FROM facts WHERE world_id=?", (self.world_id,)))
        return MemorySnapshot(self.name, self.world_id, len(self.turns), len(rows),
                              sum(estimate_tokens(render_fact(row)) for row in rows))


def render_fact(row: sqlite3.Row) -> str:
    return (f"[fact {row['fact_id']} v{row['version']} status={row['status']} "
            f"sources={row['source_turn_ids']}] {row['entity']}.{row['attribute']} = {row['value']}")


BACKENDS = {
    cls.name: cls for cls in (
        QueryOnlyMemory, FullContextMemory, RecencyMemory, VectorRAGMemory,
        HierarchicalSummaryMemory, StructuredTemporalMemory,
    )
}


def create_backend(name: str, config: ExperimentConfig, client: ClientLike | None,
                   state_dir: Path | None = None) -> MemoryBackend:
    try:
        return BACKENDS[name](config, client, state_dir)
    except KeyError as exc:
        raise ValueError(f"unknown memory architecture: {name}") from exc
