from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import hashlib
import json


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"


@dataclass(frozen=True)
class Price:
    input_per_million: float
    cached_input_per_million: float
    output_per_million: float
    cache_write_per_million: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 20260821
    primary_model: str = "gpt-5.6-luna"
    confirmation_model: str = "gpt-5.6-terra"
    embedding_model: str = "text-embedding-3-small"
    reasoning_effort: str = "low"
    context_budget_tokens: int = 8_000
    full_context_limit_tokens: int = 200_000
    max_output_tokens: int = 300
    spend_cap_usd: float = 50.0
    pricing_snapshot_date: str = "2026-08-21"
    pricing_sources: tuple[str, ...] = (
        "https://developers.openai.com/api/docs/models/gpt-5.6-luna",
        "https://developers.openai.com/api/docs/models/text-embedding-3-small",
    )
    temporal_levels: tuple[int, ...] = (10, 100, 1_000)
    interference_levels: tuple[int, ...] = (10, 100, 1_000)
    contradiction_levels: tuple[int, ...] = (0, 1, 3)
    worlds_per_screening_profile: int = 5
    worlds_per_expansion_profile: int = 5
    architectures: tuple[str, ...] = (
        "query_only",
        "full_context",
        "recency",
        "vector_rag",
        "hierarchical_summary",
        "structured_temporal",
    )
    prices: dict[str, Price] = field(default_factory=lambda: {
        # Luna rates are pinned from the research protocol's 2026-08-21 snapshot.
        "gpt-5.6-luna": Price(0.20, 0.02, 1.20),
        "text-embedding-3-small": Price(0.02, 0.0, 0.0),
    })

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        raw = json.dumps(self.public_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()


L9_PROFILES: tuple[tuple[int, int, int], ...] = (
    (10, 10, 0),
    (10, 100, 1),
    (10, 1_000, 3),
    (100, 10, 1),
    (100, 100, 3),
    (100, 1_000, 0),
    (1_000, 10, 3),
    (1_000, 100, 0),
    (1_000, 1_000, 1),
)


TASK_TYPES: tuple[str, ...] = (
    "atomic_recall",
    "current_state",
    "historical_value",
    "coexisting_facts",
    "conditional_fact",
    "temporal_ordering",
    "two_hop_reasoning",
    "four_hop_reasoning",
    "selective_forgetting",
    "abstention",
    "tool_argument_grounding",
    "tool_selection",
)
