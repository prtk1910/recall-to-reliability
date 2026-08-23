from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any
import hashlib
import json
import os


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"


@dataclass(frozen=True)
class Price:
    input_per_million: float
    cached_input_per_million: float
    output_per_million: float
    cache_write_per_million: float = 0.0
    request_usd: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 20260821
    provider: str = "openai"
    api_base_url: str = "https://api.openai.com/v1"
    api_protocol: str = "responses"
    api_key_env: str = "OPENAI_API_KEY"
    app_url: str = "https://github.com/prtk1910/recall-to-reliability"
    app_title: str = "Recall to Reliability"
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

    @classmethod
    def from_env(cls) -> "ExperimentConfig":
        """Build a reproducible provider configuration from documented environment variables."""
        base = cls()
        provider = _env_value("RELIABMEM_PROVIDER", base.provider).casefold()
        if provider not in {"openai", "openrouter"}:
            raise ValueError("RELIABMEM_PROVIDER must be 'openai' or 'openrouter'")

        if provider == "openrouter":
            default_model = f"openai/{base.primary_model}"
            default_embedding = f"openai/{base.embedding_model}"
            configured = replace(
                base,
                provider=provider,
                api_base_url="https://openrouter.ai/api/v1",
                api_protocol="chat_completions",
                api_key_env="OPENROUTER_API_KEY",
                pricing_snapshot_date="runtime-catalog",
                pricing_sources=(
                    "https://openrouter.ai/docs/api/api-reference/models/get-model",
                    "https://openrouter.ai/docs/cookbook/administration/usage-accounting",
                ),
                primary_model=_env_value("RELIABMEM_MODEL", default_model),
                confirmation_model=_env_value("RELIABMEM_CONFIRMATION_MODEL", ""),
                embedding_model=_env_value(
                    "RELIABMEM_EMBEDDING_MODEL", default_embedding,
                ),
                reasoning_effort=_env_value(
                    "RELIABMEM_REASONING_EFFORT", base.reasoning_effort,
                ),
                app_url=_env_value("OPENROUTER_SITE_URL", base.app_url),
                app_title=_env_value("OPENROUTER_APP_TITLE", base.app_title),
            )
        else:
            configured = replace(
                base,
                primary_model=_env_value("RELIABMEM_MODEL", base.primary_model),
                confirmation_model=_env_value(
                    "RELIABMEM_CONFIRMATION_MODEL", base.confirmation_model,
                ),
                embedding_model=_env_value(
                    "RELIABMEM_EMBEDDING_MODEL", base.embedding_model,
                ),
                reasoning_effort=_env_value(
                    "RELIABMEM_REASONING_EFFORT", base.reasoning_effort,
                ),
            )

        base_url = _env_value("RELIABMEM_BASE_URL", configured.api_base_url).rstrip("/")
        if not configured.primary_model:
            raise ValueError("RELIABMEM_MODEL must not be empty")
        if not configured.embedding_model:
            raise ValueError("RELIABMEM_EMBEDDING_MODEL must not be empty")
        if not base_url:
            raise ValueError("RELIABMEM_BASE_URL must not be empty")
        prices = dict(configured.prices)
        if provider == "openrouter":
            prices.setdefault(
                "openai/text-embedding-3-small", Price(0.02, 0.02, 0.0),
            )
        custom_price = _price_from_env()
        if custom_price is not None:
            prices[configured.primary_model] = custom_price
        embedding_price = _price_from_env("RELIABMEM_EMBEDDING_")
        if embedding_price is not None:
            prices[configured.embedding_model] = embedding_price
        return replace(configured, api_base_url=base_url, prices=prices)


def _price_from_env(prefix: str = "RELIABMEM_") -> Price | None:
    input_value = _optional_env_value(f"{prefix}INPUT_PRICE_PER_MILLION")
    output_value = _optional_env_value(f"{prefix}OUTPUT_PRICE_PER_MILLION")
    if input_value is None and output_value is None:
        return None
    if input_value is None or output_value is None:
        raise ValueError(
            f"{prefix}INPUT_PRICE_PER_MILLION and {prefix}OUTPUT_PRICE_PER_MILLION "
            "must be set together"
        )
    input_price = float(input_value)
    output_price = float(output_value)
    cached_price = float(_env_value(
        f"{prefix}CACHED_INPUT_PRICE_PER_MILLION", str(input_price),
    ))
    cache_write_price = float(_env_value(
        f"{prefix}CACHE_WRITE_PRICE_PER_MILLION", str(input_price),
    ))
    request_price = float(_env_value(f"{prefix}REQUEST_PRICE_USD", "0"))
    if min(input_price, output_price, cached_price, cache_write_price, request_price) < 0:
        raise ValueError("model prices must be non-negative")
    return Price(input_price, cached_price, output_price, cache_write_price, request_price)


def _optional_env_value(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None


def _env_value(name: str, default: str) -> str:
    return _optional_env_value(name) or default


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
