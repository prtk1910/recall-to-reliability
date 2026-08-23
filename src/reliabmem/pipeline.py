from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import argparse
import json
import os
import shutil
import sqlite3

from .analysis import analyze, select_interaction
from .api import APILedger, BudgetExceeded, OpenAIClient
from .benchmark import StressProfile, World, generate_world, save_worlds
from .config import ARTIFACTS, L9_PROFILES, ROOT, ExperimentConfig
from .harness import ExperimentHarness, ResultStore
from .report import generate_report
from .util import atomic_write_text, code_revision, estimate_tokens, sha256_file, sha256_json


def pilot_worlds(config: ExperimentConfig) -> list[World]:
    return [
        generate_world(StressProfile(10, 10, 0), config.seed + 1),
        generate_world(StressProfile(1_000, 1_000, 3), config.seed + 2),
    ]


def screening_worlds(config: ExperimentConfig) -> list[World]:
    worlds = []
    for profile_index, values in enumerate(L9_PROFILES):
        profile = StressProfile(*values)
        for world_index in range(config.worlds_per_screening_profile):
            worlds.append(generate_world(profile, config.seed + 1_000 + profile_index * 10 + world_index))
    return worlds


def expansion_worlds(config: ExperimentConfig, factor_a: str, factor_b: str,
                     count: int) -> list[World]:
    levels = {"temporal_distance": (10, 100, 1_000), "interference": (10, 100, 1_000),
              "contradictions": (0, 1, 3)}
    base = {"temporal_distance": 100, "interference": 100, "contradictions": 1}
    worlds = []
    cell = 0
    for left in levels[factor_a]:
        for right in levels[factor_b]:
            values = {**base, factor_a: left, factor_b: right}
            profile = StressProfile(**values)
            for world_index in range(count):
                worlds.append(generate_world(profile, config.seed + 3_000 + cell * 10 + world_index))
            cell += 1
    return worlds


def terra_worlds(config: ExperimentConfig) -> list[World]:
    profiles = (StressProfile(10, 10, 0), StressProfile(100, 100, 1),
                StressProfile(1_000, 1_000, 3))
    return [generate_world(profile, config.seed + 7_000 + pidx * 10 + world_index)
            for pidx, profile in enumerate(profiles) for world_index in range(3)]


def generate_benchmark(config: ExperimentConfig) -> dict[str, Path]:
    directory = ARTIFACTS / "benchmark"
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"pilot": directory / "pilot.jsonl", "screening": directory / "screening.jsonl"}
    save_worlds(paths["pilot"], pilot_worlds(config))
    save_worlds(paths["screening"], screening_worlds(config))
    manifest = {
        "config": config.public_dict(), "config_hash": config.hash(),
        "code_revision": code_revision(ROOT),
        "files": {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths.values()},
    }
    atomic_write_text(directory / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    atomic_write_text(ARTIFACTS / "config.json", json.dumps(config.public_dict(), indent=2, sort_keys=True) + "\n")
    return paths


def build_runtime(config: ExperimentConfig) -> tuple[APILedger, OpenAIClient, ResultStore, ExperimentHarness]:
    db_path = ARTIFACTS / "results.sqlite"
    ledger = APILedger(db_path, config.spend_cap_usd)
    client = OpenAIClient(config, ledger)
    store = ResultStore(db_path)
    harness = ExperimentHarness(config, client, store, ARTIFACTS)
    return ledger, client, store, harness


def live_smoke(config: ExperimentConfig, client: OpenAIClient) -> None:
    result = client.respond_json(
        model=config.primary_model,
        instructions="Return the requested fixed JSON fields. This is a connectivity smoke test.",
        input_text="Return status unknown, an empty answer, and null tool fields.",
        schema_name="smoke", schema={
            "type": "object", "additionalProperties": False,
            "properties": {"status": {"type": "string"}, "answer": {"type": "string"},
                           "tool_name": {"type": ["string", "null"]},
                           "tool_arguments_json": {"type": ["string", "null"]}},
            "required": ["status", "answer", "tool_name", "tool_arguments_json"],
        }, purpose="live-smoke", max_output_tokens=80,
    )
    parsed = result.json()
    if parsed.get("status") != "unknown":
        raise RuntimeError("live smoke returned an unexpected structured response")


def run_stage(config: ExperimentConfig, store: ResultStore, harness: ExperimentHarness,
              stage: str, worlds: list[World], model: str | None = None,
              repeat_indices: tuple[int, ...] = (0,),
              selections: dict[tuple[str, str], set[str]] | None = None) -> str:
    model = model or config.primary_model
    run_id = f"{stage}-{sha256_json([config.hash(), model])[:16]}"
    completed = store.db.execute("SELECT completed_at FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if completed and completed[0]:
        return run_id
    store.start_run(run_id, stage, config, code_revision(ROOT), model)
    expected = 0
    for world in worlds:
        store.register_world(world, estimate_tokens("\n".join(turn.text for turn in world.turns)))
        for architecture in config.architectures:
            task_ids = selections.get((world.world_id, architecture)) if selections is not None else None
            if selections is not None and not task_ids:
                continue
            expected += (len(task_ids) if task_ids is not None else len(world.tasks)) * len(repeat_indices)
            harness.run_world(run_id, stage, world, architecture, model=model,
                              repeat_indices=repeat_indices, task_ids=task_ids)
    observed = store.db.execute("SELECT COUNT(*) FROM results WHERE run_id=?", (run_id,)).fetchone()[0]
    if observed != expected:
        raise RuntimeError(f"stage {stage} produced {observed} results; expected {expected}")
    store.complete_run(run_id)
    return run_id


def forecast_cost(pilot_cost: float, pilot: list[World], future: list[World]) -> float:
    pilot_units = sum(len(world.turns) + 1_000 for world in pilot)
    future_units = sum(len(world.turns) + 1_000 for world in future)
    if pilot_cost <= 0 or pilot_units <= 0:
        return 0.0
    return pilot_cost * future_units / pilot_units * 1.25


def choose_optional_stages(config: ExperimentConfig, ledger: APILedger,
                           pilot_cost: float, pilot: list[World], expansion5: list[World],
                           expansion3: list[World], screening: list[World], terra: list[World]) -> dict[str, Any]:
    spent = ledger.totals()["actual"]
    screen_forecast = forecast_cost(pilot_cost, pilot, screening)
    expansion_count = 5
    expansion_forecast = forecast_cost(pilot_cost, pilot, expansion5)
    variance_forecast = max(0.01, screen_forecast * 0.05)
    terra_priced = config.confirmation_model in config.prices
    terra_forecast = forecast_cost(pilot_cost, pilot, terra) * 8.0 if terra_priced else 0.0
    decisions = {
        "config_hash": config.hash(),
        "external_sanity_check": False,
        "external_reason": "omitted: no repository-pinned, checksum-verified 60-question artifact",
        "terra_check": terra_priced, "variance_audit": True, "expansion_worlds_per_cell": 5,
        "pilot_actual_usd": pilot_cost, "screening_forecast_usd": screen_forecast,
        "expansion_forecast_usd": expansion_forecast, "terra_forecast_usd": terra_forecast,
        "variance_forecast_usd": variance_forecast,
    }
    if not terra_priced:
        decisions["terra_reason"] = "omitted: official Terra pricing was not established in the pinned snapshot"
    projected = spent + screen_forecast + expansion_forecast + variance_forecast + terra_forecast
    if projected > config.spend_cap_usd:
        decisions["terra_check"] = False
        decisions["terra_reason"] = "omitted by spend forecast after external check"
        projected -= terra_forecast
    if projected > config.spend_cap_usd:
        decisions["variance_audit"] = False
        decisions["variance_reason"] = "extra deterministic repeats omitted by spend forecast"
        projected -= variance_forecast
    if projected > config.spend_cap_usd:
        decisions["expansion_worlds_per_cell"] = 3
        decisions["expansion_reason"] = "reduced from five to three only after optional stages"
        projected -= expansion_forecast
        expansion_forecast = forecast_cost(pilot_cost, pilot, expansion3)
        decisions["expansion_forecast_usd"] = expansion_forecast
        projected += expansion_forecast
    decisions["projected_total_usd"] = projected
    if spent + screen_forecast + expansion_forecast > config.spend_cap_usd:
        raise BudgetExceeded(
            "pilot forecast says the fixed L9 screening core plus minimum expansion cannot fit "
            f"the ${config.spend_cap_usd:.2f} cap; refusing to silently shrink the core"
        )
    return decisions


def variance_selections(config: ExperimentConfig, store: ResultStore) -> tuple[list[World], dict[tuple[str, str], set[str]]]:
    store.db.row_factory = sqlite3.Row
    rows = list(store.db.execute(
        """SELECT r.world_id,r.architecture,r.task_id,w.seed,w.temporal_distance,w.interference,w.contradictions
           FROM results r JOIN worlds w USING(world_id) WHERE r.stage='screening' AND r.repeat_index=0"""))
    selected = [row for row in rows if int(sha256_json([row["world_id"], row["architecture"], row["task_id"]])[:8], 16) % 10 == 0]
    selection_map: dict[tuple[str, str], set[str]] = {}
    for row in selected:
        selection_map.setdefault((row["world_id"], row["architecture"]), set()).add(row["task_id"])
    worlds_by_id = {world.world_id: world for world in screening_worlds(config)}
    # Reconstruct from the immutable seed/profile columns if a non-default config is used.
    worlds = []
    for world_id in sorted({row["world_id"] for row in selected}):
        if world_id in worlds_by_id:
            worlds.append(worlds_by_id[world_id])
            continue
        row = next(r for r in selected if r["world_id"] == world_id)
        worlds.append(generate_world(StressProfile(row["temporal_distance"], row["interference"],
                                                   row["contradictions"]), row["seed"]))
    return worlds, selection_map


def paper(config: ExperimentConfig) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required")
    generate_benchmark(config)
    ledger, client, store, harness = build_runtime(config)
    live_smoke(config, client)
    pilot = pilot_worlds(config)
    run_stage(config, store, harness, "pilot", pilot)
    smoke_cost = store.db.execute(
        "SELECT COALESCE(SUM(actual_cost_usd),0) FROM api_calls WHERE purpose='live-smoke'"
    ).fetchone()[0]
    pilot_cost = max(0.0, ledger.totals()["actual"] - float(smoke_cost or 0))
    screening = screening_worlds(config)
    initial_expansion5 = expansion_worlds(config, "temporal_distance", "contradictions", 5)
    initial_expansion3 = expansion_worlds(config, "temporal_distance", "contradictions", 3)
    terra = terra_worlds(config)
    decisions_path = ARTIFACTS / "stage_decisions.json"
    existing_decisions = json.loads(decisions_path.read_text()) if decisions_path.exists() else None
    if existing_decisions and existing_decisions.get("config_hash") == config.hash():
        decisions = existing_decisions
    else:
        decisions = choose_optional_stages(config, ledger, pilot_cost, pilot, initial_expansion5,
                                           initial_expansion3, screening, terra)
        atomic_write_text(decisions_path, json.dumps(decisions, indent=2, sort_keys=True) + "\n")
    run_stage(config, store, harness, "screening", screening)
    screening_rows = [dict(row) for row in store.db.execute(
        """SELECT r.*,t.task_type,t.hop_count,w.temporal_distance,w.interference,w.contradictions
           FROM results r JOIN tasks t USING(task_id) JOIN worlds w USING(world_id)
           WHERE r.stage='screening' AND r.repeat_index=0""")]
    interaction = select_interaction(screening_rows, config.seed)
    expansion = expansion_worlds(config, interaction["factor_a"], interaction["factor_b"],
                                 decisions["expansion_worlds_per_cell"])
    expansion_path = ARTIFACTS / "benchmark" / "expansion.jsonl"
    save_worlds(expansion_path, expansion)
    run_stage(config, store, harness, "expansion", expansion)
    if decisions["variance_audit"]:
        variance_worlds, selections = variance_selections(config, store)
        run_stage(config, store, harness, "variance", variance_worlds,
                  repeat_indices=(0, 1, 2), selections=selections)
    if decisions["terra_check"]:
        run_stage(config, store, harness, "terra_confirmation", terra,
                  model=config.confirmation_model)
    analysis = analyze(ARTIFACTS / "results.sqlite", ARTIFACTS / "tables", config.seed)
    run_id = f"paper-{sha256_json([config.hash(), code_revision(ROOT)])[:16]}"
    generate_report(ROOT, ARTIFACTS / "results.sqlite", analysis, config, run_id)


def clean() -> None:
    if ARTIFACTS.exists():
        for child in ARTIFACTS.iterdir():
            if child.name == ".gitkeep":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("paper", "generate", "smoke", "clean"))
    args = parser.parse_args(argv)
    config = ExperimentConfig()
    if args.command == "generate":
        paths = generate_benchmark(config)
        print(f"generated {len(paths)} benchmark stages under {ARTIFACTS / 'benchmark'}")
    elif args.command == "smoke":
        _, client, _, _ = build_runtime(config)
        live_smoke(config, client)
        print("live smoke passed")
    elif args.command == "clean":
        clean()
        print("removed generated artifacts")
    else:
        paper(config)
        print(f"paper generated at {ROOT / 'PAPER.md'}")


if __name__ == "__main__":
    main()
